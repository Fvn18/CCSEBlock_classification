import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from torch import Tensor


def get_rng_device(device: Union[str, torch.device]) -> str:
    device_str = device if isinstance(device, str) else str(device)
    device_map = {
        'mps': 'cpu',
        'cuda': device_str,
        'cpu': 'cpu'
    }
    return device_map.get(device_str, 'cpu')


def set_seed(seed: int = 33) -> None:
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(True, warn_only=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        import torch.multiprocessing as mp
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn', force=True)
    except ImportError:
        pass


def worker_init_fn(worker_id: int, base_seed: int = 42) -> None:
    import os

    worker_seed = base_seed + worker_id

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    os.environ['PYTHONHASHSEED'] = str(worker_seed)


class FocalLoss(nn.Module):

    def __init__(self, gamma: float = 2.0, alpha: Optional[Union[float, list, Tensor]] = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            self.register_buffer('alpha', torch.tensor([alpha, 1-alpha]))
        elif isinstance(alpha, list):
            self.register_buffer('alpha', torch.tensor(alpha))
        else:
            self.register_buffer('alpha', alpha)

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        logpt = F.log_softmax(inputs, dim=1)
        pt = logpt.exp()

        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
            inputs = inputs.transpose(1, 2)
            inputs = inputs.contiguous().view(-1, inputs.size(2))
            target = target.view(-1)

        target = target.unsqueeze(1) if target.dim() == 1 else target
        pt = pt.gather(1, target).squeeze(1)
        logpt = logpt.gather(1, target).squeeze(1)

        loss = - (1 - pt).pow(self.gamma) * logpt

        if self.alpha is not None:
            if self.alpha.device != loss.device:
                self.alpha = self.alpha.to(loss.device)
            alpha_t = self.alpha.gather(0, target.squeeze(1) if target.dim() > 1 else target)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class GeneralizedCrossEntropy(nn.Module):

    def __init__(self, num_classes: int, q: float = 0.7, reduction: str = 'mean'):
        super().__init__()
        self.num_classes = num_classes
        self.q = q
        self.reduction = reduction

    def forward(self, pred: Tensor, labels: Tensor) -> Tensor:
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        gce = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q

        if self.reduction == 'mean':
            return gce.mean()
        elif self.reduction == 'sum':
            return gce.sum()
        else:
            return gce


class SymmetricCrossEntropy(nn.Module):

    def __init__(self, num_classes: int, alpha: float = 0.1, beta: float = 1.0, class_weights: Optional[Tensor] = None,
                 ce_only_epochs: int = 8, label_smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.ce_only_epochs = ce_only_epochs
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.class_weights = class_weights
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction=reduction, weight=self.class_weights)

    def forward(self, pred: Tensor, labels: Tensor, epoch: Optional[int] = None) -> Tensor:
        if epoch is not None and epoch < self.ce_only_epochs:
            return self.cross_entropy(pred, labels)

        ce = self.cross_entropy(pred, labels)

        pred_softmax = F.softmax(pred, dim=1)
        pred_softmax = torch.clamp(pred_softmax, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)

        rce = -torch.sum(pred_softmax * torch.log(label_one_hot + 1e-7), dim=1)

        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


def smooth_one_hots(true_labels: Tensor, classes: int, smoothing: float = 0.0) -> Tensor:
    assert 0 <= smoothing < 1, "smoothing must be in [0, 1)"
    confidence = 1.0 - smoothing
    device = true_labels.device
    true_labels = true_labels.long()

    true_dist = torch.full((true_labels.size(0), classes),
                          smoothing / (classes - 1), device=device, dtype=torch.float32)
    true_dist.scatter_(1, true_labels.unsqueeze(1), confidence)
    return true_dist


class KLLabelSmoothingLoss(nn.Module):

    def __init__(self, classes: int, smoothing: float = 0.0, reduction: str = 'mean'):
        super().__init__()
        self.classes = classes
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        if self.smoothing <= 0.0:
            return F.cross_entropy(outputs, targets, reduction=self.reduction)

        smoothed = smooth_one_hots(targets, classes=self.classes, smoothing=self.smoothing)
        log_probs = F.log_softmax(outputs, dim=1)
        loss = F.kl_div(log_probs, smoothed, reduction='batchmean' if self.reduction == 'mean' else 'sum')

        if self.reduction == 'sum':
            return loss
        elif self.reduction == 'mean':
            return loss
        else:
            batch_size = outputs.size(0)
            return loss * batch_size


def mixup_data(x: Tensor, y: Tensor, alpha: float = 0.2, device: Optional[str] = None, generator: Optional[torch.Generator] = None) -> Tuple[Tensor, Tensor, Tensor, float]:
    if alpha > 0:
        if generator is not None:
            rng_device = get_rng_device(device if device else x.device)
            u1 = torch.rand(1, device=rng_device, generator=generator).item()
            u2 = torch.rand(1, device=rng_device, generator=generator).item()
            old_state = np.random.get_state()
            np.random.seed(int(u1 * 1e9) % (2**31))
            lam = np.random.beta(alpha, alpha)
            np.random.set_state(old_state)
        else:
            lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    device = x.device if device is None else torch.device(device)
    
    if generator is not None:
        rng_device = get_rng_device(device)
        index = torch.randperm(batch_size, device=rng_device, generator=generator).to(device)
    else:
        index = torch.randperm(batch_size, device=device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x: Tensor, y: Tensor, alpha: float = 1.0, device: Optional[str] = None, generator: Optional[torch.Generator] = None) -> Tuple[Tensor, Tensor, Tensor, float]:
    if alpha > 0:
        if generator is not None:
            rng_device = get_rng_device(device if device else x.device)
            u1 = torch.rand(1, device=rng_device, generator=generator).item()
            u2 = torch.rand(1, device=rng_device, generator=generator).item()
            old_state = np.random.get_state()
            np.random.seed(int(u1 * 1e9) % (2**31))
            lam = np.random.beta(alpha, alpha)
            np.random.set_state(old_state)
        else:
            lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    device = x.device if device is None else torch.device(device)
    
    if generator is not None:
        rng_device = get_rng_device(device)
        index = torch.randperm(batch_size, device=rng_device, generator=generator).to(device)
    else:
        index = torch.randperm(batch_size, device=device)

    _, _, h, w = x.size()

    cut_ratio = np.sqrt(1. - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)

    if generator is not None:
        rng_device = get_rng_device(device)
        cx = torch.randint(0, w, (1,), device=rng_device, generator=generator).item()
        cy = torch.randint(0, h, (1,), device=rng_device, generator=generator).item()
    else:
        cx = np.random.randint(w)
        cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion_fn, pred, y_a, y_b, lam):
    return lam * criterion_fn(pred, y_a) + (1 - lam) * criterion_fn(pred, y_b)
