import os
import sys
import time
import logging
import argparse
from datetime import datetime
import json
import numpy as np
import matplotlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

import config
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

from config import get_defaults, print_config, load_config
from utils import set_seed, mixup_data, cutmix_data, KLLabelSmoothingLoss, FocalLoss, GeneralizedCrossEntropy, SymmetricCrossEntropy
from dataset import load_data

from model import (
    ExtraNet_CCSE, ExtraNet, ExtraNet_CCSE_Lite, ExtraNet_Scalable,
    ccse_resnet18, ccse_resnet34, ccse_resnet50, ccse_resnet101, ccse_resnet152,
    se_resnet18, se_resnet34, se_resnet50, se_resnet101, se_resnet152,
    resnet18, resnet34, resnet50, resnet101, resnet152
)


@dataclass
class TrainingMetrics:
    train_losses: List[float] = None
    train_accs: List[float] = None
    val_losses: List[float] = None
    val_accs: List[float] = None
    learning_rates: List[float] = None

    def __post_init__(self):
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.learning_rates = []


class Trainer:

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

        self.device = self._setup_device()
        self.result_dir = self._create_result_dir()
        self._setup_logging()

        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.metrics = TrainingMetrics()
        self.patience_counter = 0

        self.use_amp = config.get('use_amp', False) and self.device.type == 'cuda'
        self.scaler = GradScaler('cuda') if self.use_amp else None

        self.loss_type = config.get('loss_type', 'sce')
        self.base_label_smoothing = config.get('label_smoothing', 0.0)
        self.dynamic_label_smoothing = config.get('dynamic_label_smoothing', False)
        self.class_weights = None

        from utils import get_rng_device
        seed = config.get('seed', 42)
        rng_device = get_rng_device(self.device)
        self.rng = torch.Generator(device=rng_device)
        self.rng.manual_seed(seed)
        self.rng_device = rng_device

        print_config(config)

    def _setup_device(self) -> torch.device:
        device_arg = self.config.get('device', 'auto')

        if device_arg == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        else:
            device = device_arg

        torch_device = torch.device(device)
        print(f"Using device: {device}")

        if device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")

        return torch_device

    def _create_result_dir(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.config['model'] == 'extranet_scalable':
            model_name = f"{self.config['model']}_{self.config['scalable_model_scale']}"
        else:
            model_name = self.config['model']

        result_dir = Path('results') / f"{model_name}_{self.config['dataset']}_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=True)

        print(f"Results will be saved to: {result_dir}")
        return str(result_dir)

    def _setup_logging(self):
        log_file = os.path.join(self.result_dir, 'training.log')
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Training started: {datetime.now()}")
        self.logger.info(f"Result directory: {self.result_dir}")

    def _prepare_datasets(self) -> None:
        self.logger.info("="*60)
        self.logger.info("Preparing datasets...")

        self.train_loader = load_data(
            self.config['train_path'], self.config['batch_size'],
            self.config['num_workers'], "train", self.config
        )
        self.val_loader = load_data(
            self.config['val_path'], self.config['batch_size'],
            self.config['num_workers'], "val", self.config
        )
        self.test_loader = load_data(
            self.config['test_path'], self.config['batch_size'],
            self.config['num_workers'], "test", self.config
        )

        self.class_names = self.train_loader.dataset.classes
        self.num_classes = len(self.class_names)

        self.logger.info(f"Number of classes: {self.num_classes}")
        self.logger.info(f"Class names: {self.class_names}")
        self.logger.info(f"Dataset sizes - train: {len(self.train_loader.dataset)}, "
                        f"val: {len(self.val_loader.dataset)}, "
                        f"test: {len(self.test_loader.dataset)}")
        self.logger.info("="*60)

    def _build_model(self) -> None:
        self.logger.info("="*60)
        self.logger.info(f"Building model: {self.config['model']}")

        input_channels = self.config.get('grayscale_output_channels', 3)
        
        model_registry = {
            'extranet_ccse': lambda: ExtraNet_CCSE(num_classes=self.num_classes, use_simple_fusion=True, input_channels=input_channels),
            'extranet': lambda: ExtraNet(num_classes=self.num_classes, use_simple_fusion=True, input_channels=input_channels),
            'extranet_ccse_lite': lambda: ExtraNet_CCSE_Lite(num_classes=self.num_classes, use_simple_fusion=True, input_channels=input_channels),
            'extranet_scalable': lambda: ExtraNet_Scalable(
                scale=self.config.get('scalable_model_scale', 'tiny'), num_classes=self.num_classes, input_channels=input_channels
            ),
            'ccse_resnet18': lambda: ccse_resnet18(num_classes=self.num_classes, input_channels=input_channels),
            'ccse_resnet34': lambda: ccse_resnet34(num_classes=self.num_classes, input_channels=input_channels),
            'ccse_resnet50': lambda: ccse_resnet50(num_classes=self.num_classes, input_channels=input_channels),
            'ccse_resnet101': lambda: ccse_resnet101(num_classes=self.num_classes, input_channels=input_channels),
            'ccse_resnet152': lambda: ccse_resnet152(num_classes=self.num_classes, input_channels=input_channels),
            'se_resnet18': lambda: se_resnet18(num_classes=self.num_classes, input_channels=input_channels),
            'se_resnet34': lambda: se_resnet34(num_classes=self.num_classes, input_channels=input_channels),
            'se_resnet50': lambda: se_resnet50(num_classes=self.num_classes, input_channels=input_channels),
            'se_resnet101': lambda: se_resnet101(num_classes=self.num_classes, input_channels=input_channels),
            'se_resnet152': lambda: se_resnet152(num_classes=self.num_classes, input_channels=input_channels),
            'resnet18': lambda: resnet18(num_classes=self.num_classes, input_channels=input_channels),
            'resnet34': lambda: resnet34(num_classes=self.num_classes, input_channels=input_channels),
            'resnet50': lambda: resnet50(num_classes=self.num_classes, input_channels=input_channels),
            'resnet101': lambda: resnet101(num_classes=self.num_classes, input_channels=input_channels),
            'resnet152': lambda: resnet152(num_classes=self.num_classes, input_channels=input_channels),
        }

        model_name = self.config['model'].lower()
        if model_name not in model_registry:
            available_models = list(model_registry.keys())
            raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")

        self.model = model_registry[model_name]().to(self.device)

        if hasattr(torch, 'compile') and self.config.get('use_torch_compile', False):
            self.model = torch.compile(self.model)
            self.logger.info("PyTorch 2.0 compilation enabled")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.info(f"Total parameters: {total_params:,}, trainable parameters: {trainable_params:,}")
        if self.use_amp:
            self.logger.info("Mixed precision training (AMP) enabled")

        model_summary_path = Path(self.result_dir) / 'model_summary.txt'
        with open(model_summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Model: {self.config['model']}\n")
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,}\n")
            f.write(f"Mixed precision training (AMP): {self.use_amp}\n\n")
            f.write(f"Model structure:\n{self.model}")

        self.logger.info("="*60)

    def _setup_training(self) -> None:
        if self.config.get('use_class_weights', False) and self.loss_type in ['focal', 'ce']:
            self.class_weights = torch.tensor(self.config['class_weights'], device=self.device)
        else:
            self.logger.warning(f"Class weights are not used for {self.loss_type} loss types")
            self.class_weights = None
            self.logger.info("Class weights are only used for focal and ce loss types")

        self._build_loss_function(epoch=0)
        self.logger.info(f"Loss function type: {self.loss_type}")
        if self.dynamic_label_smoothing:
            self.logger.info(f"Dynamic label smoothing enabled (base value={self.base_label_smoothing})")

        self._setup_optimizer()
        self._setup_scheduler()

    def _setup_optimizer(self) -> None:
        optimizer_configs = {
            'adam': lambda: optim.Adam(
                self.model.parameters(),
                lr=self.config['lr'],
                weight_decay=self.config['weight_decay'],
                betas=tuple(self.config['betas'])
            ),
            'adamw': lambda: optim.AdamW(
                self.model.parameters(),
                lr=self.config['lr'],
                weight_decay=self.config['weight_decay'],
                betas=tuple(self.config['betas'])
            ),
            'sgd': lambda: optim.SGD(
                self.model.parameters(),
                lr=self.config['lr'],
                weight_decay=self.config['weight_decay'],
                momentum=self.config['momentum']
            )
        }

        optimizer_name = self.config['optimizer']
        if optimizer_name not in optimizer_configs:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        self.optimizer = optimizer_configs[optimizer_name]()
        self.logger.info(f"Optimizer: {optimizer_name}")

    def _setup_scheduler(self) -> None:
        scheduler_configs = {
            'cosine': lambda: optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['cosine_t_max'],
                eta_min=self.config['cosine_eta_min']
            ),
            'cosine_restarts': lambda: optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config['cosine_t_max'],
                T_mult=self.config.get('cosine_T_mult', 1),
                eta_min=self.config['cosine_eta_min']
            ),
            'step': lambda: optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['step_size'],
                gamma=self.config['gamma']
            ),
            'plateau': lambda: optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=self.config['plateau_patience'],
                factor=self.config['plateau_factor']
            )
        }

        scheduler_name = self.config['scheduler']
        if scheduler_name not in scheduler_configs:
            raise ValueError(f"Unsupported learning rate scheduler: {scheduler_name}")

        self.scheduler = scheduler_configs[scheduler_name]()
        self.logger.info(f"Learning rate scheduler: {scheduler_name}")

        self.warmup_epochs = self.config.get('warmup_epochs', 0) if self.config.get('use_warmup', False) else 0
        if self.warmup_epochs > 0:
            self.logger.info(f"Learning rate warmup: {self.warmup_epochs} epochs")

    def _build_loss_function(self, epoch: int = 0) -> None:
        smoothing = (self.base_label_smoothing * (1 - epoch / max(1, self.config['epochs']))) \
                   if self.dynamic_label_smoothing and self.loss_type == 'kl' \
                   else (self.base_label_smoothing if self.loss_type == 'kl' else 0.0)

        loss_configs = {
            'focal': lambda: FocalLoss(
                gamma=self.config['focal_gamma'],
                alpha=self.class_weights
            ).to(self.device),
            'gce': lambda: GeneralizedCrossEntropy(
                num_classes=self.num_classes,
                q=self.config['gce_q']
            ).to(self.device),
            'sce': lambda: SymmetricCrossEntropy(
                num_classes=self.num_classes,
                class_weights=self.class_weights,
                alpha=self.config.get('sce_alpha', 0.1),
                beta=self.config.get('sce_beta', 1.0),
                ce_only_epochs=self.config.get('sce_ce_only_epochs', 8),
                label_smoothing=self.base_label_smoothing
            ).to(self.device),
            'kl': lambda: KLLabelSmoothingLoss(
                classes=self.num_classes,
                smoothing=smoothing
            ).to(self.device),
            'ce': lambda: nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=self.base_label_smoothing
            ).to(self.device)
        }

        assert self.loss_type in loss_configs, \
            f"Unsupported loss type: {self.loss_type} " \
            f"Available types: {list(loss_configs.keys())}"
        self.criterion = loss_configs[self.loss_type]()

        self._loss_fn = lambda pred, target, epoch=None: self.criterion(pred, target, epoch) \
            if self.loss_type == 'sce' else self.criterion(pred, target)

        self.logger.info(f"Loss function: {self.loss_type}")

        if self.loss_type == 'kl' and self.dynamic_label_smoothing:
            self.logger.info(f"Dynamic label smoothing (KL) updated to: {smoothing:.5f} (epoch {epoch+1})")

    def _warmup_lr(self, epoch: int) -> None:
        if epoch < self.warmup_epochs:
            warmup_lr = self.config['lr'] * float(epoch + 1) / float(self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()

        if self.dynamic_label_smoothing and self.loss_type == 'kl':
            self._build_loss_function(epoch)

        mixup_enabled = self.config.get('mixup_alpha', 0) > 0
        cutmix_enabled = self.config.get('cutmix_alpha', 0) > 0
        mixup_prob = self.config.get('mixup_prob', 0.5) if mixup_enabled else 0.0
        cutmix_prob = self.config.get('cutmix_prob', 0.5) if cutmix_enabled else 0.0

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Train]", leave=False)

        for batch_idx, (images, labels) in enumerate(pbar):
            labels = labels.long().to(self.device, non_blocking=True)
            images = images.to(self.device, non_blocking=True)

            if labels.max() >= self.num_classes or labels.min() < 0:
                raise ValueError(f"Invalid label range: min={labels.min().item()}, max={labels.max().item()}, "
                               f"expected range [0, {self.num_classes-1}]")

            mixed, labels_a, labels_b, lam = self._apply_data_augmentation(
                images, labels, mixup_enabled, cutmix_enabled, mixup_prob, cutmix_prob
            )

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self._compute_loss(outputs, labels, labels_a, labels_b, lam, mixed, epoch)

            self._backward_pass(loss)

            running_loss += loss.item() * images.size(0)
            total += images.size(0)
            correct += self._compute_accuracy(outputs, labels, labels_a, labels_b, lam, mixed)

            if batch_idx % self.config.get('log_freq', 5) == 0:
                current_loss = running_loss / total
                current_acc = 100 * correct / total
                pbar.set_postfix({'loss': f"{current_loss:.4f}", 'acc': f"{current_acc:.2f}%"})

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def _apply_data_augmentation(self, images: torch.Tensor, labels: torch.Tensor,
                                mixup_enabled: bool, cutmix_enabled: bool,
                                mixup_prob: float, cutmix_prob: float) -> Tuple[bool, Optional[torch.Tensor], Optional[torch.Tensor], float]:
        mixed, labels_a, labels_b, lam = False, None, None, 1.0

        rand_vals = torch.rand(2, device=self.rng_device, generator=self.rng)
        rand_val1, rand_val2 = rand_vals[0].item(), rand_vals[1].item()
        
        aug_methods = []
        if mixup_enabled and rand_val1 < mixup_prob:
            aug_methods.append(('mixup', self.config['mixup_alpha']))
        if cutmix_enabled and rand_val2 < cutmix_prob:
            aug_methods.append(('cutmix', self.config['cutmix_alpha']))

        if aug_methods:
            method, alpha = aug_methods[torch.randint(len(aug_methods), (1,), device=self.rng_device, generator=self.rng).item()]
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=alpha, device=self.device, generator=self.rng) \
                if method == 'mixup' \
                    else cutmix_data(images, labels, alpha=alpha, device=self.device, generator=self.rng)
            mixed = True

        return mixed, labels_a, labels_b, lam

    def _compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor,
                     labels_a: Optional[torch.Tensor], labels_b: Optional[torch.Tensor],
                     lam: float, mixed: bool, epoch: int) -> torch.Tensor:
        if mixed:
            loss = lam * self._loss_fn(outputs, labels_a, epoch) + \
                   (1 - lam) * self._loss_fn(outputs, labels_b, epoch)
        else:
            loss = self._loss_fn(outputs, labels, epoch)
        return loss

    def _backward_pass(self, loss: torch.Tensor) -> None:
        if self.use_amp:
            self.scaler.scale(loss).backward()
            if self.config.get('gradient_clip', 0) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
            self.optimizer.step()

    def _compute_accuracy(self, outputs: torch.Tensor, labels: torch.Tensor,
                         labels_a: Optional[torch.Tensor], labels_b: Optional[torch.Tensor],
                         lam: float, mixed: bool) -> float:
        _, predicted = torch.max(outputs, 1)

        if mixed:
            correct = (lam * predicted.eq(labels_a).sum().float() +
                      (1 - lam) * predicted.eq(labels_b).sum().float()).item()
        else:
            correct = (predicted == labels).sum().item()

        return correct

    def validate_epoch(self, epoch: int) -> Tuple[float, float, list, list]:
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Val]", leave=False)

        with torch.no_grad():
            for images, labels in pbar:
                labels = labels.to(self.device, non_blocking=True)
                images = images.to(self.device, non_blocking=True)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self._loss_fn(outputs, labels, epoch)

                running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

                current_loss = running_loss / total
                current_acc = 100 * correct / total
                pbar.set_postfix({'loss': f"{current_loss:.4f}", 'acc': f"{current_acc:.2f}%"})

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc, all_labels, all_preds

    def train(self) -> None:
        self._prepare_datasets()
        self._build_model()
        self._setup_training()

        config_json_path = Path(self.result_dir) / 'config.json'
        with open(config_json_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)


        self.logger.info("Starting training...")
        self.logger.info("="*60)

        start_time = time.time()
        switch_epoch = self.config.get('switch_to_plateau_epoch', 999)
        patience = self.config.get('patience', 50)

        for epoch in range(self.config['epochs']):
            if self.config.get('use_warmup', False):
                self._warmup_lr(epoch)

            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc, val_labels, val_preds = self.validate_epoch(epoch)

            self.metrics.train_losses.append(train_loss)
            self.metrics.train_accs.append(train_acc)
            self.metrics.val_losses.append(val_loss)
            self.metrics.val_accs.append(val_acc)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.metrics.learning_rates.append(current_lr)

            self.logger.info(
                f"Epoch {epoch+1:03d}/{self.config['epochs']} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            self._update_scheduler(epoch, val_acc, switch_epoch)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0

                model_path = Path(self.result_dir) / 'best_model.pth'
                torch.save(self.model.state_dict(), model_path)
                self.logger.info(f">>> Saving best model (validation accuracy: {val_acc:.4f})")
            else:
                self.patience_counter += 1

            if self.patience_counter >= patience:
                self.logger.info(f"Early stopping triggered, no improvement after {self.patience_counter} epochs")
                break

        total_time = time.time() - start_time
        self.logger.info("="*60)
        self.logger.info(f"Training completed! Total time: {total_time/3600:.2f} hours")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.4f} (epoch {self.best_epoch+1})")
        self.logger.info("="*60)

        self._save_visualizations()
        self._save_confusion_matrix()

        self.logger.info(f"All results saved to: {self.result_dir}")
        self.logger.info(f"Best model: {Path(self.result_dir) / 'best_model.pth'}")

    def _update_scheduler(self, epoch: int, val_acc: float, switch_epoch: int) -> None:
        if epoch == switch_epoch:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max',
                patience=self.config['plateau_patience'],
                factor=self.config['plateau_factor']
            )
            self.logger.info(
                f"Switching learning rate scheduler to ReduceLROnPlateau (epoch {epoch+1}, "
                f"patience={self.config['plateau_patience']}, factor={self.config['plateau_factor']})"
            )

        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_acc)
        elif epoch >= self.warmup_epochs:
            try:
                self.scheduler.step()
            except TypeError:
                self.scheduler.step(epoch)

    def _save_visualizations(self) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        epochs = range(1, len(self.metrics.train_losses) + 1)

        axes[0, 0].plot(epochs, self.metrics.train_losses, label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.metrics.val_losses, label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        train_acc_pct = [acc * 100 for acc in self.metrics.train_accs]
        val_acc_pct = [acc * 100 for acc in self.metrics.val_accs]
        axes[0, 1].plot(epochs, train_acc_pct, label='Train Acc', linewidth=2)
        axes[0, 1].plot(epochs, val_acc_pct, label='Val Acc', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=self.best_val_acc * 100, color='r', linestyle='--', alpha=0.7,
                           label='.1f')
        axes[0, 1].legend()

        axes[1, 0].plot(epochs, self.metrics.learning_rates, linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(epochs, train_acc_pct, 'o-', label='Train Acc', linewidth=2, markersize=3)
        axes[1, 1].plot(epochs, val_acc_pct, 's-', label='Val Acc', linewidth=2, markersize=3)
        axes[1, 1].fill_between(epochs, train_acc_pct, val_acc_pct, alpha=0.2, label='Gap')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('Overfitting Analysis')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(Path(self.result_dir) / 'training_history.png', dpi=200, bbox_inches='tight')
        plt.close()
        self.logger.info("Training visualization results saved")

    def _save_confusion_matrix(self):
        self.logger.info("Generating confusion matrix...")
        self.model.load_state_dict(torch.load(os.path.join(self.result_dir, 'best_model.pth')))
        self.model.eval()
        all_labels, all_preds = [], []
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Evaluating", leave=False):
                images = images.to(self.device, non_blocking=True)
                if self.use_amp:
                    with autocast('cuda'):
                        outputs = self.model(images)
                else:
                    outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.numpy())
                all_preds.extend(predicted.cpu().numpy())
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names, cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {self.config["model"]}\nValidation Accuracy: {self.best_val_acc:.4f}', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'confusion_matrix.png'), dpi=200, bbox_inches='tight')
        plt.close()
        report = classification_report(all_labels, all_preds, target_names=self.class_names, digits=4, zero_division=1)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)
        with open(os.path.join(self.result_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
            f.write("="*60 + "\nValidation Set (PublicTest) Classification Report\n" + "="*60 + "\n")
            f.write(f"Model: {self.config['model']}\nDataset: {self.config['dataset']}\nBest Epoch: {self.best_epoch + 1}\nValidation Size: {len(all_labels)} (PublicTest)\n")
            f.write(f"Total Parameters: {sum(p.numel() for p in self.model.parameters()):,}\nMixed Precision (AMP): {self.use_amp}\n\n")
            f.write(f"PublicTest Accuracy: {accuracy:.6f}\nWeighted F1 Score: {f1:.6f}\n\nDetailed Classification Report:\n{report}\n\nConfusion Matrix:\n{cm}")
        self.logger.info(f"Confusion matrix and classification report saved\nPublicTest Accuracy: {accuracy:.6f}\nWeighted F1 Score: {f1:.6f}")

def parse_args():
    parser = argparse.ArgumentParser(description='FER2013 Emotion Recognition Training Script')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file (default: config.yaml)')
    parser.add_argument('--model', type=str, default=None, help='Model name (overrides config)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (overrides config)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Warning: Config file {args.config} not found, using default configuration")
        config = get_defaults()
    
    if args.model: config['model'] = args.model
    if args.epochs: config['epochs'] = args.epochs
    if args.batch_size: config['batch_size'] = args.batch_size
    if args.lr: config['lr'] = args.lr
    
    set_seed(config['seed'])
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
