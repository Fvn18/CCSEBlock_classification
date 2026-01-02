import os
import sys
import time
import random
import logging
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

import config
from config import get_defaults, print_config, load_config
from utils import (
    set_seed, mixup_data, cutmix_data, KLLabelSmoothingLoss, 
    FocalLoss, GeneralizedCrossEntropy, SymmetricCrossEntropy, get_rng_device
)
from dataset import load_data
import model as models_module


@dataclass
class TrainingMetrics:
    train_losses: List[float] = None
    train_accs: List[float] = None
    val_losses: List[float] = None
    val_accs: List[float] = None
    learning_rates: List[float] = None

    def __post_init__(self):
        if self.train_losses is None:
            self.train_losses = []
        if self.train_accs is None:
            self.train_accs = []
        if self.val_losses is None:
            self.val_losses = []
        if self.val_accs is None:
            self.val_accs = []
        if self.learning_rates is None:
            self.learning_rates = []


class Trainer:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.config = cfg

        self.device = self._setup_device()
        self.resume_dir, self.result_dir = self._create_result_dir()
        self._setup_logging()

        self.best_val_acc: float = 0.0
        self.best_epoch: int = 0    
        self.metrics = TrainingMetrics()
        self.patience_counter: int = 0

        self.use_amp = cfg.get("use_amp", False) and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        self.loss_type = cfg.get("loss_type", "sce")
        self.base_label_smoothing = cfg.get("label_smoothing", 0.0)
        self.dynamic_label_smoothing = cfg.get("dynamic_label_smoothing", False)
        self.class_weights = None

        seed = cfg.get("seed", 42)
        rng_device = get_rng_device(self.device)
        self.rng = torch.Generator(device=rng_device)
        self.rng.manual_seed(seed)
        self.rng_device = rng_device

        if hasattr(config, 'print_config'):
            print_config(cfg)

    def _setup_device(self) -> torch.device:
        device_arg = self.config.get("device", "auto")
        if device_arg == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = device_arg

        torch_device = torch.device(device)
        print(f"Using device: {device}")

        if device == "cuda":
            try:
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(f"CUDA version: {torch.version.cuda}")
                print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
            except Exception:
                pass

        return torch_device

    def _create_result_dir(self) -> str:
        resume_path = self.config.get("resume", None)
        if resume_path:
            if os.path.isfile(resume_path):
                resume_dir = Path(resume_path).resolve().parent
            elif os.path.isdir(resume_path):
                resume_dir = Path(resume_path).resolve()
            else:
                raise ValueError(f"Resume path does not exist: {resume_path}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.config["model"]
        if model_name == "extranet_scalable":
            model_name = f"{model_name}_{self.config.get('scalable_model_scale','tiny')}"
        result_dir = Path("results") / f"{model_name}_{self.config['dataset']}_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {result_dir}")
        print(f"Resume directory: {resume_dir}")
        return resume_dir, str(result_dir)

    def _setup_logging(self) -> None:
        log_file = os.path.join(self.result_dir, "training.log")
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Training started: {datetime.now()}")
        self.logger.info(f"Result directory: {self.result_dir}")

    def _prepare_datasets(self) -> None:
        self.logger.info("=" * 60)
        self.logger.info("Preparing datasets...")

        self.train_loader = load_data(
            self.config["train_path"], self.config["batch_size"],
            self.config["num_workers"], "train", self.config
        )
        self.val_loader = load_data(
            self.config["val_path"], self.config["batch_size"],
            self.config["num_workers"], "val", self.config
        )

        if hasattr(self.config, "test_path") and self.config.get("test_path"):
            self.test_loader = load_data(
                self.config["test_path"], self.config["batch_size"],
                self.config["num_workers"], "test", self.config
            )
        else:
            self.test_loader = None

        self.class_names = getattr(self.train_loader.dataset, 'classes', None)
        if self.class_names is None and hasattr(self.train_loader.dataset, 'targets'):
            unique = sorted(set(self.train_loader.dataset.targets))
            self.class_names = [str(x) for x in unique]

        self.num_classes = len(self.class_names) if self.class_names is not None else 0

        self.logger.info(f"Number of classes: {self.num_classes}")
        self.logger.info(f"Class names: {self.class_names}")
        self.logger.info(
            f"Dataset sizes - train: {len(self.train_loader.dataset)}, "
            f"val: {len(self.val_loader.dataset)}"
        )

        try:
            if hasattr(self.train_loader.dataset, 'targets'):
                counts = np.bincount(np.array(self.train_loader.dataset.targets))
                self.logger.info("Per-class sample counts (train):")
                for idx, cnt in enumerate(counts):
                    name = self.class_names[idx] if idx < len(self.class_names) else str(idx)
                    self.logger.info(f"  {idx}: {name} -> {cnt}")
        except Exception:
            pass

        self.logger.info("=" * 60)

    def _build_model(self) -> None:
        self.logger.info("=" * 60)
        self.logger.info(f"Building model: {self.config['model']}")

        input_channels = self.config.get("grayscale_output_channels", 3)

        model_registry = {
            "extranet_ccse": lambda: models_module.ExtraNet_CCSE(num_classes=self.num_classes, use_simple_fusion=True, input_channels=input_channels),
            "extranet": lambda: models_module.ExtraNet(num_classes=self.num_classes, use_simple_fusion=True, input_channels=input_channels),
            "extranet_ccse_lite": lambda: models_module.ExtraNet_CCSE_Lite(num_classes=self.num_classes, use_simple_fusion=True, input_channels=input_channels),
            "extranet_scalable": lambda: models_module.ExtraNet_Scalable(scale=self.config.get("scalable_model_scale", "tiny"), num_classes=self.num_classes, input_channels=input_channels),
        }

        resnet_names = [
            "ccse_resnet18", "ccse_resnet34", "ccse_resnet50", "ccse_resnet101", "ccse_resnet152",
            "se_resnet18", "se_resnet34", "se_resnet50", "se_resnet101", "se_resnet152",
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
        ]

        for name in resnet_names:
            if hasattr(models_module, name):
                model_registry[name] = lambda n=name: getattr(models_module, n)(num_classes=self.num_classes, input_channels=input_channels)

        model_name = self.config["model"].lower()
        if model_name not in model_registry:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_registry.keys())}")

        self.model = model_registry[model_name]().to(self.device)

        if hasattr(torch, "compile") and self.config.get("use_torch_compile", False):
            self.model = torch.compile(self.model)
            self.logger.info("PyTorch 2.0 compilation enabled")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.logger.info(f"Total parameters: {total_params:,}, trainable parameters: {trainable_params:,}")
        if self.use_amp:
            self.logger.info("Mixed precision training (AMP) enabled")

        model_summary_path = Path(self.result_dir) / "model_summary.txt"
        with open(model_summary_path, "w", encoding="utf-8") as f:
            f.write(f"Model: {self.config['model']}\n")
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,}\n")
            f.write(f"Mixed precision training (AMP): {self.use_amp}\n\n")
            f.write(f"Model structure:\n{self.model}")

        self.logger.info("=" * 60)

    def _setup_training(self) -> None:
        if self.config.get("use_class_weights", False) and self.loss_type in ["focal", "ce", "sce"]:
            self.class_weights = torch.tensor(self.config["class_weights"], device=self.device)
        else:
            if self.config.get("use_class_weights", False):
                self.logger.warning(f"Class weights are not used for {self.loss_type} loss types")
            self.class_weights = None

        self._build_loss_function(epoch=0)
        self.logger.info(f"Loss function type: {self.loss_type}")
        if self.dynamic_label_smoothing:
            self.logger.info(f"Dynamic label smoothing enabled (base value={self.base_label_smoothing})")

        self._setup_optimizer()
        self._setup_scheduler()

    def _setup_optimizer(self) -> None:
        optimizer_configs = {
            "adam": lambda: optim.Adam(
                self.model.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
                betas=tuple(self.config["betas"])
            ),
            "adamw": lambda: optim.AdamW(
                self.model.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
                betas=tuple(self.config["betas"])
            ),
            "sgd": lambda: optim.SGD(
                self.model.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config["weight_decay"],
                momentum=self.config["momentum"]
            )
        }

        optimizer_name = self.config["optimizer"]
        if optimizer_name not in optimizer_configs:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        self.optimizer = optimizer_configs[optimizer_name]()
        self.logger.info(f"Optimizer: {optimizer_name}")

    def _setup_scheduler(self) -> None:
        scheduler_configs = {
            "cosine": lambda: optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["cosine_t_max"],
                eta_min=self.config["cosine_eta_min"]
            ),
            "cosine_restarts": lambda: optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config["cosine_t_max"],
                T_mult=self.config.get("cosine_T_mult", 1),
                eta_min=self.config["cosine_eta_min"]
            ),
            "step": lambda: optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config["step_size"],
                gamma=self.config["gamma"]
            ),
            "plateau": lambda: optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                patience=self.config["plateau_patience"],
                factor=self.config["plateau_factor"]
            )
        }

        scheduler_name = self.config["scheduler"]
        if scheduler_name not in scheduler_configs:
            raise ValueError(f"Unsupported learning rate scheduler: {scheduler_name}")

        self.scheduler = scheduler_configs[scheduler_name]()
        self.logger.info(f"Learning rate scheduler: {scheduler_name}")

        self.warmup_epochs = self.config.get("warmup_epochs", 0) if self.config.get("use_warmup", False) else 0
        if self.warmup_epochs > 0:
            self.logger.info(f"Learning rate warmup: {self.warmup_epochs} epochs")

    def _build_loss_function(self, epoch: int = 0) -> None:
        smoothing = (
            self.base_label_smoothing * (1 - epoch / max(1, self.config["epochs"]))
            if self.dynamic_label_smoothing and self.loss_type == "kl"
            else (self.base_label_smoothing if self.loss_type == "kl" else 0.0)
        )

        loss_configs = {
            "focal": lambda: FocalLoss(gamma=self.config["focal_gamma"], alpha=self.class_weights).to(self.device),
            "gce": lambda: GeneralizedCrossEntropy(num_classes=self.num_classes, q=self.config["gce_q"]).to(self.device),
            "sce": lambda: SymmetricCrossEntropy(
                num_classes=self.num_classes,
                class_weights=self.class_weights,
                alpha=self.config.get("sce_alpha", 0.1),
                beta=self.config.get("sce_beta", 1.0),
                ce_only_epochs=self.config.get("sce_ce_only_epochs", 8),
                label_smoothing=self.base_label_smoothing
            ).to(self.device),
            "kl": lambda: KLLabelSmoothingLoss(classes=self.num_classes, smoothing=smoothing).to(self.device),
            "ce": lambda: nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=self.base_label_smoothing).to(self.device)
        }

        assert self.loss_type in loss_configs, f"Unsupported loss type: {self.loss_type}"
        self.criterion = loss_configs[self.loss_type]()
        self._loss_fn = (lambda pred, target, epoch=None: self.criterion(pred, target, epoch)) if self.loss_type == "sce" else self.criterion

        if self.loss_type == "kl" and self.dynamic_label_smoothing:
            self.logger.info(f"Dynamic label smoothing (KL) updated to: {smoothing:.5f} (epoch {epoch+1})")

    def _warmup_lr(self, epoch: int) -> None:
        if epoch < self.warmup_epochs:
            warmup_lr = self.config["lr"] * float(epoch + 1) / float(self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = warmup_lr

    def save_checkpoint(self, epoch: int, is_best: bool = False, note: Optional[str] = None) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": getattr(self, "optimizer", None).state_dict() if hasattr(self, "optimizer") else None,
            "scheduler_state_dict": getattr(self, "scheduler", None).state_dict() if hasattr(self, "scheduler") else None,
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "patience_counter": self.patience_counter,
            "metrics": {
                "train_losses": self.metrics.train_losses,
                "train_accs": self.metrics.train_accs,
                "val_losses": self.metrics.val_losses,
                "val_accs": self.metrics.val_accs,
                "learning_rates": self.metrics.learning_rates,
            },
        }

        if self.use_amp and getattr(self, "scaler", None) is not None:
            try:
                ckpt["scaler_state_dict"] = self.scaler.state_dict()
            except Exception:
                ckpt["scaler_state_dict"] = None

        if note:
            ckpt["note"] = note

        checkpoint_dir = Path(self.config.get("resume", "./checkpoint"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        latest_path = checkpoint_dir / f"{self.config['model']}_{self.config['dataset']}_latest_checkpoint.pth"
        torch.save(ckpt, latest_path)

        if is_best:
            best_model_path = Path(self.result_dir) / "best_model.pth"
            torch.save(self.model.state_dict(), best_model_path)
            self.logger.info(f">>> Saved best model accuracy: {self.best_val_acc:.8f} at epoch {epoch+1}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        path = Path(checkpoint_path)
        if path.is_dir():
            path = path / f"{self.config['model']}_{self.config['dataset']}_latest_checkpoint.pth"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        if "model_state_dict" in ckpt and ckpt["model_state_dict"] is not None:
            try:
                self.model.load_state_dict(ckpt["model_state_dict"])
            except Exception as e:
                self.logger.warning(f"Model state load warning: {e}")
                self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif "model" in ckpt and ckpt["model"] is not None:
            try:
                self.model.load_state_dict(ckpt["model"])
            except Exception as e:
                self.logger.warning(f"Model state load warning: {e}")
                self.model.load_state_dict(ckpt["model"], strict=False)

        if "optimizer_state_dict" in ckpt and hasattr(self, "optimizer") and ckpt["optimizer_state_dict"] is not None:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                self.logger.warning(f"Failed to restore optimizer state: {e}")
        elif "optimizer" in ckpt and hasattr(self, "optimizer") and ckpt["optimizer"] is not None:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                self.logger.warning(f"Failed to restore optimizer state: {e}")

        if "scheduler_state_dict" in ckpt and hasattr(self, "scheduler") and ckpt["scheduler_state_dict"] is not None:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                self.logger.warning(f"Failed to restore scheduler state: {e}")
        elif "scheduler" in ckpt and hasattr(self, "scheduler") and ckpt["scheduler"] is not None:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                self.logger.warning(f"Failed to restore scheduler state: {e}")

        if self.use_amp and getattr(self, "scaler", None) is not None and "scaler_state_dict" in ckpt:
            try:
                self.scaler.load_state_dict(ckpt.get("scaler_state_dict"))
            except Exception as e:
                self.logger.warning(f"Failed to restore AMP scaler: {e}")

        self.best_val_acc = ckpt.get("best_val_acc", ckpt.get("best_acc", self.best_val_acc))
        self.best_epoch = ckpt.get("best_epoch", ckpt.get("best_epoch_index", self.best_epoch))
        self.patience_counter = ckpt.get("patience_counter", self.patience_counter)
        metrics = ckpt.get("metrics", None)
        if metrics:
            if isinstance(metrics, dict):
                self.metrics.train_losses = metrics.get("train_losses", metrics.get("train_loss", []))
                self.metrics.train_accs = metrics.get("train_accs", metrics.get("train_accuracy", []))
                self.metrics.val_losses = metrics.get("val_losses", metrics.get("val_loss", []))
                self.metrics.val_accs = metrics.get("val_accs", metrics.get("val_accuracy", []))
                self.metrics.learning_rates = metrics.get("learning_rates", metrics.get("learning_rate", []))

        next_epoch = ckpt.get("epoch", 0) + 1
        self.logger.info(f"Loaded checkpoint '{path}' -> resuming from epoch {next_epoch}")
        return next_epoch

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.train()
        
        if self.dynamic_label_smoothing and self.loss_type == 'kl':
            self._build_loss_function(epoch)

        mixup_enabled = self.config.get("mixup_alpha", 0) > 0
        cutmix_enabled = self.config.get("cutmix_alpha", 0) > 0
        mixup_prob = self.config.get("mixup_prob", 0.5) if mixup_enabled else 0.0
        cutmix_prob = self.config.get("cutmix_prob", 0.5) if cutmix_enabled else 0.0

        running_loss = 0.0
        correct = 0.0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Train]", leave=False)

        checkpoint_freq = self.config.get("checkpoint_freq", 0) if self.config.get("save_checkpoints", False) else 0

        for batch_idx, (images, labels) in enumerate(pbar):
            labels = labels.long().to(self.device, non_blocking=True)
            images = images.to(self.device, non_blocking=True)

            if labels.max() >= self.num_classes or labels.min() < 0:
                raise ValueError(
                    f"Invalid label range: min={labels.min().item()}, max={labels.max().item()}, expected range [0, {self.num_classes-1}]"
                )

            images, mixed, labels_a, labels_b, lam = self._apply_data_augmentation(
                images, labels, mixup_enabled, cutmix_enabled, mixup_prob, cutmix_prob
            )

            if mixed:
                if labels_a is not None:
                    labels_a = labels_a.long().to(self.device, non_blocking=True)
                if labels_b is not None:
                    labels_b = labels_b.long().to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp, device_type=self.device.type):
                outputs = self.model(images)
                loss = self._compute_loss(outputs, labels, labels_a, labels_b, lam, mixed, epoch)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.config.get("gradient_clip", 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["gradient_clip"])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.get("gradient_clip", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["gradient_clip"])
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            total += images.size(0)
            correct += self._compute_accuracy(outputs, labels, labels_a, labels_b, lam, mixed)

            if batch_idx % self.config.get("log_freq", 5) == 0:
                current_loss = running_loss / total
                current_acc = 100 * correct / total
                pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.2f}%"})

            if checkpoint_freq > 0 and (batch_idx + 1) % checkpoint_freq == 0:
                try:
                    self.save_checkpoint(epoch, is_best=False, note=f"batch_{batch_idx+1}")
                except Exception as e:
                    self.logger.error(f"Failed to save batch checkpoint: {e}")

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def _apply_data_augmentation(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        mixup_enabled: bool,
        cutmix_enabled: bool,
        mixup_prob: float,
        cutmix_prob: float,
    ) -> Tuple[torch.Tensor, bool, Optional[torch.Tensor], Optional[torch.Tensor], float]:
        mixed, labels_a, labels_b, lam = False, None, None, 1.0

        rand_vals = torch.rand(2, device=self.rng_device, generator=self.rng)
        rand_val1, rand_val2 = rand_vals[0].item(), rand_vals[1].item()

        aug_methods = []
        if mixup_enabled and rand_val1 < mixup_prob:
            aug_methods.append(("mixup", self.config["mixup_alpha"]))
        if cutmix_enabled and rand_val2 < cutmix_prob:
            aug_methods.append(("cutmix", self.config["cutmix_alpha"]))

        if aug_methods:
            idx = torch.randint(len(aug_methods), (1,), device=self.rng_device, generator=self.rng).item()
            method, alpha = aug_methods[idx]
            if method == "mixup":
                images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=alpha, device=self.device, generator=self.rng)
            else:
                images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=alpha, device=self.device, generator=self.rng)
            mixed = True

        return images, mixed, labels_a, labels_b, lam

    def _compute_loss(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        labels_a: Optional[torch.Tensor],
        labels_b: Optional[torch.Tensor],
        lam: float,
        mixed: bool,
        epoch: int,
    ) -> torch.Tensor:
        if mixed:
            if self.loss_type == "sce":
                loss_a = self._loss_fn(outputs, labels_a, epoch)
                loss_b = self._loss_fn(outputs, labels_b, epoch)
            else:
                loss_a = self._loss_fn(outputs, labels_a)
                loss_b = self._loss_fn(outputs, labels_b)
            return lam * loss_a + (1 - lam) * loss_b
        if self.loss_type == "sce":
            return self._loss_fn(outputs, labels, epoch)
        else:
            return self._loss_fn(outputs, labels)

    def _compute_accuracy(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        labels_a: Optional[torch.Tensor],
        labels_b: Optional[torch.Tensor],
        lam: float,
        mixed: bool,
    ) -> float:
        _, predicted = torch.max(outputs, 1)
        if mixed:
            if labels_a is not None:
                labels_a = labels_a.to(predicted.device)
            if labels_b is not None:
                labels_b = labels_b.to(predicted.device)
            correct = lam * predicted.eq(labels_a).sum().float() + (1 - lam) * predicted.eq(labels_b).sum().float()
        else:
            correct = predicted.eq(labels).sum().float()
        return float(correct)

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

                with autocast(enabled=self.use_amp, device_type=self.device.type):
                    outputs = self.model(images)
                    loss = self._loss_fn(outputs, labels, epoch) if self.loss_type == "sce" else self._loss_fn(outputs, labels)

                running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

                current_loss = running_loss / total
                current_acc = 100 * correct / total
                pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.2f}%"})

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc, all_labels, all_preds

    def train(self) -> None:
        self._prepare_datasets()
        self._build_model()
        self._setup_training()

        config_json_path = Path(self.result_dir) / "config.json"
        with open(config_json_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)

        start_epoch = 0
        resume_path = self.config.get("resume", None)
        if_first_train = self.config.get("if_first_train", True)

        if resume_path and not if_first_train:
            try:
                start_epoch = self.load_checkpoint(resume_path)
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint for resume: {e}")
                raise

        total_epochs = int(self.config["epochs"])
        if start_epoch >= total_epochs:
            self.logger.info(f"Start epoch ({start_epoch}) >= total epochs ({total_epochs}), nothing to do.")
            return

        self.logger.info("Starting training...")
        self.logger.info("=" * 60)

        start_time = time.time()
        switch_epoch = self.config.get("switch_to_plateau_epoch", 999)
        patience = self.config.get("patience", 50)

        training_interrupted = False

        try:
            for epoch in range(start_epoch, total_epochs):
                if self.config.get("use_warmup", False):
                    self._warmup_lr(epoch)

                train_loss, train_acc = self.train_epoch(epoch)
                val_loss, val_acc, val_labels, val_preds = self.validate_epoch(epoch)

                self.metrics.train_losses.append(train_loss)
                self.metrics.train_accs.append(train_acc)
                self.metrics.val_losses.append(val_loss)
                self.metrics.val_accs.append(val_acc)
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.metrics.learning_rates.append(current_lr)

                self.logger.info(
                    f"Epoch {epoch+1:03d}/{total_epochs} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                    f"LR: {current_lr:.6f}"
                )

                self._update_scheduler(epoch, val_acc, switch_epoch)

                improved = val_acc > self.best_val_acc
                if improved:
                    self.best_val_acc = val_acc
                    self.best_epoch = epoch
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                try:
                    self.save_checkpoint(epoch, is_best=improved, note="best_checkpoint")
                except Exception as e:
                    self.logger.error(f"Failed to save checkpoint at epoch {epoch+1}: {e}")

                if self.patience_counter >= patience:
                    self.logger.info(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                    break

        except KeyboardInterrupt:
            training_interrupted = True
            try:
                self.save_checkpoint(epoch, is_best=False, note="interrupted_by_user")
                self.logger.info(f"Training interrupted by user, checkpoint saved at epoch {epoch+1}")
            except Exception as e:
                self.logger.error(f"Failed to save checkpoint after interrupt: {e}")
            raise

        except Exception as e:
            self.logger.error(f"Unhandled exception during training: {e}", exc_info=True)
            try:
                self.save_checkpoint(epoch, is_best=False, note="crash_checkpoint")
                self.logger.info(f"Saved crash checkpoint at epoch {epoch+1}")
            except Exception as se:
                self.logger.error(f"Failed to save crash checkpoint: {se}")
            raise

        finally:
            total_time = time.time() - start_time
            self.logger.info("=" * 60)
            self.logger.info(f"Training finished. Total time: {total_time/3600:.4f} hours")
            self.logger.info(f"Best validation accuracy: {self.best_val_acc:.4f} (epoch {self.best_epoch+1})")
            self.logger.info("=" * 60)
            self.save_checkpoint(epoch, is_best=False, note="final_checkpoint")

            try:
                self._save_visualizations()
            except Exception as e:
                self.logger.error(f"Failed to save training visualizations: {e}")

            if not training_interrupted:
                try:
                    self._save_confusion_matrix()
                except Exception as e:
                    self.logger.error(f"Failed to save confusion matrix/report: {e}")
            else:
                self.logger.info("Skipping confusion matrix generation due to training interruption")

            self.logger.info(f"All results saved to: {self.result_dir}")
            self.logger.info(f"Best model: {Path(self.result_dir) / 'best_model.pth'}")

    def _update_scheduler(self, epoch: int, val_acc: float, switch_epoch: int) -> None:
        if epoch == switch_epoch:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max",
                patience=self.config["plateau_patience"],
                factor=self.config["plateau_factor"]
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

        axes[0, 0].plot(epochs, self.metrics.train_losses, label="Train Loss", linewidth=2)
        axes[0, 0].plot(epochs, self.metrics.val_losses, label="Val Loss", linewidth=2)
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        train_acc_pct = [acc * 100 for acc in self.metrics.train_accs]
        val_acc_pct = [acc * 100 for acc in self.metrics.val_accs]
        axes[0, 1].plot(epochs, train_acc_pct, label="Train Acc", linewidth=2)
        axes[0, 1].plot(epochs, val_acc_pct, label="Val Acc", linewidth=2)
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy (%)")
        axes[0, 1].set_title("Training and Validation Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=self.best_val_acc * 100, color="r", linestyle="--", alpha=0.7)

        axes[1, 0].plot(epochs, self.metrics.learning_rates, linewidth=2, color="green")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Learning Rate")
        axes[1, 0].set_title("Learning Rate Schedule")
        axes[1, 0].set_yscale("log")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(epochs, train_acc_pct, "o-", label="Train Acc", linewidth=2, markersize=3)
        axes[1, 1].plot(epochs, val_acc_pct, "s-", label="Val Acc", linewidth=2, markersize=3)
        axes[1, 1].fill_between(epochs, train_acc_pct, val_acc_pct, alpha=0.2, label="Gap")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Accuracy (%)")
        axes[1, 1].set_title("Overfitting Analysis")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(Path(self.result_dir) / "training_history.png", dpi=200, bbox_inches="tight")
        plt.close()
        self.logger.info("Training visualization results saved")

    def _save_confusion_matrix(self) -> None:
        self.logger.info("Generating confusion matrix...")
        best_path = os.path.join(self.result_dir, "best_model.pth")
        if not os.path.exists(best_path):
            self.logger.warning("Best model not found for confusion matrix generation.")
            return

        try:
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))
            self.model.eval()
            all_labels, all_preds = [], []
            with torch.no_grad():
                for images, labels in tqdm(self.val_loader, desc="Evaluating", leave=False):
                    images = images.to(self.device, non_blocking=True)
                    with autocast(enabled=self.use_amp, device_type=self.device.type):
                        outputs = self.model(images)
                    _, predicted = torch.max(outputs, 1)
                    all_labels.extend(labels.numpy())
                    all_preds.extend(predicted.cpu().numpy())

            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=self.class_names if self.class_names else [f'Class {i}' for i in range(self.num_classes)],
                yticklabels=self.class_names if self.class_names else [f'Class {i}' for i in range(self.num_classes)],
                cbar_kws={"label": "Count"}
            )
            plt.title(f"Confusion Matrix - {self.config['model']}\nValidation Accuracy: {self.best_val_acc:.4f}", fontsize=14)
            plt.xlabel("Predicted Label", fontsize=12)
            plt.ylabel("True Label", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(self.result_dir, "confusion_matrix.png"), dpi=200, bbox_inches="tight")
            plt.close()

            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=1)
            with open(os.path.join(self.result_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
                f.write(f"Validation Accuracy: {accuracy:.6f}\nWeighted F1 Score: {f1:.6f}\n")
            self.logger.info(f"Confusion matrix and classification report saved\nValidation Accuracy: {accuracy:.6f}\nWeighted F1 Score: {f1:.6f}")
        except Exception as e:
            self.logger.error(f"Failed to save confusion matrix due to: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="General Classification Training Script")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file (default: config.yaml)")
    parser.add_argument("--model", type=str, default=None, help="Model name (overrides config)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file or directory to resume training")
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.config):
        cfg = load_config(args.config)
    else:
        print(f"Warning: Config file {args.config} not found, using default configuration")
        cfg = get_defaults()

    if args.model:
        cfg["model"] = args.model
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.lr:
        cfg["lr"] = args.lr
    if args.resume:
        cfg["resume"] = args.resume

    set_seed(cfg["seed"])

    core = Trainer(cfg)
    core.train()


if __name__ == "__main__":
    main()
