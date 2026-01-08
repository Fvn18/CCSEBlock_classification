import os
import sys
import time
import logging
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

from config import get_defaults, load_config
from utils import set_seed, mixup_data, cutmix_data
from dataset import load_data
from model import *
try:
    from torchinfo import summary
except ImportError:
    summary = None


class Trainer:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.config = cfg
        self.device = self._setup_device()
        
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            
        self.output_dir = self._create_result_dir()
        self._setup_logging()

        self.best_val_acc: float = 0.0
        self.best_epoch: int = 0    
        self.patience_counter: int = 0

        self.use_amp = cfg["basic"].get("use_amp", False) and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

    def _setup_device(self) -> torch.device:
        device_arg = self.config["basic"].get("device", "auto")
        if device_arg == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device_arg
        return torch.device(device)

    def _create_result_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.config["model"]["name"]
        dir_name = f"{model_name}_{self.config['basic']['dataset']}_{timestamp}"
        result_dir = Path("results") / dir_name
        result_dir.mkdir(parents=True, exist_ok=True)
        
        with open(result_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=4)
            
        return result_dir

    def _setup_logging(self) -> None:
        log_file = self.output_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Artifacts dir: {self.output_dir}")

    def _build_components(self):
        self.train_loader = load_data(
            self.config["data"]["train_path"], self.config["training"]["batch_size"],
            self.config["basic"]["num_workers"], "train", self.config
        )
        self.val_loader = load_data(
            self.config["data"]["val_path"], self.config["training"]["batch_size"],
            self.config["basic"]["num_workers"], "val", self.config
        )
        
        if hasattr(self.train_loader.dataset, 'classes'):
            self.num_classes = len(self.train_loader.dataset.classes)
        else:
            self.num_classes = len(set(self.train_loader.dataset.targets))
        self.logger.info(f"Num Classes: {self.num_classes}")

        self.logger.info(f"Building Model: {self.config['model']['name']}")
        input_channels = self.config["data"].get("grayscale_output_channels", 3)
        
        model_cls = getattr(sys.modules[__name__], self.config['model']['name'])
        self.model = model_cls(
            num_classes=self.num_classes, 
            input_channels=input_channels, 
            pretrained=self.config["model"].get("pretrained", False)
        )

        self.model = self.model.to(self.device)
        
        self.criterion = self._get_loss_fn()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

    def _get_loss_fn(self):
        loss_type = self.config["training"].get("loss_type", "ce")
        if loss_type == "ce":
            return nn.CrossEntropyLoss(label_smoothing=self.config["training"].get("label_smoothing", 0.0))
        elif loss_type == "sce":
            from utils import SymmetricCrossEntropy
            return SymmetricCrossEntropy(
                num_classes=self.num_classes,
                alpha=self.config["training"].get("sce_alpha", 0.1),
                beta=self.config["training"].get("sce_beta", 1.0),
                label_smoothing=self.config["training"].get("label_smoothing", 0.0)
            ).to(self.device)
        else:
            return nn.CrossEntropyLoss()

    def _get_optimizer(self):
        if self.config["optimizer"]["optimizer"] == "adamw":
            return optim.AdamW(self.model.parameters(), lr=self.config["optimizer"]["lr"], weight_decay=self.config["optimizer"]["weight_decay"])
        elif self.config["optimizer"]["optimizer"] == "sgd":
            return optim.SGD(self.model.parameters(), lr=self.config["optimizer"]["lr"], momentum=self.config["optimizer"]["momentum"], weight_decay=self.config["optimizer"]["weight_decay"])
        else:
            return optim.Adam(self.model.parameters(), lr=self.config["optimizer"]["lr"])

    def _get_scheduler(self):
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config["training"]["epochs"], eta_min=1e-6
        )

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        mixup_alpha = self.config["training"].get("mixup_alpha", 0)
        cutmix_alpha = self.config["training"].get("cutmix_alpha", 0)
        do_mixup = mixup_alpha > 0 or cutmix_alpha > 0

        pbar = tqdm(self.train_loader, desc=f"Ep {epoch+1}", leave=False)
        
        for images, labels in pbar:
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

            mixed = False
            labels_a, labels_b, lam = None, None, 1.0
            
            if do_mixup and torch.rand(1).item() < self.config["training"].get("mixup_prob", 0.5):
                if cutmix_alpha > 0 and mixup_alpha > 0:
                    if torch.rand(1).item() < 0.5:
                        images, labels_a, labels_b, lam = cutmix_data(images, labels, cutmix_alpha, self.device)
                    else:
                        images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha, self.device)
                    mixed = True
                elif cutmix_alpha > 0:
                    images, labels_a, labels_b, lam = cutmix_data(images, labels, cutmix_alpha, self.device)
                    mixed = True
                elif mixup_alpha > 0:
                    images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha, self.device)
                    mixed = True

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp, device_type=self.device.type):
                outputs = self.model(images)
                
                if mixed:
                    loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
                else:
                    loss = self.criterion(outputs, labels)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size
            
            _, predicted = outputs.max(1)
            if mixed:
                correct += (lam * predicted.eq(labels_a).sum().float() + (1 - lam) * predicted.eq(labels_b).sum().float()).item()
            else:
                correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        return running_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        
        for images, labels in self.val_loader:
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            
            with autocast(enabled=self.use_amp, device_type=self.device.type):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return running_loss / total, correct / total

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_acc": self.best_val_acc,
            "config": self.config
        }
        
        torch.save(state, self.output_dir / "last.pth")
        if is_best:
            torch.save(state, self.output_dir / "best.pth")
            self.logger.info(f"Saved Best Model: Acc {val_acc:.4f}")

    def run(self):
        self._build_components()
        
        epochs = self.config["training"]["epochs"]
        self.logger.info(f"Start Training for {epochs} epochs...")
        
        self.logger.info(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<10} {'Val Loss':<10} {'Val Acc':<10} {'LR':<12}")
        self.logger.info("-" * 72)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.evaluate()
            
            self.scheduler.step()

            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            time_elapsed = time.time() - epoch_start_time
            lr = self.optimizer.param_groups[0]["lr"]

            self.logger.info(
                f"{epoch+1:03d}/{epochs:<4} "
                f" {train_loss:<12.4f} {train_acc:<10.4f} "
                f" {val_loss:<10.4f} {val_acc:<10.4f} "
                f" {lr:<12.6f}"
            )
            
            self.save_checkpoint(epoch, val_acc, is_best)
            
            patience = self.config["training"].get("patience", 20)
            if self.patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        total_time = time.time() - start_time
        
        self.model.load_state_dict(torch.load(self.output_dir / "best.pth", map_location=self.device)["state_dict"])
        self.model.eval()
        all_labels, all_preds = [], []
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.numpy())
                all_preds.extend(predicted.cpu().numpy())
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=1)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info("=" * 72)
        self.logger.info("Training Summary:")
        self.logger.info(f"  Model: {self.config['model']['name']}")
        self.logger.info(f"  Dataset: {self.config['basic']['dataset']}")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        self.logger.info(f"  Best validation accuracy: {self.best_val_acc:.6f} (epoch {self.best_epoch+1})")
        self.logger.info(f"  Weighted F1 score: {f1:.6f}")
        self.logger.info(f"  Total training time: {total_time/3600:.4f} hours")
        
        import json
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        results_path = self.output_dir / "results.txt"
        with open(results_path, 'w') as f:
            f.write("Training Results Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Model: {self.config['model']['name']}\n")
            f.write(f"Dataset: {self.config['basic']['dataset']}\n")
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,}\n")
            f.write(f"Best validation accuracy: {self.best_val_acc:.6f} (epoch {self.best_epoch+1})\n")
            f.write(f"Weighted F1 score: {f1:.6f}\n")
            f.write(f"Total training time: {total_time/3600:.4f} hours\n")
            f.write(f"Final validation accuracy: {val_acc:.6f}\n")
            f.write(f"Final training accuracy: {train_acc:.6f}\n")
            f.write(f"Final training loss: {train_loss:.6f}\n")
            f.write(f"Final validation loss: {val_loss:.6f}\n")
        
        last_model_path = self.output_dir / "last_model.pth"
        torch.save({
            'epoch': self.best_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
        }, last_model_path)
        
        self.logger.info("=" * 72)
        self.logger.info(f"All results saved to: {self.output_dir}")
        self.logger.info(f"Configuration saved to: {config_path}")
        self.logger.info(f"Results summary saved to: {results_path}")
        self.logger.info(f"Best model saved to: {self.output_dir / 'best.pth'}")
        self.logger.info(f"Last model saved to: {last_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    if cfg is None: 
        cfg = get_defaults()
        
    set_seed(cfg["basic"].get("seed", 42))
    trainer = Trainer(cfg)
    trainer.run()