import os
import sys
import time
import logging
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

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

class Trainer:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.config = cfg
        self.device = self._setup_device()
        
            
        self.output_dir = self._create_result_dir()
        self._setup_logging()

        self.best_val_acc: float = 0.0
        self.best_epoch: int = 0    
        self.patience_counter: int = 0

        self.use_amp = cfg["basic"].get("use_amp", False) and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        
        self.accum_steps = cfg["training"].get("accumulation_steps", 1)
        self.logger.info(f"Using AMP: {self.use_amp}")
        self.logger.info(f"Gradient Accumulation Steps: {self.accum_steps}")

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
        dataset_name = self.config['basic']['dataset']
        dir_name = f"{model_name}_{dataset_name}_{timestamp}"
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

    def _build_components(self):
        self.logger.info("Loading Data...")
        self.train_loader = load_data(
            self.config["data"]["train_path"], 
            self.config["training"]["batch_size"],
            self.config["basic"]["num_workers"], 
            "train", 
            self.config
        )
        self.val_loader = load_data(
            self.config["data"]["val_path"], 
            self.config["training"]["batch_size"],
            self.config["basic"]["num_workers"], 
            "val", 
            self.config
        )
        
        if hasattr(self.train_loader.dataset, 'classes'):
            self.num_classes = len(self.train_loader.dataset.classes)
        elif hasattr(self.train_loader.dataset, 'targets'):
             self.num_classes = len(set(self.train_loader.dataset.targets))
        else:
             self.num_classes = 1000 
        
        self.logger.info(f"Num Classes: {self.num_classes}")

        self.logger.info(f"Building Model: {self.config['model']['name']}")
        input_channels = self.config["data"].get("grayscale_output_channels", 3)
        
        try:
            model_cls = getattr(sys.modules['model'], self.config['model']['name'])
        except AttributeError:
            model_cls = globals().get(self.config['model']['name'])
            if model_cls is None:
                raise ValueError(f"Model {self.config['model']['name']} not found in model.py")

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
        label_smoothing = self.config["training"].get("label_smoothing", 0.0)
        
        self.logger.info(f"Loss: {loss_type}, Label Smoothing: {label_smoothing}")
        
        if loss_type == "ce":
            return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        return nn.CrossEntropyLoss()

    def _get_optimizer(self):
        opt_name = self.config["optimizer"]["optimizer"].lower()
        lr = self.config["optimizer"]["lr"]
        wd = self.config["optimizer"]["weight_decay"]
        
        self.logger.info(f"Optimizer: {opt_name}, LR: {lr}, WD: {wd}")
        
        if opt_name == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            return optim.SGD(
                self.model.parameters(), lr=lr, 
                momentum=self.config["optimizer"].get("momentum", 0.9), 
                weight_decay=wd,
                nesterov=self.config["optimizer"].get("nesterov", True)
            )
        else:
            return optim.Adam(self.model.parameters(), lr=lr)

    def _get_scheduler(self):
        epochs = self.config["training"]["epochs"]
        warmup = self.config["scheduler"].get("warmup_epochs", 0)
        
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=epochs, 
            eta_min=self.config["scheduler"]["min_lr"]
        )

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        mixup_prob = self.config["training"].get("mixup_prob", 0.0)
        cutmix_prob = self.config["training"].get("cutmix_prob", 0.0)
        mixup_alpha = self.config["training"].get("mixup_alpha", 0.0)
        cutmix_alpha = self.config["training"].get("cutmix_alpha", 0.0)
        
        do_mixup = (mixup_alpha > 0 or cutmix_alpha > 0) and (mixup_prob > 0 or cutmix_prob > 0)

        pbar = tqdm(self.train_loader, desc=f"Ep {epoch+1}", leave=False)
        
        self.optimizer.zero_grad() 

        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

            mixed = False
            labels_a, labels_b, lam = labels, labels, 1.0
            
            if do_mixup and torch.rand(1).item() < mixup_prob:
                mixed = True
                use_cutmix = cutmix_alpha > 0 and (torch.rand(1).item() < cutmix_prob or mixup_alpha == 0)
                
                if use_cutmix:
                    images, labels_a, labels_b, lam = cutmix_data(images, labels, cutmix_alpha, self.device)
                else:
                    images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha, self.device)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                
                if mixed:
                    loss = lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)
                else:
                    loss = self.criterion(outputs, labels)

                loss = loss / self.accum_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                if (i + 1) % self.accum_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                if (i + 1) % self.accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            current_loss = loss.item() * self.accum_steps
            batch_size = images.size(0)
            running_loss += current_loss * batch_size
            total += batch_size
            
            _, predicted = outputs.max(1)
            if mixed:
                part_a = predicted.eq(labels_a).sum().float()
                part_b = predicted.eq(labels_b).sum().float()
                correct += (lam * part_a + (1 - lam) * part_b).item()
            else:
                correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({"Loss": f"{current_loss:.4f}", "Lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"})

        return running_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        
        for images, labels in self.val_loader:
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            
            with autocast(device_type=self.device.type, enabled=self.use_amp):
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
            self.logger.info(f"Saved Best Model: {val_acc:.4f}")

    def run(self):
        self._build_components()
        
        epochs = self.config["training"]["epochs"]
        self.logger.info(f"Start Training: {epochs} Epochs")
        self.logger.info("-" * 90)
        self.logger.info(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<10} {'Val Acc':<10} {'LR':<10} {'Time'}")
        self.logger.info("-" * 90)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
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
                
            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]

            self.logger.info(
                f"{epoch+1:03d}/{epochs:<4} "
                f"{train_loss:<12.4f} {train_acc:<12.4f} "
                f"{val_loss:<10.4f} {val_acc:<10.4f} "
                f"{lr:<10.2e} {epoch_time:.0f}s"
            )
            
            self.save_checkpoint(epoch, val_acc, is_best)
            
            if self.patience_counter >= self.config["training"].get("patience", 999):
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        total_time = time.time() - start_time
        self.logger.info(f"Training Finished. Total Time: {total_time/3600:.2f} Hours")
        self.logger.info(f"Best Val Acc: {self.best_val_acc:.4f} at Epoch {self.best_epoch+1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    if cfg is None: 
        print("Config file not found or empty.")
        sys.exit(1)
        
    set_seed(cfg["basic"].get("seed", 42))
    trainer = Trainer(cfg)
    trainer.run()