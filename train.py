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
from utils import (
    set_seed, mixup_data, cutmix_data, KLLabelSmoothingLoss, 
    FocalLoss, GeneralizedCrossEntropy, SymmetricCrossEntropy, get_rng_device
)
from dataset import load_data
import model as models_module

class Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        self.device = self._select_computation_device()

        self.resume_path_obj, self.results_directory = self._create_results_directory_and_resume_path()

        self._setup_logging()

        self.best_validation_accuracy = 0.0
        self.best_epoch_index = 0
        self.early_stop_counter = 0

        self.metrics = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": []
        }

        self.scaler = (
            GradScaler()
            if self.config.get("use_amp") and self.device.type == "cuda"
            else None
        )

        self.rng = (
            torch.Generator(device=get_rng_device(self.device))
            .manual_seed(self.config["seed"])
        )

    def _select_device(self):
        device_choice = self.config.get("device", "auto")

        if device_choice == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"Using CUDA device: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                print("Using MPS device")
            else:
                device = torch.device("cpu")
                print("Using CPU device")
        else:
            device = torch.device(device_choice)
            print(f"Using specified device: {device_choice}")

        return device

    def _select_computation_device(self):
        return self._select_device()

    def _initialize_results_directory(self):
        resume_path = self.config.get("resume")

        if resume_path:
            resume_path_obj = Path(resume_path)
            if resume_path_obj.is_file():
                resume_path_obj = resume_path_obj.parent


        model_name = self.config["model"]
        if model_name == "extranet_scalable":
            model_name = (
                model_name
                + f"_{self.config.get('scalable_model_scale', 'tiny')}"
            )

        dir_path = (
            Path("./results")
            / f"{model_name}_{self.config['dataset']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        dir_path.mkdir(parents=True, exist_ok=True)

        return resume_path_obj, dir_path

    def _create_results_directory_and_resume_path(self):
        return self._initialize_results_directory()

    def _initialize_logger(self):
        log_file = str(self.results_directory) + "/training.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)

    def _setup_logging(self):
        return self._initialize_logger()

    def _prepare_data_loaders(self):

        self.train_loader = load_data(
            self.config["train_path"],
            self.config["batch_size"],
            self.config["num_workers"],
            "train",
            self.config,
        )

        self.val_loader = load_data(
            self.config["val_path"],
            self.config["batch_size"],
            self.config["num_workers"],
            "val",
            self.config,
        )

        self.class_names = getattr(self.train_loader.dataset, 'classes', None)
        if self.class_names is None and hasattr(self.train_loader.dataset, 'targets'):
            unique = sorted(set(self.train_loader.dataset.targets))
            self.class_names = [str(x) for x in unique]

        self.num_classes = len(self.class_names) if self.class_names is not None else 0
        print("Dataset summary:")
        print(f"  Number of classes: {self.num_classes}")
        if self.class_names is not None:
            print(f"  Class names: {self.class_names}")

        train_samples = len(self.train_loader.dataset)
        val_samples = len(self.val_loader.dataset)
        print(f"  Training set samples: {train_samples}")
        print(f"  Validation set samples: {val_samples}")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Data loader workers: {self.config.get('num_workers', 0)}")


        try:
            if hasattr(self.train_loader.dataset, 'targets'):
                counts = np.bincount(np.array(self.train_loader.dataset.targets))
                print("  Per-class sample counts (train):")
                for idx, cnt in enumerate(counts):
                    name = self.class_names[idx] if idx < len(self.class_names) else str(idx)
                    print(f"    {idx}: {name} -> {cnt}")
        except Exception:
            pass

    def _prepare_data_loaders_with_info(self):
        return self._prepare_data_loaders()

    def _construct_model(self):
        extra_map = {
            "extranet_ccse": models_module.ExtraNet_CCSE,
            "extranet": models_module.ExtraNet,
            "extranet_ccse_lite": models_module.ExtraNet_CCSE_Lite,
            "extranet_scalable": (
                lambda **k: models_module.ExtraNet_Scalable(
                    scale=self.config.get("scalable_model_scale", "tiny"),
                    **k
                )
            )
        }

        resnet_names = [
            "ccse_resnet18", "ccse_resnet34", "ccse_resnet50", "ccse_resnet101", "ccse_resnet152",
            "se_resnet18", "se_resnet34", "se_resnet50", "se_resnet101", "se_resnet152",
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
        ]

        for name in resnet_names:
            extra_map[name] = getattr(models_module, name)

        model_key = self.config["model"].lower()

        input_channels = self.config.get("grayscale_output_channels", 3)

        self.model = (
            extra_map[model_key](
                num_classes=self.num_classes,
                input_channels=input_channels
            )
            .to(self.device)
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"Model: {self.config['model']}")
        print(f"Input channels: {input_channels}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


        try:
            trainable_pct = 100.0 * trainable_params / max(1, total_params)
            print(f"Trainable params percent: {trainable_pct:.2f}%")
        except Exception:
            pass

        if self.config.get("use_torch_compile") and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
            print("Using PyTorch compile mode")

    def _build_model_architecture(self):
        return self._construct_model()

    def _configure_training_components(self, epoch=0):
        class_weights_tensor = None
        if (
            self.config.get("use_class_weights")
            and self.config["loss_type"] in ["focal", "ce", "sce"]
        ):
            class_weights_tensor = torch.tensor(
                self.config["class_weights"],
                device=self.device
            )

        label_smoothing = self.config.get("label_smoothing", 0.0)

        if (
            self.config.get("dynamic_label_smoothing")
            and self.config["loss_type"] == "kl"
        ):
            label_smoothing *= (
                1 - epoch / max(1, self.config["epochs"]) 
            )

        losses = {
            "focal": (
                lambda: FocalLoss(gamma=self.config["focal_gamma"], alpha=class_weights_tensor)
            ),
            "gce": (
                lambda: GeneralizedCrossEntropy(
                    num_classes=self.num_classes,
                    q=self.config["gce_q"]
                )
            ),
            "sce": (
                lambda: SymmetricCrossEntropy(
                    num_classes=self.num_classes,
                    class_weights=class_weights_tensor,
                    alpha=self.config.get("sce_alpha", 0.1),
                    beta=self.config.get("sce_beta", 1.0),
                    ce_only_epochs=self.config.get("sce_ce_only_epochs", 8),
                    label_smoothing=label_smoothing
                )
            ),
            "kl": (
                lambda: KLLabelSmoothingLoss(
                    classes=self.num_classes,
                    smoothing=label_smoothing
                )
            ),
            "ce": (
                lambda: nn.CrossEntropyLoss(
                    weight=class_weights_tensor,
                    label_smoothing=label_smoothing
                )
            )
        }

        self.criterion = losses[self.config["loss_type"]]().to(self.device)

        optimizer_map = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "sgd": optim.SGD
        }

        optimizer_kwargs = {
            "lr": self.config["lr"],
            "weight_decay": self.config["weight_decay"]
        }

        if self.config["optimizer"] == "sgd":
            optimizer_kwargs["momentum"] = self.config["momentum"]
        else:
            optimizer_kwargs["betas"] = tuple(self.config["betas"])

        self.optimizer = optimizer_map[self.config["optimizer"]](
            self.model.parameters(),
            **optimizer_kwargs
        )

        scheduler_map = {
            "cosine": (
                lambda: optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config["cosine_t_max"],
                    eta_min=self.config["cosine_eta_min"]
                )
            ),
            "cosine_restarts": (
                lambda: optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=self.config["cosine_t_max"],
                    T_mult=self.config.get("cosine_T_mult", 1),
                    eta_min=self.config["cosine_eta_min"]
                )
            ),
            "step": (
                lambda: optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=self.config["step_size"],
                    gamma=self.config["gamma"]
                )
            ),
            "plateau": (
                lambda: optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode="max",
                    patience=self.config["plateau_patience"],
                    factor=self.config["plateau_factor"]
                )
            )
        }

        self.scheduler = scheduler_map[self.config["scheduler"]]()

    def _setup_loss_optimizer_scheduler(self, epoch=0):
        return self._configure_training_components(epoch)

    def save_checkpoint(self, epoch_index, is_best=False, note=""):
        checkpoint = {
            "epoch": epoch_index,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_acc": self.best_validation_accuracy,
            "metrics": self.metrics,
            "rng": torch.get_rng_state(),
            "np": np.random.get_state(),
            "py": random.getstate(),
            "scaler": self.scaler.state_dict() if self.scaler else None
        }

        checkpoint_path = (
            self.resume_path_obj
            / f"{self.config['model']}_{self.config['dataset']}_latest.pth"
        )

        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_model_path = self.results_directory / "best_model.pth"
            torch.save(self.model.state_dict(), best_model_path)
            print(f" >>> new best model accuracy: {self.best_validation_accuracy:.8f} at epoch {epoch_index+1}, saved to {best_model_path}")

    def load_checkpoint(self, path):
        path_obj = Path(path)

        if path_obj.is_dir():
            path_obj = (
                path_obj
                / f"{self.config['model']}_{self.config['dataset']}_latest.pth"
            )

        checkpoint_data = torch.load(path_obj, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint_data["model"], strict=False)

        if "optimizer" in checkpoint_data:
            self.optimizer.load_state_dict(checkpoint_data["optimizer"])

        if "scheduler" in checkpoint_data:
            self.scheduler.load_state_dict(checkpoint_data["scheduler"])

        if self.scaler and checkpoint_data.get("scaler"):
            self.scaler.load_state_dict(checkpoint_data["scaler"])

        self.best_validation_accuracy = checkpoint_data.get("best_acc", 0.0)

        self.metrics = checkpoint_data.get("metrics", self.metrics)

        return checkpoint_data["epoch"] + 1

    def load_full_checkpoint(self, path):
        return self.load_checkpoint(path)

    def _execute_epoch(self, epoch_index, mode="train"):
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()

        data_loader = (
            self.train_loader if mode == "train" else self.val_loader
        )

        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0

        print(f"  Starting {mode} phase with {len(data_loader)} batches...")
        pbar = tqdm(data_loader, desc=f"Epoch {epoch_index+1} [{mode}]", leave=False)

        batch_count = 0
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.long().to(self.device, non_blocking=True)

            augmented_images = images
            mixed = False
            label_a = None
            label_b = None
            lam = 1.0

            if mode == "train":
                r1, r2 = torch.rand(2, generator=self.rng).tolist()
                mixup_alpha = self.config.get("mixup_alpha", 0)
                cutmix_alpha = self.config.get("cutmix_alpha", 0)

                if mixup_alpha > 0 and r1 < self.config.get("mixup_prob", 0.5):
                    augmented_images, label_a, label_b, lam = mixup_data(
                        images, labels, mixup_alpha, self.device, self.rng
                    )
                    mixed = True

                elif cutmix_alpha > 0 and r2 < self.config.get("cutmix_prob", 0.5):
                    augmented_images, label_a, label_b, lam = cutmix_data(
                        images, labels, cutmix_alpha, self.device, self.rng
                    )
                    mixed = True

            with autocast(enabled=bool(self.scaler), device_type=self.device.type):
                outputs = self.model(augmented_images)

                if mixed:
                    def mixed_loss_fn(pred, target):
                        if isinstance(self.criterion, SymmetricCrossEntropy):
                            return self.criterion(pred, target, epoch_index)
                        return self.criterion(pred, target)

                    loss = lam * mixed_loss_fn(outputs, label_a) + (
                        (1 - lam) * mixed_loss_fn(outputs, label_b)
                    )
                else:
                    if isinstance(self.criterion, SymmetricCrossEntropy):
                        loss = self.criterion(outputs, labels, epoch_index)
                    else:
                        loss = self.criterion(outputs, labels)

            if mode == "train":
                self.optimizer.zero_grad(set_to_none=True)

                if self.scaler:
                    self.scaler.scale(loss).backward()

                    if self.config.get("gradient_clip", 0) > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config["gradient_clip"]
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()

                    if self.config.get("gradient_clip", 0) > 0:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config["gradient_clip"]
                        )

                    self.optimizer.step()

            total_loss += loss.item() * images.size(0)

            _, predictions = outputs.max(1)

            total_samples += images.size(0)

            if mixed:
                total_correct += (
                    lam * predictions.eq(label_a).sum().item()
                    + (1 - lam) * predictions.eq(label_b).sum().item()
                )
            else:
                total_correct += predictions.eq(labels).sum().item()

            batch_count += 1
            if batch_count % max(1, len(data_loader) // 5) == 0:  # Print every 20% of batches
                batch_acc = 100 * predictions.eq(labels).sum().item() / images.size(0)
                print(f"    Batch {batch_count}/{len(data_loader)} - loss: {loss.item():.4f}, batch_acc: {batch_acc:.2f}%")

            pbar.set_postfix(
                loss=f"{total_loss/total_samples:.4f}",
                accuracy=f"{100*total_correct/total_samples:.2f}%"
            )

        return total_loss / total_samples, total_correct / total_samples

    def _evaluate_model(self):
        self.model.eval()
        all_predictions = []
        all_labels = []

        print(f"  Starting detailed evaluation with {len(self.val_loader)} batches...")
        pbar = tqdm(self.val_loader, desc="Detailed Evaluation", leave=False)

        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.long().to(self.device, non_blocking=True)

                outputs = self.model(images)
                _, predictions = outputs.max(1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return all_predictions, all_labels

    def run_training(self):

        self._prepare_data_loaders()
        self._construct_model()
        self._configure_training_components()

        with open(self.results_directory / "config.json", "w") as f:
            json.dump(self.config, f, indent=4)

        print(f"\n{'='*100}")
        print(f"DETAILED TRAINING CONFIGURATION")
        print(f"{'='*100}")
        print(f"Model Architecture: {self.config['model']}")
        print(f"Dataset: {self.config['dataset']}")
        print(f"Total Training Epochs: {self.config['epochs']}")
        print(f"Batch Size: {self.config['batch_size']}")
        print(f"Initial Learning Rate: {self.config['lr']}")
        print(f"Device: {self.device}")
        print(f"Results Directory: {self.results_directory}")
        print(f"Number of Classes: {self.num_classes}")
        print(f"Number of Training Samples: {len(self.train_loader.dataset)}")
        print(f"Number of Validation Samples: {len(self.val_loader.dataset)}")
        print(f"Number of Training Batches: {len(self.train_loader)}")
        print(f"Number of Validation Batches: {len(self.val_loader)}")
        print(f"Number of Data Loader Workers: {self.config.get('num_workers', 0)}")
        
        # Data augmentation settings
        if self.config.get("mixup_alpha", 0) > 0:
            print(f"Mixup Augmentation: Enabled (alpha={self.config['mixup_alpha']}, probability={self.config['mixup_prob']})")
        if self.config.get("cutmix_alpha", 0) > 0:
            print(f"Cutmix Augmentation: Enabled (alpha={self.config['cutmix_alpha']}, probability={self.config['cutmix_prob']})")
        
        # Regularization settings
        if self.config.get("label_smoothing", 0) > 0:
            print(f"Label Smoothing: {self.config['label_smoothing']}")
        
        if self.config.get("gradient_clip", 0) > 0:
            print(f"Gradient Clipping: {self.config['gradient_clip']}")
        
        if self.config.get("weight_decay", 0) > 0:
            print(f"Weight Decay: {self.config['weight_decay']}")
        
        # Scheduler settings
        if self.config.get("use_warmup", False):
            print(f"Warmup Epochs: {self.config.get('warmup_epochs', 0)}")
        
        if self.config.get("scheduler", "step") != "step":
            print(f"Learning Rate Scheduler: {self.config.get('scheduler')}")
        
        print(f"Early Stopping Patience: {self.config.get('patience', 50)} epochs")
        print(f"{'='*100}")

        start_epoch = 0
        self.training_start_time = time.time()  # Record start time for total training duration
        if (
            self.config.get("resume")
            and not self.config.get("if_first_train", True)
        ):
            print(f"Resuming training from checkpoint: {self.config['resume']}")
            start_epoch = self.load_checkpoint(self.config["resume"])
            print(f"Successfully loaded checkpoint. Starting from epoch {start_epoch+1}")

        self.logger.info(
            f"Start training from epoch {start_epoch+1}"
        )

        total_epochs = self.config["epochs"]
        print(f"Starting training for {total_epochs} epochs, resuming from epoch {start_epoch+1}")

        for epoch_index in range(start_epoch, self.config["epochs"]):
            current_lr = self.optimizer.param_groups[0]["lr"]
            progress_percent = ((epoch_index+1)/total_epochs)*100
            remaining_epochs = total_epochs - (epoch_index+1)
            print(f"\n{'='*100}")
            print(f"EPOCH {epoch_index+1}/{total_epochs} | Progress: {progress_percent:.1f}% | Remaining: {remaining_epochs} epochs")
            print(f"Learning Rate: {current_lr:.8f} | Best Validation Accuracy: {self.best_validation_accuracy:.4f} (Epoch {self.best_epoch_index+1})")
            print(f"Time Estimate: ~{remaining_epochs * 2:.1f} minutes remaining (approximate)")
            print(f"{'='*100}")
            
            epoch_start_time = time.time()

            if (
                self.config.get("use_warmup")
                and epoch_index < self.config.get("warmup_epochs", 0)
            ):
                lr = self.config["lr"] * (epoch_index + 1) / self.config["warmup_epochs"]
                print(f"  Warmup Phase: Adjusting learning rate from {self.config['lr']:.8f} to {lr:.8f}")

                for group in self.optimizer.param_groups:
                    group["lr"] = lr

            print(f"  Starting Training Phase...")
            train_loss, train_acc = self._execute_epoch(epoch_index, "train")
            print(f"  Training Phase Complete - Loss: {train_loss:.6f}, Accuracy: {train_acc:.4f}")
            
            print(f"  Starting Validation Phase...")
            val_loss, val_acc = self._execute_epoch(epoch_index, "val")
            print(f"  Validation Phase Complete - Loss: {val_loss:.6f}, Accuracy: {val_acc:.4f}")

            epoch_duration = time.time() - epoch_start_time
            remaining_epochs = max(0, total_epochs - (epoch_index + 1))
            eta_seconds = epoch_duration * remaining_epochs
            eta = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
            print(f"Epoch {epoch_index+1} duration: {epoch_duration:.1f}s, ETA remaining: {eta}")

            # GPU memory info if available
            try:
                if self.device.type == 'cuda':
                    allocated = torch.cuda.memory_allocated(self.device)
                    reserved = torch.cuda.memory_reserved(self.device)
                    print(f"GPU memory - allocated: {allocated/1024**2:.1f}MB, reserved: {reserved/1024**2:.1f}MB")
            except Exception:
                pass

            self.metrics["train_loss"].append(train_loss)
            self.metrics["train_accuracy"].append(train_acc)

            self.metrics["val_loss"].append(val_loss)
            self.metrics["val_accuracy"].append(val_acc)

            self.metrics["learning_rate"].append(self.optimizer.param_groups[0]["lr"])

            train_improvement = train_acc - self.metrics["train_accuracy"][-2] if len(self.metrics["train_accuracy"]) > 1 else 0
            val_improvement = val_acc - self.metrics["val_accuracy"][-2] if len(self.metrics["val_accuracy"]) > 1 else 0
            
            epoch_info = f"Epoch {epoch_index+1:03d} | Training Loss: {train_loss:.6f} Accuracy: {train_acc:.4f} (Δ: {train_improvement:+.4f}) | Validation Loss: {val_loss:.6f} Accuracy: {val_acc:.4f} (Δ: {val_improvement:+.4f})"
            print(f"  {epoch_info}")
            
            self.logger.info(epoch_info)

            if epoch_index == self.config.get("switch_to_plateau_epoch"):
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode="max",
                    patience=self.config["plateau_patience"],
                    factor=self.config["plateau_factor"]
                )
                print(f"Plateau scheduler initialized with patience {self.config['plateau_patience']} and factor {self.config['plateau_factor']}")

            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_acc)
            else:
                self.scheduler.step()

            is_best_now = val_acc > self.best_validation_accuracy

            if is_best_now:
                self.best_validation_accuracy = val_acc
                self.best_epoch_index = epoch_index
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            self.save_checkpoint(epoch_index, is_best_now)

            if self.early_stop_counter >= self.config.get("patience", 50):
                print(f"\nEarly stopping triggered after {epoch_index+1} epochs. No improvement for {self.early_stop_counter} epochs.")
                break

        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED")
        print(f"Total epochs run: {epoch_index+1}")
        print(f"Best validation accuracy: {self.best_validation_accuracy:.4f} at epoch {self.best_epoch_index+1}")
        print(f"Final validation accuracy: {val_acc:.4f}")
        print(f"Final training accuracy: {train_acc:.4f}")
        print(f"Results saved to: {self.results_directory}")
        print(f"{'='*60}")
        
        self._finalize_and_save_results()

    def _finalize_and_save_results(self):
        print(f"Saving final checkpoint and results...")
        self.save_checkpoint(self.best_epoch_index, note="final")

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        epochs = range(1, len(self.metrics["train_loss"]) + 1)

        ax[0].plot(epochs, self.metrics["train_loss"], label="Train")
        ax[0].plot(epochs, self.metrics["val_loss"], label="Val")
        ax[0].legend()
        ax[0].set_title("Loss")

        ax[1].plot(epochs, self.metrics["train_accuracy"], label="Train")
        ax[1].plot(epochs, self.metrics["val_accuracy"], label="Val")
        ax[1].legend()
        ax[1].set_title("Acc")

        history_path = self.results_directory / "history.png"
        plt.savefig(history_path)
        print(f"Training history plot saved to: {history_path}")

        print(f"Generating detailed evaluation metrics...")
        all_predictions, all_labels = self._evaluate_model()
        
        avg_train_acc = sum(self.metrics['train_accuracy']) / len(self.metrics['train_accuracy'])
        avg_val_acc = sum(self.metrics['val_accuracy']) / len(self.metrics['val_accuracy'])
        max_train_acc = max(self.metrics['train_accuracy'])
        max_val_acc = max(self.metrics['val_accuracy'])
        
        improvement_from_start = self.metrics['val_accuracy'][-1] - self.metrics['val_accuracy'][0]
        
        detailed_accuracy = accuracy_score(all_labels, all_predictions)
        detailed_f1_score = f1_score(all_labels, all_predictions, average='weighted')
        
        total_training_time = time.time() - self.training_start_time
        total_training_time_str = time.strftime('%H:%M:%S', time.gmtime(total_training_time))
        
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names if self.class_names else [f'Class {i}' for i in range(self.num_classes)],
            yticklabels=self.class_names if self.class_names else [f'Class {i}' for i in range(self.num_classes)]
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        cm_path = self.results_directory / "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        print(f"Confusion matrix saved to: {cm_path}")
        
        report = classification_report(
            all_labels, 
            all_predictions, 
            target_names=self.class_names if self.class_names else [f'Class {i}' for i in range(self.num_classes)],
            output_dict=True
        )
        
        report_path = self.results_directory / "classification_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"Classification report saved to: {report_path}")
        
        
        final_msg = (
            f"\n{'='*100}\n"
            f"TRAINING COMPLETED SUCCESSFULLY!\n"
            f"{'='*100}\n"
            f"Best Validation Accuracy: {self.best_validation_accuracy:.4f} at Epoch {self.best_epoch_index+1}\n"
            f"Final Training Accuracy: {self.metrics['train_accuracy'][-1]:.4f}\n"
            f"Final Validation Accuracy: {self.metrics['val_accuracy'][-1]:.4f}\n"
            f"Detailed Accuracy (recomputed): {detailed_accuracy:.4f}\n"
            f"Weighted F1 Score: {detailed_f1_score:.4f}\n"
            f"Average Training Accuracy: {avg_train_acc:.4f}\n"
            f"Average Validation Accuracy: {avg_val_acc:.4f}\n"
            f"Highest Training Accuracy: {max_train_acc:.4f}\n"
            f"Highest Validation Accuracy: {max_val_acc:.4f}\n"
            f"Improvement from Start: {improvement_from_start:+.4f}\n"
            f"Total Training Time: {total_training_time_str} (HH:MM:SS)\n"
            f"Results saved to: {self.results_directory}\n"
            f"{'='*100}\n"
            f"DETAILED CLASSIFICATION REPORT:\n"
            f"{'-'*100}\n"
            f"{report}\n"
            f"{'-'*100}"
        )
        print(final_msg)
        self.logger.info(final_msg)

def main():
    p = argparse.ArgumentParser(description="Train and validate a deep learning classification model on custom datasets.")
    p.add_argument("--config", default="config.yaml", type=str, help="Path to config file (YAML or JSON).")

    p.add_argument("--model", type=str, help="Model architecture name (e.g., 'resnet18', 'ccse_resnet50').")
    p.add_argument("--epochs", type=int, help="Number of total training epochs to run.")
    p.add_argument("--batch_size", type=int, help="Batch size per optimizer step.")
    p.add_argument("--lr", type=float, help="Learning rate for optimizer.")
    p.add_argument("--resume", type=str, help="Path to checkpoint to resume training from.")

    args = p.parse_args()

    config_dict = (
        config.load_config(args.config)
        if os.path.exists(args.config)
        else config.get_defaults()
    )

    for key, value in vars(args).items():
        if value is not None:
            config_dict[key] = value

    set_seed(config_dict["seed"])

    Trainer(config_dict).run_training()

if __name__ == "__main__":
    main()
