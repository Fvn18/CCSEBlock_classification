from __future__ import annotations
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Mapping
import re

_scientific_pattern = re.compile(r'^[+-]?(?:\d+\.?\d*|\.\d+)[eE][+-]?\d+$')

def convert_scientific_notation(x: Any) -> Any:
    if isinstance(x, str) and _scientific_pattern.match(x):
        try:
            return float(x)
        except (ValueError, OverflowError):
            return x
    return x

def walk_convert(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: walk_convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [walk_convert(v) for v in obj]
    return convert_scientific_notation(obj)

def deep_merge(a: Dict[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    for k, v in b.items():
        if isinstance(v, Mapping) and isinstance(a.get(k), dict):
            deep_merge(a[k], v)
        else:
            a[k] = v
    return a

DEFAULT = {
    "basic": dict(dataset="fer2013", device="auto", num_workers=8, seed=33, use_amp=True),
    "model": dict(name="extranet_scalable", scalable_model_scale="tiny", use_torch_compile=False),
    "data": dict(root="./fer2013", train_path="./fer2013/train", val_path="./fer2013/val", test_path="./fer2013/test"),
    "augmentation": dict(
        use_tta=True, tta_angles=[5.0, -5.0, 2.5, -2.5],
        resize_size=96, crop_size=80, horizontal_flip_prob=0.5,
        random_rotation_degrees=10, random_erasing_prob=0.25, color_jitter_prob=0.0,
        random_affine_prob=0.3, gaussian_blur_prob=0.2,
        color_jitter_brightness=0.2, color_jitter_contrast=0.2, color_jitter_saturation=0.1,
        random_affine_translate=[0.1, 0.1], random_affine_scale=[0.9, 1.1],
        gaussian_blur_sigma=[0.1, 1.5],
        use_randaugment=False, randaugment_n=2, randaugment_m=8,
        augmentationon=True,
        enable_resize=True,
        enable_random_crop=True,
        random_crop_padding=4,
        enable_rotation=False,
        enable_color_jitter=False,
        enable_affine=False,
        enable_blur=False,
        enable_erasing=False
    ),
    "training": dict(batch_size=128, epochs=450, lr=0.00055, weight_decay=0.045, patience=999, gradient_clip=1.0),
    "optimizer": dict(name="adamw", betas=[0.9, 0.999], momentum=0.9),
    "scheduler": dict(
        name="cosine", switch_to_plateau_epoch=999,
        use_warmup=True, warmup_epochs=15,
        cosine_t_max=450, cosine_eta_min=5e-07,
        step_size=30, gamma=0.1, plateau_patience=10, plateau_factor=0.1
    ),
    "loss": dict(type="sce", focal_gamma=2.1, gce_q=0.9, sce_alpha=0.8, sce_beta=1.0,
                 sce_ce_only_epochs=10, label_smoothing=0.15, dynamic_label_smoothing=False),
    "strategies": dict(mixup_alpha=0.25, mixup_prob=0.5, cutmix_alpha=0.9, cutmix_prob=0.5,
                       use_class_weights=False, class_weights=[1.4, 4.5, 1.3, 1.1, 1.2, 1.3, 1.5]),
    "logging": dict(save_checkpoints=False, save_freq=10, log_freq=5, save_best_only=True),
    "normalization": dict(mean=[0.5], std=[0.5]),
}

FLAT_KEYS = {
    "dataset": ("basic", "dataset"),
    "device": ("basic", "device"),
    "num_workers": ("basic", "num_workers"),
    "seed": ("basic", "seed"),
    "use_amp": ("basic", "use_amp"),
    "model": ("model", "name"),
    "scalable_model_scale": ("model", "scalable_model_scale"),
    "use_torch_compile": ("model", "use_torch_compile"),
    "data_root": ("data", "root"),
    "train_path": ("data", "train_path"),
    "val_path": ("data", "val_path"),
    "test_path": ("data", "test_path"),
    "grayscale_output_channels": ("data", "grayscale_output_channels"),
    "use_tta": ("augmentation", "use_tta"),
    "use_tencrop": ("augmentation", "use_tencrop"),
    "tta_angles": ("augmentation", "tta_angles"),
    "resize_size": ("augmentation", "resize_size"),
    "crop_size": ("augmentation", "crop_size"),
    "horizontal_flip_prob": ("augmentation", "horizontal_flip_prob"),
    "random_rotation_degrees": ("augmentation", "random_rotation_degrees"),
    "random_erasing_prob": ("augmentation", "random_erasing_prob"),
    "color_jitter_prob": ("augmentation", "color_jitter_prob"),
    "random_affine_prob": ("augmentation", "random_affine_prob"),
    "gaussian_blur_prob": ("augmentation", "gaussian_blur_prob"),
    "color_jitter_brightness": ("augmentation", "color_jitter_brightness"),
    "color_jitter_contrast": ("augmentation", "color_jitter_contrast"),
    "color_jitter_saturation": ("augmentation", "color_jitter_saturation"),
    "random_affine_translate": ("augmentation", "random_affine_translate"),
    "random_affine_scale": ("augmentation", "random_affine_scale"),
    "gaussian_blur_sigma": ("augmentation", "gaussian_blur_sigma"),
    "use_randaugment": ("augmentation", "use_randaugment"),
    "randaugment_n": ("augmentation", "randaugment_n"),
    "randaugment_m": ("augmentation", "randaugment_m"),
    "batch_size": ("training", "batch_size"),
    "epochs": ("training", "epochs"),
    "lr": ("training", "lr"),
    "weight_decay": ("training", "weight_decay"),
    "patience": ("training", "patience"),
    "gradient_clip": ("training", "gradient_clip"),
    "optimizer": ("optimizer", "name"),
    "betas": ("optimizer", "betas"),
    "momentum": ("optimizer", "momentum"),
    "scheduler": ("scheduler", "name"),
    "switch_to_plateau_epoch": ("scheduler", "switch_to_plateau_epoch"),
    "use_warmup": ("scheduler", "use_warmup"),
    "warmup_epochs": ("scheduler", "warmup_epochs"),
    "cosine_t_max": ("scheduler", "cosine_t_max"),
    "cosine_eta_min": ("scheduler", "cosine_eta_min"),
    "step_size": ("scheduler", "step_size"),
    "gamma": ("scheduler", "gamma"),
    "plateau_patience": ("scheduler", "plateau_patience"),
    "plateau_factor": ("scheduler", "plateau_factor"),
    "loss_type": ("loss", "type"),
    "focal_gamma": ("loss", "focal_gamma"),
    "gce_q": ("loss", "gce_q"),
    "sce_alpha": ("loss", "sce_alpha"),
    "sce_beta": ("loss", "sce_beta"),
    "sce_ce_only_epochs": ("loss", "sce_ce_only_epochs"),
    "label_smoothing": ("loss", "label_smoothing"),
    "dynamic_label_smoothing": ("loss", "dynamic_label_smoothing"),
    "mixup_alpha": ("strategies", "mixup_alpha"),
    "mixup_prob": ("strategies", "mixup_prob"),
    "cutmix_alpha": ("strategies", "cutmix_alpha"),
    "cutmix_prob": ("strategies", "cutmix_prob"),
    "use_class_weights": ("strategies", "use_class_weights"),
    "class_weights": ("strategies", "class_weights"),
    "save_checkpoints": ("logging", "save_checkpoints"),
    "save_freq": ("logging", "save_freq"),
    "log_freq": ("logging", "log_freq"),
    "save_best_only": ("logging", "save_best_only"),
    "mean": ("normalization", "mean"),
    "std": ("normalization", "std"),
    "enable_resize": ("augmentation", "enable_resize"),
    "enable_random_crop": ("augmentation", "enable_random_crop"),
    "random_crop_padding": ("augmentation", "random_crop_padding"),
    "enable_rotation": ("augmentation", "enable_rotation"),
    "enable_color_jitter": ("augmentation", "enable_color_jitter"),
    "enable_affine": ("augmentation", "enable_affine"),
    "enable_blur": ("augmentation", "enable_blur"),
    "enable_erasing": ("augmentation", "enable_erasing"),

}

TUPLE_FIELDS = {
    "betas",
    "mean", "std",
    "random_affine_translate", "random_affine_scale", "gaussian_blur_sigma",
}

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file {p} does not exist")

    if path.endswith('.yaml') or path.endswith('.yml'):
        with p.open("r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        user = walk_convert(user)

        cfg = deep_merge({k: (v.copy() if isinstance(v, dict) else v) for k, v in DEFAULT.items()}, user)

        out: Dict[str, Any] = {}
        for flat_k, (sec, k) in FLAT_KEYS.items():
            out[flat_k] = cfg[sec][k]
    else:
        with p.open("r", encoding="utf-8") as f:
            out = json.load(f)
        
        for flat_k, (sec, k) in FLAT_KEYS.items():
            if flat_k not in out:
                out[flat_k] = DEFAULT[sec][k]

    for k in list(out.keys()):
        if k in TUPLE_FIELDS and isinstance(out[k], list):
            out[k] = tuple(out[k])

    return out



def get_defaults() -> Dict[str, Any]:
    return load_config("config.yaml")


def print_config(config: Dict[str, Any]) -> None:
    print("\n" + "="*60)
    print("FER2013 Experiment Configuration:")
    print("="*60)

    categories = {
        'Basic Configuration': ['dataset', 'device', 'num_workers', 'seed', 'use_amp'],
        'Model Configuration': ['model', 'scalable_model_scale', 'use_torch_compile'],
        'Data Configuration': ['data_root', 'train_path', 'val_path', 'test_path'],
        'Training Configuration': ['batch_size', 'epochs', 'lr', 'weight_decay', 'patience', 'gradient_clip'],
        'Optimizer Configuration': ['optimizer', 'betas', 'momentum'],
        'Scheduler Configuration': ['scheduler', 'use_warmup', 'warmup_epochs', 'cosine_t_max', 'cosine_eta_min'],
        'Loss Function Configuration': ['loss_type', 'focal_gamma', 'sce_alpha', 'sce_beta', 'label_smoothing'],
        'Data Augmentation Configuration': ['use_tta', 'resize_size', 'crop_size', 'horizontal_flip_prob', 'use_randaugment'],
        'Training Strategies': ['mixup_alpha', 'mixup_prob', 'cutmix_alpha', 'cutmix_prob', 'use_class_weights']
    }

    for category, keys in categories.items():
        print(f"\n{category}:")
        print("-" * 40)
        for key in keys:
            if key in config:
                value = config[key]
                if isinstance(value, (list, tuple)) and len(value) > 5:
                    value_str = f"[{value[0]}, {value[1]}, ..., {value[-1]}] ({len(value)} items)"
                else:
                    value_str = str(value)
                print(f"  {key:25s}: {value_str}")

    print("="*60 + "\n")
