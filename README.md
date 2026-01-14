# CCSEBlock Classification

A PyTorch implementation of Cross-Channel Segmentation Excitation (CCSE) Block for image classification tasks. This project includes various CNN architectures enhanced with attention mechanisms and advanced training techniques.

## Features

- **CCSE Block**: Cross-Channel Segmentation Excitation Block for improved feature representation
- **Multiple Architectures**: Support for ResNet, SE-ResNet, CCSE-ResNet, and ExtraNet variants
- **Advanced Training**: Mixup, Cutmix, and other data augmentation techniques
- **Model Scalability**: Support for scalable model variants
- **Comprehensive Evaluation**: F1 score, confusion matrix, and classification reports

## Installation
# CCSEBlock Classification

This repository provides a PyTorch implementation of the Cross-Channel Segmentation Excitation (CCSE) block and several CNN variants that integrate attention modules for improved image classification.

Key points:

- Lightweight library for experimenting with CCSE, SE and related attention blocks.
- Configurable training pipeline with common augmentation and regularization techniques.
- Example model variants based on ResNet and custom ExtraNet architectures.

## Features

- CCSE block: cross-channel segmentation excitation for richer feature modulation.
- Multiple attention modules: SE, ECA, CoordAtt, SimAM, CCFiLM and more.
- Training utilities: mixup, cutmix, AMP support and scheduling.
- Logging and checkpoints: save best/last models and evaluation results.

## Quickstart

1. Clone the repository and enter the folder:

```bash
git clone <repository-url>
cd CCSEBlock_classification
```

2. (Recommended) create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

4. Train a model with the default configuration:

```bash
python train.py
```

## Configuration

Training and model settings are defined in `config.yaml` and `config.py`. Common options include:

- `model`: choose a model variant (example names: `ccse_resnet18`, `se_resnet50`, `simse_resnet18`, `extra_net_ccse`).
- Dataset paths, input size and normalization parameters.
- Optimizer, learning rate schedule and batch size.
- Flags for AMP, mixup, cutmix and other augmentations.

Adjust `config.yaml` to match your dataset and experiment settings.

## Models Included

The `model/` directory contains implementations for the attention modules and example architectures, including:

- `CCSEBlock.py`, `SEBlock.py`, `ECABlock.py`, `CoordAttBlock.py` — attention blocks.
- `ResNet.py` — ResNet backbone variants.
- `SimSE.py`, `SimAM.py`, `CCFiLM.py`, `SimCCFiLM.py` — experimental modules.

## Project Layout

Top-level files and folders:

- `train.py` — main training script.
- `trainall.py` — helper to run multiple experiments.
- `dataset.py` — dataset loading and preprocessing.
- `config.yaml`, `config.py` — configuration files.
- `model/` — attention blocks and model definitions.
- `results/` — output checkpoints and logs (gitignored).

## Outputs

During training the script will produce:

- Model checkpoints (`best.pth`, `last.pth`).
- Training and validation logs (loss, accuracy, F1).
- Evaluation artifacts (confusion matrix, classification report).

## Requirements

- Python 3.8+
- PyTorch (compatible version for your CUDA/CPU setup)
- torchvision, scikit-learn, numpy, matplotlib, tqdm, pyyaml

Install dependencies using `pip install -r requirements.txt` or the single-line shown above.

## Contributing

Feel free to open issues and pull requests. Suggested contributions:

- Add new attention modules and model variants.
- Improve training recipes and augmentation strategies.
- Add example notebooks or scripts for evaluation and visualization.

## License

This project is distributed under the terms of the LICENSE file included in the repository.

---

If you want, I can further tailor the README for a specific dataset (CIFAR, CUB, FER2013) or add example commands for common experiments.
