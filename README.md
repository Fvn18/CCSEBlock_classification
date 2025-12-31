# CCSEBlock Classification

This repository contains a comprehensive PyTorch training framework and model implementations for image classification experiments using CCSE (Combined Channel and Spatial Excitation) and related attention blocks. The project provides a modular, extensible architecture for training and evaluating deep learning models with advanced attention mechanisms.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Models](#models)
- [Training Process](#training-process)
- [Results and Evaluation](#results-and-evaluation)
- [Advanced Features](#advanced-features)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

CCSEBlock Classification is a research-oriented deep learning framework designed for experimenting with attention mechanisms in image classification tasks. The project implements various model architectures including ExtraNet variants with CCSE blocks, ResNet models with SE/CCSE modifications, and provides comprehensive training utilities with advanced features like MixUp, CutMix, label smoothing, and multiple loss functions.

## Features

- **Modular Architecture**: Clean, extensible codebase with clear separation of concerns
- **Multiple Model Architectures**: Support for ExtraNet, ResNet, and custom attention blocks
- **Advanced Training Techniques**: MixUp, CutMix, label smoothing, focal loss, and more
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and classification reports
- **Flexible Configuration**: YAML-based configuration with extensive customization options
- **Mixed Precision Training**: Automatic mixed precision support for faster training
- **Model Checkpointing**: Resume training from saved checkpoints
- **Early Stopping**: Automatic early stopping to prevent overfitting
- **Rich Logging**: Detailed training logs and metrics visualization

## Project Structure

```
CCSEBlock_classification/
├── config.py                 # Configuration loading and defaults
├── config.yaml              # Default configuration file
├── dataset.py               # Dataset loading utilities and preprocessing
├── train.py                 # Main training script with comprehensive logging
├── model/                   # Model definitions and architectures
│   ├── __init__.py
│   └── [model files]
├── utils/                   # Utility functions (losses, data augmentation, etc.)
│   ├── __init__.py
│   └── [utility files]
├── results/                 # Training results, checkpoints, and logs (git-ignored)
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore patterns for ML projects
└── README.md               # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CCSEBlock_classification.git
cd CCSEBlock_classification
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python train.py --config config.yaml
```

### Custom Configuration

```bash
# Override specific parameters via command line
python train.py --model ccse_resnet50 --epochs 100 --batch_size 64 --lr 0.001

# Use custom config file
python train.py --config my_config.yaml
```

### Resume Training

```bash
python train.py --resume /path/to/checkpoint.pth
```

## Configuration

The training process is controlled through `config.yaml` which supports the following key parameters:

### Model Configuration
- `model`: Model architecture (e.g., `resnet18`, `ccse_resnet50`, `extranet_ccse`)
- `num_classes`: Number of output classes
- `input_channels`: Input channel count (default: 3)

### Training Parameters
- `epochs`: Number of training epochs
- `batch_size`: Batch size for training
- `lr`: Initial learning rate
- `optimizer`: Optimizer type (`adam`, `adamw`, `sgd`)
- `weight_decay`: Weight decay for regularization

### Data Augmentation
- `mixup_alpha`: Alpha parameter for MixUp augmentation
- `mixup_prob`: Probability of applying MixUp
- `cutmix_alpha`: Alpha parameter for CutMix augmentation
- `cutmix_prob`: Probability of applying CutMix

### Advanced Features
- `label_smoothing`: Label smoothing factor
- `use_amp`: Enable automatic mixed precision
- `gradient_clip`: Gradient clipping threshold
- `patience`: Early stopping patience

## Models

The project includes several model architectures:

### ExtraNet Variants
- `extranet_ccse`: ExtraNet with CCSE blocks
- `extranet_ccse_lite`: Lightweight version of ExtraNet with CCSE
- `extranet`: Standard ExtraNet architecture
- `extranet_scalable`: Scalable ExtraNet with different scales

### ResNet Variants
- Standard ResNet models: `resnet18`, `resnet34`, `resnet50`, etc.
- SE-ResNet: `se_resnet18`, `se_resnet34`, etc. (with Squeeze-and-Excitation)
- CCSE-ResNet: `ccse_resnet18`, `ccse_resnet34`, etc. (with CCSE blocks)

### Custom Blocks
- CCSE (Combined Channel and Spatial Excitation) blocks
- Standard SE (Squeeze-and-Excitation) blocks
- Configurable attention mechanisms

## Training Process

The training process includes several advanced features:

### Data Augmentation
- **MixUp**: Combines two training samples and their labels
- **CutMix**: Combines images by cutting and pasting patches
- **Standard augmentations**: Random cropping, flipping, normalization

### Loss Functions
- Cross-Entropy Loss (with label smoothing)
- Focal Loss for handling class imbalance
- Symmetric Cross-Entropy Loss
- Generalized Cross-Entropy Loss
- KL Divergence with Label Smoothing

### Learning Rate Scheduling
- Cosine Annealing
- Cosine Annealing with Warm Restarts
- Step Decay
- Reduce on Plateau

### Mixed Precision Training
- Automatic mixed precision (AMP) for faster training
- Gradient scaling to prevent underflow

## Results and Evaluation

The training process generates comprehensive results including:

### Metrics
- Training and validation loss
- Training and validation accuracy
- Learning rate tracking
- Best model checkpoint

### Visualizations
- Training history plots (loss and accuracy curves)
- Confusion matrix heatmaps
- Classification reports

### Detailed Statistics
- Best validation accuracy and epoch
- Final training and validation accuracy
- Average and maximum accuracy across epochs
- Total training time
- Improvement from start to finish

### Output Files
- Model checkpoints saved in `results/` directory
- Training logs with detailed metrics
- Configuration files for reproducibility
- Performance plots and visualizations

## Advanced Features

### Class Weights
Support for weighted loss functions to handle imbalanced datasets.

### Warmup Scheduling
Learning rate warmup for stable training initialization.

### Dynamic Label Smoothing
Adjustable label smoothing that changes during training.

### Multiple Evaluation Metrics
- Accuracy
- F1 Score (weighted average)
- Per-class precision and recall
- Detailed classification reports

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Original ResNet authors for the foundational architecture
- Squeeze-and-Excitation (SE) block authors
- MixUp and CutMix authors for advanced data augmentation techniques
- FER2013, CIFAR, and other dataset creators
- The deep learning research community for continuous innovation