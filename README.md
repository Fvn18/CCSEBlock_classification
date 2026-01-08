# CCSEBlock Classification

A PyTorch implementation of Cross-Channel Segmentation Excitation (CCSE) Block for image classification tasks. This project includes various CNN architectures enhanced with attention mechanisms and advanced training techniques.

## Features

- **CCSE Block**: Cross-Channel Segmentation Excitation Block for improved feature representation
- **Multiple Architectures**: Support for ResNet, SE-ResNet, CCSE-ResNet, and ExtraNet variants
- **Advanced Training**: Mixup, Cutmix, and other data augmentation techniques
- **Model Scalability**: Support for scalable model variants
- **Comprehensive Evaluation**: F1 score, confusion matrix, and classification reports

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd CCSEBlock_classification

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision scikit-learn numpy matplotlib tqdm pyyaml
```

## Usage

### Training

```bash
python train.py
```

### Configuration

The training process can be configured using `config.yaml`:

- **Model selection**: Choose from various architectures (extra_net_original, extra_net_ccse, extra_net_scalable, ccse_resnet18, etc.)
- **Data settings**: Configure dataset paths, image size, normalization parameters
- **Training parameters**: Set epochs, batch size, learning rate, and optimizer settings
- **Advanced features**: Enable/disable AMP, mixup, cutmix, and other techniques

### Supported Models

- `extra_net_original`: Original ExtraNet architecture
- `extra_net_ccse`: ExtraNet with CCSE blocks
- `extra_net_scalable`: Scalable ExtraNet variant
- `ccse_resnet18/ccse_resnet34/ccse_resnet50/ccse_resnet101`: ResNet variants with CCSE blocks
- `se_resnet18/se_resnet34/se_resnet50/se_resnet101`: ResNet variants with SE blocks

## Project Structure

```
CCSEBlock_classification/
├── model/                  # Model definitions
│   ├── CCSEBlock.py        # CCSE block implementation
│   ├── CCSE_ResNet.py      # CCSE-ResNet variants
│   ├── SE_ResNet.py        # SE-ResNet variants
│   └── __init__.py         # Model exports
├── utils/                  # Utility functions
├── config.py               # Configuration management
├── dataset.py              # Dataset loading and preprocessing
├── train.py                # Training script
├── utils.py                # Common utilities
└── config.yaml             # Configuration file
```

## Customization

### Adding New Models

1. Create your model in the `model/` directory
2. Add the import to `model/__init__.py`
3. Update the model loading logic in `train.py` if needed

### Configuration

Modify `config.yaml` to customize:
- Model architecture
- Training hyperparameters
- Data augmentation settings
- Hardware settings (device, AMP, etc.)

## Results

The training script outputs:
- Training/validation accuracy and loss curves
- F1 scores and classification reports
- Confusion matrices
- Model checkpoints
- Training summary with performance metrics

## Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision
- scikit-learn
- numpy
- matplotlib
- tqdm
- pyyaml