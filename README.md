# CrossChannelSegmentationExcitationBlock Classification

This repository implements CrossChannelSegmentationExcitationBlock (CCSE) and various neural network architectures for image classification tasks. The project includes comprehensive experiments on multiple datasets including FER2013 emotion recognition, CIFAR-10, and CIFAR-100.

## Features

- **CrossChannelSegmentationExcitationBlock Implementation**: Novel attention mechanism for enhancing spatial and channel-wise feature interactions through cross-channel segmentation and excitation
- **Multiple Model Architectures**:
  - CCSE-ResNet variants (ResNet18, 34, 50, 101, 152)
  - SE-ResNet variants with Squeeze-and-Excitation blocks
  - ExtraNet series with CCSE enhancement
  - Scalable ExtraNet architectures (pico, nano, micro, tiny, small, medium, large)
- **Advanced Training Strategies**: Mixup, CutMix, Test-Time Augmentation (TTA)
- **Loss Functions**: Symmetric Cross Entropy (SCE), Focal Loss, Generalized Cross Entropy (GCE), and more
- **Data Augmentation**: Comprehensive pipeline with geometric and color transformations
- **Performance Optimization**: Automatic Mixed Precision (AMP) and PyTorch compilation
- **Experiment Tracking**: Automated logging, visualization, and model checkpointing

## Datasets

The project supports multiple datasets:

### FER2013 (Emotion Recognition)
- 35,887 grayscale facial images
- 7 emotion categories: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- Challenging real-world dataset with significant class imbalance

### CIFAR-10 & CIFAR-100
- 60,000 32x32 color images
- CIFAR-10: 10 classes, CIFAR-100: 100 classes
- Standard benchmark datasets for image classification

## Model Architectures

### CrossChannelSegmentationExcitationBlock (CCSE)
Core attention mechanism that enhances both spatial and channel dimensions through cross-channel segmentation and excitation, providing better feature representation learning by dividing channels into even and odd groups and applying cross-excitation between them.

### ResNet Variants with CCSE
- **CCSE-ResNet18/34/50/101/152**: ResNet architectures enhanced with CrossChannelSegmentationExcitationBlock
- Superior performance compared to standard ResNet and SE-ResNet on various tasks

### ExtraNet Series
- **ExtraNet**: Base architecture with efficient residual connections
- **ExtraNet_CCSE**: Enhanced with CrossChannelSegmentationExcitationBlock for better feature extraction
- **ExtraNet_CCSE_Lite**: Lightweight version optimized for mobile deployment
- **ExtraNet_Scalable**: Configurable model sizes from pico (minimal) to large (comprehensive)

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (recommended for GPU acceleration)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/CCSEBlock_classification.git
cd CCSEBlock_classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For development:
```bash
pip install pytest black isort flake8
```

## Usage

### Configuration
All experiment parameters are defined in `config.yaml`:

- **Model Settings**: Architecture selection, scaling parameters, CCSE block configuration
- **Training Parameters**: Batch size, learning rate, epochs, optimizer settings
- **Data Augmentation**: Transformation pipelines and probabilities
- **Loss Functions**: Multiple loss options with customizable parameters
- **Scheduler**: Learning rate scheduling with warmup and cosine annealing

### Training
Run training with default configuration:
```bash
python train.py
```

Run with custom configuration:
```bash
python train.py --config config.yaml
```

### Data Preparation
Organize datasets in the following structure:
```
data/
├── fer2013/
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   ├── val/
│   └── test/
├── cifar10/
│   ├── train/
│   └── test/
└── cifar100/
    ├── train/
    └── test/
```

### Model Benchmarking
Benchmark different model architectures:
```bash
python utiles/model_benchmark.py
```

## Key Components

### CrossChannelSegmentationExcitationBlock Implementation (`model/CCSEBlock.py`)
- Cross-Channel Segmentation and Excitation mechanism
- Efficient attention computation through channel segmentation
- Compatible with any CNN architecture

### Training Features
- **Advanced Loss Functions**: SCE for noisy labels, Focal Loss for class imbalance, GCE, and more
- **Data Augmentation**: Random erasing, Gaussian blur, affine transformations
- **Class Balancing**: Automatic class weight calculation for imbalanced datasets
- **Early Stopping**: Prevents overfitting with configurable patience
- **Mixed Precision Training**: Faster training with reduced memory usage

### Evaluation & Visualization
- Comprehensive metrics: Accuracy, F1-Score, Precision, Recall
- Confusion matrices with class-wise analysis
- Training history plots (loss and accuracy curves)
- Classification reports with detailed statistics

## Project Structure

```
CCSEBlock_classification/
├── config.py                 # Configuration management
├── config.yaml              # Experiment configurations
├── dataset.py               # Dataset loading and preprocessing
├── train.py                # Main training script
├── utils.py                # Utility functions and loss functions
├── requirements.txt         # Python dependencies
├── model/                  # Model architectures
│   ├── CCSEBlock.py        # CrossChannelSegmentationExcitationBlock implementation
│   ├── CCSE_ResNet.py      # CCSE-enhanced ResNet variants
│   ├── SE_ResNet.py        # SE-ResNet baselines
│   ├── ExtraNet*.py        # ExtraNet variants
│   └── __init__.py
├── utiles/                 # Utility scripts
│   ├── model_benchmark.py  # Architecture benchmarking
│   ├── loss.py            # Additional loss functions
│   ├── calculate_class_weights.py
│   └── check_model_type.py
├── data/                   # Datasets (not included in repo)
├── results/               # Experiment outputs (not included in repo)
├── .gitignore             # Git ignore rules
├── LICENSE                # License file
└── README.md
```

## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Use black for code formatting
- Add type hints for new functions
- Include docstrings for all public functions
- Add unit tests for new features
- Update documentation for API changes

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{crosschannelsegmentationexcitation2024,
  title={CrossChannelSegmentationExcitationBlock: Cross-Channel Segmentation and Excitation for Image Classification},
  author={Fu Bin},
  year={2024},
  publisher={GitHub},
  journal={GitHub repository},
  url={https://github.com/Fvn18/CCSEBlock_classification}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FER2013 dataset: [Goodfellow et al., 2013]
- CIFAR datasets: [Krizhevsky et al., 2009]
- PyTorch team for the excellent deep learning framework
