# MNIST Training Framework

A modular deep learning training framework supporting multiple datasets and model architectures.

## Quick Start

```bash
# Train with default settings (MNIST + MyNet)
python train.py

# Train with specific dataset and model
python train.py --dataset mnist --model mynet --epochs 10 --batch-size 64

# Train with CIFAR-10
python train.py --dataset cifar10 --model fpn_vit --epochs 50

# Resume training from experiment
python train.py --resume exp1

# Fork training (load checkpoint, save to new experiment)
python train.py --fork exp1
```

## Available Datasets

| Dataset | Classes | Input Channels | Image Size |
|---------|---------|----------------|------------|
| mnist | 10 | 1 | 28x28 |
| cifar10 | 10 | 3 | 32x32 |
| subset_631 | 631 | 3 | 64x64 |
| subset_1000 | 1000 | 3 | 64x64 |

## Available Models

| Model | Description |
|-------|-------------|
| lenet | Classic LeNet-5 architecture |
| mynet | Simple CNN for MNIST |
| bottleneck_vit | Vision Transformer with bottleneck blocks |
| fpn_vit | Feature Pyramid Network with ViT |

## Features

- **Modular Design**: Easy to add new datasets and models via registry pattern
- **Experiment Tracking**: Automatic experiment directory management (YOLO-style)
- **Checkpointing**: Automatic model saving with best/latest/epoch checkpoints
- **Resume Training**: Resume or fork training from any experiment
- **Layer Freezing**: Freeze specific layers for transfer learning
- **Early Stopping**: Prevent overfitting with configurable patience
- **Data Augmentation**: Reapply transforms each epoch for better generalization

## Project Structure

```
src/
├── config/          # Configuration management
├── datasets/         # Dataset implementations
│   ├── base.py      # BaseDataset abstract class
│   ├── registry.py  # DatasetRegistry
│   ├── mnist.py    # MNIST dataset
│   ├── cifar.py    # CIFAR-10 dataset
│   └── ...
├── models/          # Model implementations
│   ├── base.py     # BaseModel abstract class
│   ├── registry.py # ModelRegistry
│   ├── mynet.py    # Simple CNN
│   ├── lenet.py    # LeNet
│   ├── bottleneck_vit.py
│   └── fpn_vit.py
├── training/        # Training utilities
│   ├── trainer.py         # Main Trainer class
│   ├── checkpoint.py      # CheckpointManager
│   ├── experiment.py     # ExperimentManager
│   └── metrics.py        # MetricsTracker
└── utils/           # Utility functions
    ├── device.py   # Device detection
    └── logger.py   # Logging setup

train.py             # Main training script
```
