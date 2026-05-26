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

| Dataset | Classes | Input Channels (default) | Image Size |
|---------|---------|--------------------------|------------|
| mnist | 10 | 1 | 28×28 |
| cifar10 | 10 | 3 | 32×32 |
| subset_631 | 631 | 1 | 64×64 |
| subset_1000 | 1000 | 1 | 64×64 |
| triplet_mnist | 10 | 1 | 28×28 |
| balanced_triplet_mnist | 10 | 1 | 28×28 |
| triplet_subset_1000 | 1000 | 1 | 64×64 |

## Available Models

| Model | Description |
|-------|-------------|
| lenet | Classic LeNet-5 architecture |
| mynet | Simple CNN for MNIST |
| alexnet | AlexNet adapted for smaller inputs |
| bottleneck_vit | Vision Transformer with bottleneck blocks |
| fpn_vit | Feature Pyramid Network with ViT |
| fpn_vit_tiny / fpn_vit_small / fpn_vit_large | FPN-ViT variants (Tiny/Small/Large) |
| siamese | Siamese network for metric learning |
| siamese_fpn_vit | Siamese + FPN-ViT |
| siamese_fpn_vit_tiny / siamese_fpn_vit_small / siamese_fpn_vit_large | Siamese FPN-ViT variants |
| fpn_moe_vit | FPN-ViT with Mixture of Experts |
| fpn_moe_vit_tiny / fpn_moe_vit_small / fpn_moe_vit_large | FPN-MoE-ViT variants |
| siamese_fpn_moe_vit | Siamese + FPN-MoE-ViT |
| siamese_fpn_moe_vit_tiny / siamese_fpn_moe_vit_small / siamese_fpn_moe_vit_large | Siamese FPN-MoE-ViT variants |

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
