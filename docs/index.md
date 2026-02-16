# Documentation Index

Welcome to the MNIST Training Framework documentation.

## Getting Started

- [README](README.md) - Overview and quick start
- [Configuration Guide](configuration.md) - Command line and YAML config

## API Reference

- [Dataset API](dataset.md) - Using and creating datasets
- [Model API](model.md) - Using and creating models
- [Training API](training.md) - Trainer, checkpoints, experiments

## Quick Links

| Topic | Description |
|-------|-------------|
| [README](README.md) | Quick start, features, project structure |
| [Dataset API](dataset.md) | Dataset registry, custom datasets |
| [Model API](model.md) | Model registry, custom models, FPN-ViT |
| [Training API](training.md) | Training loop, checkpointing, experiments |
| [Configuration](configuration.md) | CLI args, YAML config, examples |

## Common Tasks

### Train a Model
```bash
python train.py --dataset mnist --model mynet --epochs 10
```

### Use Custom Dataset
See [Dataset API](dataset.md#creating-custom-datasets)

### Use Custom Model
See [Model API](model.md#creating-custom-models)

### Resume Training
```bash
python train.py --resume exp1
```

### Transfer Learning
```bash
python train.py --dataset subset_631 --model fpn_vit --freeze encoder
```
