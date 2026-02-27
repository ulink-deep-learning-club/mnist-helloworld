# Modular Neural Network Training Framework

A refactored, modular deep learning framework that supports multiple datasets, model architectures, and advanced training features.

## Features

### Modular Architecture
- **Dataset Abstraction**: Easy addition of new datasets
- **Model Registry**: Pluggable model architectures
- **Training Framework**: Reusable training components
- **Configuration Management**: YAML and command-line configuration
- **Experiment Management**: YOLO-style runs/expX directories with resume and fork support

### Supported Datasets
- MNIST (28x28 grayscale, 10 classes)
- CIFAR-10 (32x32 RGB, 10 classes)
- Subset631 (CASIA-HWDB 2.1 subset with 631 samples)
- Subset1000 (CASIA-HWDB 2.1 subset with 1000 samples)
- TripletMNIST (Triplet learning dataset)
- BalancedTripletMNIST (Balanced triplet dataset)
- TripletSubset1000 (CASIA-HWDB 2.1 triplet subset with 1000 samples)
- Easy to extend with new datasets

### Supported Models

#### Classic Models
- LeNet-5 (classic architecture)
- AlexNet
- MyNet (custom architecture)

#### Vision Transformer (ViT) Models
- BottleneckViT
- FeaturePyramidViT (descendant of BottleneckViT) (Tiny, Small, Large)
- SiameseFPNViT (descendant of BottleneckViT) (Tiny, Small, Large)

#### Mixture of Experts (MoE) Models (descendant of BottleneckViT)
- FeaturePyramidMoEViT (Tiny, Small, Large)
- SiameseFPNMoEViT (Tiny, Small, Large)

#### Siamese Networks
- SiameseNetwork (for metric learning)

### Training Features
- **Layer Freezing**: Freeze layers by ID, range, or name pattern
- **Optimizers**: AdamW, Adam, SGD, Muon, MuonWithAuxAdam
- **Learning Rate Schedulers**: Step, Cosine, Plateau, Exponential
- **Mixed Precision Training**: FP16 support for faster training
- **Early Stopping**: Configurable patience
- **Checkpoint Management**: Auto-save, resume, and fork experiments
- **Siamese/Triplet Loss**: Support for metric learning

## Project Structure

```
mnist-helloworld/
├── src/
│   ├── datasets/                # Dataset implementations
│   │   ├── base.py              # Base dataset class
│   │   ├── mnist.py             # MNIST dataset
│   │   ├── cifar.py             # CIFAR-10 dataset
│   │   ├── subset_631.py        # Subset631 dataset
│   │   ├── subset_1000.py       # Subset1000 dataset
│   │   ├── triplet_mnist.py     # TripletMNIST dataset
│   │   └── registry.py          # Dataset registry
│   ├── models/                  # Model implementations
│   │   ├── base.py              # Base model class
│   │   ├── lenet.py             # LeNet-5
│   │   ├── mynet.py             # Modern adaption of LeNet
│   │   ├── alexnet.py           # AlexNet
│   │   ├── bottleneck_vit.py    # BottleneckViT
│   │   ├── fpn_vit.py           # FPN + ViT models
│   │   ├── fpn_moe_vit.py       # FPN + MoE + ViT models
│   │   ├── siamese.py           # Siamese network
│   │   └── registry.py          # Model registry
│   ├── training/                # Training framework
│   │   ├── trainer.py           # Main trainer
│   │   ├── metrics.py           # Metrics tracking
│   │   ├── checkpoint.py        # Checkpoint management
│   │   └── experiment.py        # Experiment manager
│   ├── config/                  # Configuration management
│   │   └── config.py            # Config parser and loader
│   └── utils/                   # Utilities
│       ├── device.py            # Device detection
│       ├── logger.py            # Logging setup
│       └── qdrant_search.py     # Vector search integration
├── gui-example/                 # GUI application example
│   └── main_gui.py              # Tkinter-based GUI
├── train.py                     # Main training script
├── config.yaml                  # Default configuration
└── requirements.txt             # Dependencies
```

## Usage

### Basic Usage

Train with default configuration (MNIST dataset, MyNet model):

```bash
python train.py
```

### Command Line Options

Train with CIFAR-10 dataset and LeNet model:

```bash
python train.py --dataset cifar10 --model lenet --epochs 30 --batch-size 128
```

Resume training from an experiment:

```bash
python train.py --resume exp1
```

Fork an experiment (start new experiment from existing checkpoint):

```bash
python train.py --fork exp1
```

Freeze specific layers:

```bash
python train.py --freeze "2-1" --freeze "features"
```

Full list of options:

```bash
python train.py --help
```

### Configuration File

Use a YAML configuration file:

```bash
python train.py --config my_config.yaml
```

Example configuration:

```yaml
# Dataset configuration
dataset:
  name: mnist  # Options: mnist, cifar10, subset_631, subset_1000, triplet_mnist, balanced_triplet_mnist, triplet_subset_1000
  root: ./data
  download: true

# Model configuration
model:
  name: mynet  # Options: lenet, mynet, alexnet, bottleneck_vit, fpn_vit, fpn_vit_tiny, fpn_vit_small, fpn_vit_large, siamese, siamese_fpn_vit, siamese_fpn_vit_tiny, siamese_fpn_vit_small, siamese_fpn_vit_large, fpn_moe_vit, fpn_moe_vit_tiny, fpn_moe_vit_small, fpn_moe_vit_large, siamese_fpn_moe_vit, siamese_fpn_moe_vit_tiny, siamese_fpn_moe_vit_small, siamese_fpn_moe_vit_large, alexnet
  num_classes: 10
  embedding_dim: 128  # For siamese models

# Training configuration
training:
  epochs: 20
  batch_size: 64
  num_workers: 4
  shuffle_train: true

# Optimization configuration
optimization:
  learning_rate: 1e-3
  optimizer: adamw  # Options: adamw, adam, sgd, muon, muon_with_aux_adam
  weight_decay: 0.01
  momentum: 0.9  # Only used for SGD
  scheduler: cosine  # Options: none, step, cosine, plateau, exponential
  # Step scheduler options
  scheduler_step_size: 10
  scheduler_gamma: 0.1
  # Cosine scheduler options
  scheduler_t_max: 20
  scheduler_eta_min: 1e-6
  # Plateau scheduler options
  scheduler_patience: 5
  scheduler_factor: 0.1
  # Muon-specific parameters (only used when optimizer is muon or muon_with_aux_adam)
  muon_momentum: 0.95
  muon_ns_steps: 5
  # Parameters for muon_with_aux_adam
  adam_lr: 3e-4
  adam_betas: [0.9, 0.95]

# Checkpointing configuration
checkpointing:
  checkpoint_dir: checkpoints
  save_frequency: 10
```

### Training Options

#### Dataset & Model
| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | Dataset name | `mnist` |
| `--model` | Model architecture | `mynet` |
| `--data-root` | Dataset root directory | `./data` |
| `--num-classes` | Number of classes | Dataset default |

#### Device & Training
| Option | Description | Default |
|--------|-------------|---------|
| `--device` | Device (cuda/cpu/mps) | `cuda` |
| `--epochs` | Number of epochs | `20` |
| `--batch-size` | Batch size | `64` |
| `--num-workers` | Data loading workers | `4` |
| `--reapply-transforms` | Reapply transforms each epoch | `false` |
| `--mixed-precision` | Enable FP16 training | `false` |

#### Optimization
| Option | Description | Default |
|--------|-------------|---------|
| `--learning-rate` | Learning rate | `1e-3` |
| `--optimizer` | Optimizer | `adamw` |
| `--weight-decay` | Weight decay | `0.01` |
| `--momentum` | SGD momentum | `0.9` |

#### Muon Optimizer
| Option | Description | Default |
|--------|-------------|---------|
| `--muon-momentum` | Muon momentum | `0.95` |
| `--muon-ns-steps` | Newton-Schulz steps | `5` |
| `--adam-lr` | Adam LR (MuonWithAuxAdam) | `3e-4` |
| `--adam-betas` | Adam betas (MuonWithAuxAdam) | `[0.9, 0.95]` |

#### Learning Rate Scheduler
| Option | Description | Default |
|--------|-------------|---------|
| `--scheduler` | LR scheduler | `none` |
| `--scheduler-step-size` | StepLR step size | `10` |
| `--scheduler-gamma` | LR decay factor | `0.1` |
| `--scheduler-t-max` | Cosine T_max | `100` |
| `--scheduler-eta-min` | Cosine eta_min | `1e-6` |
| `--scheduler-patience` | Plateau patience | `5` |
| `--scheduler-factor` | Plateau factor | `0.1` |

#### Metric Learning
| Option | Description | Default |
|--------|-------------|---------|
| `--triplet-margin` | Triplet loss margin | `1.0` |
| `--embedding-dim` | Embedding dimension | `256` |

#### Checkpointing
| Option | Description | Default |
|--------|-------------|---------|
| `--checkpoint-dir` | Checkpoint directory | `checkpoints` |
| `--save-frequency` | Save frequency (epochs) | `1` |
| `--resume` | Resume from experiment | - |
| `--fork` | Fork from experiment | - |

#### Layer Control
| Option | Description | Default |
|--------|-------------|---------|
| `--freeze` | Freeze layers (ID, range, or name) | - |

#### Early Stopping
| Option | Description | Default |
|--------|-------------|---------|
| `--patience` | Early stopping patience | `0` |

## Adding New Components

### Adding a New Dataset

1. Create a new file in `src/datasets/`:

```python
from .base import BaseDataset
import torchvision.transforms as transforms

class MyDataset(BaseDataset):
    def get_train_transform(self):
        return transforms.Compose([...])
    
    def get_test_transform(self):
        return transforms.Compose([...])
    
    def load_data(self):
        # Load your dataset
        pass
    
    @property
    def num_classes(self):
        return 10
    
    @property
    def input_channels(self):
        return 3
    
    @property
    def input_size(self):
        return (32, 32)
```

2. Register it in `src/datasets/registry.py`:

```python
from .my_dataset import MyDataset
DatasetRegistry.register('mydataset', MyDataset)
```

### Adding a New Model

1. Create a new file in `src/models/`:

```python
from .base import BaseModel
import torch.nn as nn

class MyModel(BaseModel):
    def __init__(self, num_classes=10, input_channels=1, **kwargs):
        super().__init__(num_classes, input_channels)
        # Define your model architecture
        
    def forward(self, x):
        # Define forward pass
        return x
```

2. Register it in `src/models/registry.py`:

```python
from .my_model import MyModel
ModelRegistry.register('mymodel', MyModel)
```

## GUI Application

A simple Tkinter-based GUI for MNIST classification (supports MyNet and LeNet models):

```bash
cd gui-example
pip install -r requirements.txt
python main_gui.py
```

## Requirements

This is a uv project. Install dependencies with:

```bash
uv sync
```

Core dependencies:
- torch
- torchvision
- tqdm
- pyyaml
- torchinfo
- albumentations
- qdrant-client
- muon-optimizer

## Project Outputs

Experiment outputs are saved in YOLO-style directories:
- `runs/exp1/`, `runs/exp2/`, etc. - Experiment directories
- Each experiment contains: `checkpoints/`, `logs/`, `config.yaml`
- Checkpoints include: `latest_checkpoint.pt`, `best_model.pt`, `final_model.pt`
