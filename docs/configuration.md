# Configuration Guide

## Command Line Arguments

### Dataset Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | mnist | Dataset to use (mnist, cifar10, subset_631, subset_1000) |
| `--data-root` | str | ./data | Root directory for datasets |

### Model Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | mynet | Model architecture (lenet, mynet, bottleneck_vit, fpn_vit) |
| `--num-classes` | int | 10 | Number of output classes |

### Training Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--epochs` | int | 20 | Number of training epochs |
| `--batch-size` | int | 64 | Batch size |
| `--num-workers` | int | 4 | Data loading workers |

### Optimization Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--learning-rate` | float | 1e-3 | Learning rate |
| `--optimizer` | str | adamw | Optimizer (adamw, adam, sgd) |
| `--scheduler` | str | none | Scheduler (none, step, cosine, plateau, exponential) |
| `--scheduler-step-size` | int | 10 | Step size for StepLR |
| `--scheduler-gamma` | float | 0.1 | Decay factor for LR |
| `--scheduler-t-max` | int | 100 | T_max for CosineAnnealing |
| `--scheduler-eta-min` | float | 1e-6 | Min LR for CosineAnnealing |
| `--scheduler-patience` | int | 5 | Patience for ReduceLROnPlateau |
| `--scheduler-factor` | float | 0.1 | Factor for ReduceLROnPlateau |

### Checkpoint Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint-dir` | str | checkpoints | Checkpoint directory |
| `--save-frequency` | int | 1 | Save checkpoint every N epochs |

### Other Options

| Argument | Type | Description |
|----------|------|-------------|
| `--config` | str | Path to YAML configuration file |
| `--resume` | str | Resume from experiment (e.g., exp1) |
| `--fork` | str | Fork from experiment |
| `--freeze` | str[] | Freeze layers (layer IDs, ranges, or name patterns) |
| `--reapply-transforms` | flag | Reapply transforms each epoch |
| `--patience` | int | Early stopping patience (0=disabled) |

## YAML Configuration

Create a config file for complex configurations:

```yaml
# config.yaml
dataset:
  name: subset_631
  root: ./data
  download: true

model:
  name: fpn_vit
  num_classes: 631
  input_channels: 3
  input_size: [64, 64]
  embed_dim: 192
  patch_size: 16
  depth: 6
  num_heads: 12
  mlp_ratio: 4.0
  drop_rate: 0.2
  linear_attention: true
  linear_layer_limit: 4

training:
  epochs: 100
  batch_size: 32
  num_workers: 4
  shuffle_train: true

optimization:
  learning_rate: 1e-4
  optimizer: adamw
  weight_decay: 0.01
  momentum: 0.9

checkpointing:
  checkpoint_dir: checkpoints
  save_frequency: 10
```

### Load Configuration

```bash
python train.py --config config.yaml
```

## Config Class API

### Creating Config

```python
from src.config import Config, load_config

# From YAML file
config = Config.from_yaml("config.yaml")

# From arguments
from src.config import create_config_parser
parser = create_config_parser()
args = parser.parse_args()
config = Config.from_args(args)

# Default config
config = load_config()
```

### Accessing Config

```python
# Via properties
config.dataset      # {'name': 'mnist', 'root': './data', ...}
config.model        # {'name': 'mynet', 'num_classes': 10, ...}
config.training     # {'epochs': 20, 'batch_size': 64, ...}
config.optimization # {'learning_rate': 0.001, 'optimizer': 'adamw', ...}
config.checkpointing # {'checkpoint_dir': 'checkpoints', 'save_frequency': 1}

# Via get()
config.get("dataset.name")  # Not supported, use config.dataset["name"]

# To dict
config_dict = config.to_dict()
```

## Layer Freezing Examples

### Freeze by Layer ID

Get layer IDs from torchinfo:

```bash
# Layer IDs shown in torchinfo output
# Example: 2-1, 2-2, 3-1, etc.
python train.py --freeze 2-1 2-2 2-3
```

### Freeze by Range

```bash
# Freeze layers 2-1 through 2-5
python train.py --freeze 2-1:2-5
```

### Freeze by Name Pattern

```bash
# Freeze all layers starting with "features"
python train.py --freeze features

# Freeze multiple patterns
python train.py --freeze features encoder
```

## Examples

### Basic MNIST Training

```bash
python train.py --dataset mnist --model mynet --epochs 10
```

### CIFAR-10 with FPN-ViT

```bash
python train.py \
  --dataset cifar10 \
  --model fpn_vit \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 1e-4
```

### Subset-631 with Custom Config

```yaml
# fpn_631.yaml
dataset:
  name: subset_631
  root: ./data

model:
  name: fpn_vit
  num_classes: 631

training:
  epochs: 100
  batch_size: 16

optimization:
  learning_rate: 5e-5
  optimizer: adamw
  weight_decay: 0.05
```

```bash
python train.py --config fpn_631.yaml
```

### Transfer Learning

```bash
# Freeze encoder, train only classifier
python train.py \
  --dataset subset_631 \
  --model fpn_vit \
  --freeze encoder \
  --epochs 20
```

### Resume with Early Stopping

```bash
python train.py --resume exp1 --patience 10
```

### Fork for Hyperparameter Search

```bash
# Fork exp1 with different learning rate
python train.py --fork exp1 --learning-rate 1e-5
```

## Learning Rate Schedulers

### Available Schedulers

| Scheduler | Description |
|-----------|-------------|
| `none` | No scheduler (constant LR) |
| `step` | Step decay: LR * gamma every step_size epochs |
| `cosine` | Cosine annealing from initial LR to eta_min |
| `plateau` | Reduce LR when metric stops improving |
| `exponential` | Exponential decay: LR * gamma each epoch |

### Examples

```bash
# Step LR: decay by 0.1 every 10 epochs
python train.py --scheduler step --scheduler-step-size 10 --scheduler-gamma 0.1

# Cosine annealing
python train.py --scheduler cosine --scheduler-t-max 100 --scheduler-eta-min 1e-6

# Reduce on plateau: reduce by 0.5 when accuracy stops improving for 5 epochs
python train.py --scheduler plateau --scheduler-patience 5 --scheduler-factor 0.5

# Exponential decay: reduce LR by 5% each epoch
python train.py --scheduler exponential --scheduler-gamma 0.95
```

### YAML Configuration

```yaml
optimization:
  learning_rate: 1e-3
  optimizer: adamw
  scheduler: cosine
  scheduler_t_max: 100
  scheduler_eta_min: 1e-6
```
