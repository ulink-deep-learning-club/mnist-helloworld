# Training API

## Overview

The training framework provides modular components for training neural networks with experiment tracking, checkpointing, and early stopping.

## Quick Start

```python
from src.datasets import DatasetRegistry
from src.models import ModelRegistry
from src.training import Trainer, CheckpointManager, ExperimentManager
from src.utils import get_device
import torch.nn as nn
import torch.optim as optim

# Setup
device = get_device()

# Create dataset
dataset = DatasetRegistry.create("mnist")
train_loader, val_loader = dataset.get_dataloaders(batch_size=64)

# Create model
model = ModelRegistry.create("mynet", num_classes=10).to(device)

# Create training components
experiment_manager = ExperimentManager(base_dir="runs")
checkpoint_manager = CheckpointManager(
    checkpoints_dir=experiment_manager.checkpoints_dir,
    save_frequency=1
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    experiment_manager=experiment_manager,
    checkpoint_manager=checkpoint_manager,
    patience=5  # Early stopping
)

# Train
results = trainer.train(epochs=20)

# Results
print(f"Best accuracy: {results['best_accuracy']:.2f}%")
print(f"Training time: {results['training_time']:.2f}s")
```

## Trainer

### Constructor

```python
Trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    experiment_manager: ExperimentManager,
    checkpoint_manager: Optional[CheckpointManager] = None,
    scheduler: Optional[Any] = None,
    dataset: Optional[BaseDataset] = None,
    patience: int = 0
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | nn.Module | Neural network model |
| `train_loader` | DataLoader | Training data loader |
| `val_loader` | DataLoader | Validation data loader |
| `criterion` | nn.Module | Loss function |
| `optimizer` | optim.Optimizer | Optimizer |
| `device` | torch.device | Device to train on |
| `experiment_manager` | ExperimentManager | Experiment tracking |
| `checkpoint_manager` | Optional[CheckpointManager] | Checkpoint saving |
| `scheduler` | Optional[Any] | Learning rate scheduler |
| `dataset` | Optional[BaseDataset] | Dataset (for transform reapplication) |
| `patience` | int | Early stopping patience (0=disabled) |

### Methods

#### `train(epochs: int, start_epoch: int = 0) -> Dict`

Train the model for specified epochs.

```python
results = trainer.train(epochs=20, start_epoch=0)
```

Returns:
```python
{
    'epochs_trained': int,           # Number of epochs trained
    'best_accuracy': float,         # Best validation accuracy
    'training_time': float,         # Total training time in seconds
    'history': dict,                # Training history
    'experiment_dir': str,          # Experiment directory path
    'stopped_early': bool,          # Whether early stopping triggered
    'best_epoch': int               # Epoch with best accuracy
}
```

#### `train_epoch(epoch: int) -> (Dict, float)`

Train for one epoch. Returns metrics dict and speed (it/s).

#### `validate() -> (Dict, float)`

Validate the model. Returns metrics dict and speed (it/s).

#### `load_history_from_log() -> int`

Load training history from log file. Returns last epoch number.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `history` | dict | Training history (train_loss, val_loss, etc.) |
| `best_accuracy` | float | Best validation accuracy |
| `best_epoch` | int | Epoch with best accuracy |
| `epochs_without_improvement` | int | Epochs without improvement |

## CheckpointManager

### Constructor

```python
CheckpointManager(
    checkpoints_dir: str,
    save_frequency: int = 10
)
```

### Methods

#### `save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath, additional_info=None)`

Save a checkpoint.

```python
checkpoint_manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=10,
    loss=0.5,
    accuracy=85.0,
    filepath="checkpoints/epoch_10.pt"
)
```

#### `save_best_model(model, optimizer, epoch, loss, accuracy, additional_info=None) -> bool`

Save model if accuracy improved. Returns True if saved.

```python
is_best = checkpoint_manager.save_best_model(
    model, optimizer, epoch, loss, accuracy
)
```

#### `save_latest_checkpoint(model, optimizer, epoch, loss, accuracy, additional_info=None)`

Save latest checkpoint (overwrites).

#### `save_epoch_checkpoint(model, optimizer, epoch, loss, accuracy, additional_info=None)`

Save epoch-specific checkpoint.

#### `load_checkpoint(filepath, model, optimizer=None, strict=True) -> Dict`

Load checkpoint.

```python
info = checkpoint_manager.load_checkpoint(
    "checkpoints/latest_checkpoint.pt",
    model,
    optimizer,
    strict=False  # Partial loading
)
```

Returns:
```python
{
    'epoch': int,
    'loss': float,
    'accuracy': float,
    'fully_restored': bool,
    'loaded_layers': int,
    'total_layers': int
}
```

### Checkpoint Contents

Checkpoints contain:
- `epoch`: Current epoch number
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `loss`: Validation loss
- `accuracy`: Validation accuracy
- `model_config`: Model configuration

## ExperimentManager

### Constructor

```python
ExperimentManager(
    base_dir: str = "runs",
    resume_exp: Optional[str] = None,
    fork_exp: Optional[str] = None
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_dir` | str | Base directory for experiments |
| `resume_exp` | Optional[str] | Resume from experiment name |
| `fork_exp` | Optional[str] | Fork from experiment name |

### Usage

```python
# New experiment (auto-increments exp1, exp2, ...)
em = ExperimentManager(base_dir="runs")

# Resume existing experiment
em = ExperimentManager(base_dir="runs", resume_exp="exp5")

# Fork experiment (load checkpoint, save to new experiment)
em = ExperimentManager(base_dir="runs", fork_exp="exp5")
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `experiment_dir` | str | Current experiment directory |
| `checkpoints_dir` | str | Checkpoints subdirectory |
| `log_file` | str | Training log file path |
| `plot_file` | str | Training curves plot path |
| `config_file` | str | Configuration file path |

## MetricsTracker

Tracks training metrics (loss, accuracy).

```python
from src.training.metrics import MetricsTracker

tracker = MetricsTracker()

# Update with batch results
tracker.update(loss_value, outputs, labels)

# Get metrics
metrics = tracker.get_metrics()
# {'loss': 0.5, 'accuracy': 85.0, 'total_samples': 1000}

# Reset for next epoch
tracker.reset()
```

## Resume and Fork Training

### Resume Training

Continue training from where it left off:

```bash
python train.py --resume exp1
```

### Fork Training

Load checkpoint but save to new experiment:

```bash
python train.py --fork exp1
```

This is useful for:
- Running with different hyperparameters
- Testing different configurations
- Continuing from a pretrained model

## Early Stopping

Enable early stopping to prevent overfitting:

```bash
python train.py --patience 10
```

Training stops if validation accuracy doesn't improve for 10 epochs.

## Data Augmentation

Reapply random transforms each epoch for better generalization:

```bash
python train.py --reapply-transforms
```

This applies new random augmentations to training data after each epoch.
