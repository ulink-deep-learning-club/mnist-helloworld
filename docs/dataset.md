# Dataset API

## Overview

The framework uses a registry pattern for datasets. All datasets inherit from `BaseDataset` and are registered automatically.

## Using Datasets

### Command Line

```bash
python train.py --dataset mnist
python train.py --dataset cifar10
python train.py --dataset subset_631
python train.py --dataset subset_1000
```

### Python API

```python
from src.datasets import DatasetRegistry

# Create dataset
dataset = DatasetRegistry.create(
    "mnist",
    root="./data",
    download=True,
    reapply_transforms=False
)

# Get data loaders
train_loader, val_loader = dataset.get_dataloaders(
    batch_size=64,
    num_workers=4,
    shuffle_train=True
)

# Access dataset properties
print(f"Classes: {dataset.num_classes}")
print(f"Input channels: {dataset.input_channels}")
print(f"Input size: {dataset.input_size}")
```

## Creating Custom Datasets

### Step 1: Inherit from BaseDataset

```python
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from src.datasets.base import BaseDataset

class MyDataset(BaseDataset):
    def __init__(self, root="./data", download=True, reapply_transforms=False):
        super().__init__(root, download, reapply_transforms)
        self.load_data()

    def get_train_transform(self):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def get_test_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def load_data(self):
        # Load your dataset here
        self._train_dataset = ...
        self._test_dataset = ...

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def input_channels(self) -> int:
        return 3

    @property
    def input_size(self) -> tuple:
        return (64, 64)

    def _reload_train_data(self):
        # Implement to reload training data with new transforms
        self._train_dataset = ...
```

### Step 2: Register Your Dataset

```python
from src.datasets import DatasetRegistry
from src.datasets.my_dataset import MyDataset

# Register manually
DatasetRegistry.register("my_dataset", MyDataset)

# Or add to registry.py for auto-registration
# DatasetRegistry.register("my_dataset", MyDataset)
```

### Step 3: Use Your Dataset

```python
dataset = DatasetRegistry.create("my_dataset")
```

## BaseDataset API Reference

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_classes` | int | Number of classes in dataset |
| `input_channels` | int | Number of input channels (1=grayscale, 3=RGB) |
| `input_size` | tuple | Image size as (height, width) |
| `reapply_transforms` | bool | Whether to reapply transforms each epoch |

### Methods

#### `get_train_transform() -> transforms.Compose`

Returns the transformation pipeline for training data (includes augmentation).

#### `get_test_transform() -> transforms.Compose`

Returns the transformation pipeline for test/validation data (no augmentation).

#### `load_data()`

Loads the dataset. Called automatically by `get_dataloaders()` if not already loaded.

#### `get_dataloaders(batch_size, num_workers, shuffle_train) -> (train_loader, val_loader)`

Returns DataLoader tuples for training and validation.

#### `reset_train_transforms()`

Regenerates training transforms for new random augmentations. Used internally when `reapply_transforms=True`.

## DatasetRegistry API Reference

### Methods

#### `DatasetRegistry.register(name: str, dataset_class: Type[BaseDataset])`

Register a new dataset class.

```python
DatasetRegistry.register("my_dataset", MyDataset)
```

#### `DatasetRegistry.get(name: str) -> Type[BaseDataset]`

Get a dataset class by name.

```python
cls = DatasetRegistry.get("mnist")
```

#### `DatasetRegistry.list_available() -> list`

List all available dataset names.

```python
print(DatasetRegistry.list_available())
# ['mnist', 'cifar10', 'subset_631', 'subset_1000']
```

#### `DatasetRegistry.create(name: str, **kwargs) -> BaseDataset`

Create a dataset instance.

```python
dataset = DatasetRegistry.create("mnist", root="./data", download=True)
```
