# Model API

## Overview

The framework uses a registry pattern for models. All models inherit from `BaseModel` and are registered automatically.

## Using Models

### Command Line

```bash
python train.py --model lenet
python train.py --model mynet
python train.py --model bottleneck_vit
python train.py --model fpn_vit
```

### Python API

```python
from src.models import ModelRegistry

# Create model
model = ModelRegistry.create(
    "fpn_vit",
    num_classes=10,
    input_channels=3,
    input_size=(64, 64)
)

# Get model info
info = model.get_model_info()
print(f"Parameters: {info['total_parameters']}")
print(f"Trainable: {info['trainable_parameters']}")
```

## Creating Custom Models

### Step 1: Inherit from BaseModel

```python
import torch.nn as nn
from src.models.base import BaseModel

class MyModel(BaseModel):
    def __init__(self, num_classes=10, input_channels=1, **kwargs):
        super().__init__(num_classes, input_channels)
        
        # Define your architecture
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

### Step 2: Register Your Model

```python
from src.models import ModelRegistry
from src.models.my_model import MyModel

# Register manually
ModelRegistry.register("mymodel", MyModel)

# Or add to registry.py for auto-registration
```

### Step 3: Use Your Model

```python
model = ModelRegistry.create("mymodel", num_classes=10, input_channels=1)
```

## Built-in Models

### LeNet

Classic LeNet-5 architecture for MNIST.

```python
model = ModelRegistry.create("lenet", num_classes=10, input_channels=1)
```

### MyNet

Simple CNN optimized for MNIST digit classification.

```python
model = ModelRegistry.create("mynet", num_classes=10, input_channels=1)
```

### BottleneckViT

Vision Transformer with bottleneck blocks.

```python
model = ModelRegistry.create(
    "bottleneck_vit",
    num_classes=10,
    input_channels=3,
    input_size=(32, 32),
    embed_dim=192,
    depth=6,
    num_heads=8,
    mlp_ratio=4.0,
    drop_rate=0.2
)
```

### FeaturePyramidViT (FPN-ViT)

Feature Pyramid Network with Vision Transformer and linear attention.

```python
model = ModelRegistry.create(
    "fpn_vit",
    num_classes=631,
    input_channels=3,
    input_size=(64, 64),
    embed_dim=192,
    patch_size=16,
    depth=6,
    num_heads=12,
    linear_attention=True,
    linear_layer_limit=4
)
```

#### FPN-ViT Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | 192 | Embedding dimension |
| `patch_size` | 16 | Patch size for ViT |
| `input_channels` | 3 | Input image channels |
| `num_classes` | 631 | Number of output classes |
| `depth` | 6 | Number of transformer blocks |
| `num_heads` | 12 | Number of attention heads |
| `mlp_ratio` | 4.0 | MLP hidden dim multiplier |
| `drop_rate` | 0.2 | Dropout rate |
| `linear_attention` | True | Use linear attention |
| `linear_layer_limit` | 4 | Number of layers using linear attention |

#### Attention Mechanisms

- **Standard Multi-Head Attention**: Full quadratic complexity O(N²d)
- **FocusedLinearAttention (ICCV 2023)**: Uses ELU kernel with focusing factor for sharper attention distribution
- **LinearAttention**: Efficient linear attention with O(Nd²) complexity

## BaseModel API Reference

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_classes` | int | Number of output classes |
| `input_channels` | int | Number of input channels |

### Methods

#### `forward(x: Tensor) -> Tensor`

Forward pass. Must be implemented by subclasses.

#### `get_model_info() -> dict`

Returns model information including parameter counts.

```python
{
    'name': 'MyModel',
    'num_classes': 10,
    'input_channels': 1,
    'total_parameters': 1000000,
    'trainable_parameters': 1000000
}
```

## ModelRegistry API Reference

### Methods

#### `ModelRegistry.register(name: str, model_class: Type[BaseModel])`

Register a new model class.

```python
ModelRegistry.register("mymodel", MyModel)
```

#### `ModelRegistry.get(name: str) -> Type[BaseModel]`

Get a model class by name.

```python
cls = ModelRegistry.get("fpn_vit")
```

#### `ModelRegistry.list_available() -> list`

List all available model names.

```python
print(ModelRegistry.list_available())
# ['lenet', 'mynet', 'bottleneck_vit', 'fpn_vit']
```

#### `ModelRegistry.create(name: str, **kwargs) -> BaseModel`

Create a model instance.

```python
model = ModelRegistry.create("fpn_vit", num_classes=10, input_channels=3)
```

## Layer Freezing

Freeze model layers for transfer learning:

```bash
# Freeze by layer ID (from torchinfo)
python train.py --freeze 2-1 2-2

# Freeze by ID range
python train.py --freeze 2-1:2-5

# Freeze by name pattern
python train.py --freeze features encoder.patch_embed
```

### Get Layer IDs

```python
from torchinfo import summary
from train import get_layer_id_mapping

model = ModelRegistry.create("fpn_vit", num_classes=10)
summary(model, input_size=(1, 3, 64, 64), depth=3)

id_to_name = get_layer_id_mapping(model)
for layer_id, name in id_to_name.items():
    print(f"{layer_id}: {name}")
```
