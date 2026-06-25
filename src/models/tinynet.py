import torch
import torch.nn as nn
from typing import Any

try:
    from .base import BaseModel
    from ..training.metrics import MetricsTracker
except ImportError:
    from base import BaseModel
    from training.metrics import MetricsTracker


class TinyNet(BaseModel):
    """Deliberately tiny modern CNN for MNIST.

    Uses modern design patterns (3x3 conv, batch norm, ReLU, global avg pooling)
    but with extremely few channels to keep accuracy around or below ~80%.
    Total parameters: ~438.
    """

    @property
    def model_type(self) -> str:
        return "classification"

    @classmethod
    def get_criterion(cls, **kwargs) -> nn.Module:
        return nn.CrossEntropyLoss()

    @classmethod
    def get_metrics_tracker(cls, **kwargs) -> Any:
        return MetricsTracker()

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 1,
        **kwargs,
    ):
        super().__init__(num_classes, input_channels)
        self.input_size = 28

        self.features = nn.Sequential(
            # 28x28 -> 14x14
            nn.Conv2d(input_channels, 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),

            # 14x14 -> 7x7
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
