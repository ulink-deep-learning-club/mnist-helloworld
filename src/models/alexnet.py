import torch.nn as nn
from typing import Any
from .base import BaseModel
from ..training.metrics import MetricsTracker


class AlexNet(BaseModel):
    """AlexNet implementation adapted for smaller inputs like MNIST."""

    @property
    def model_type(self) -> str:
        return "classification"

    @classmethod
    def get_criterion(cls, **kwargs) -> nn.Module:
        """Return CrossEntropyLoss for classification."""
        return nn.CrossEntropyLoss()

    @classmethod
    def get_metrics_tracker(cls, **kwargs) -> Any:
        """Return standard metrics tracker for classification."""
        return MetricsTracker()

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 1,
        input_size: tuple = (28, 28),
        dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__(num_classes, input_channels)

        self.input_size = input_size
        self.dropout_rate = dropout

        # Feature extractor
        self.features = nn.Sequential(
            # Conv1: input_channels -> 64
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv2: 64 -> 192
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv3: 192 -> 384
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv4: 384 -> 256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv5: 256 -> 256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Calculate flattened size based on input size
        # For 28x28: 28/2=14, 14/2=7, 7/2=3 (floor), so 256*3*3 = 2304
        # For 32x32: 32/2=16, 16/2=8, 8/2=4, so 256*4*4 = 4096
        h, w = input_size
        h = h // 2  # After first pool
        h = h // 2  # After second pool
        h = h // 2  # After third pool
        w = w // 2  # After first pool
        w = w // 2  # After second pool
        w = w // 2  # After third pool
        self.flattened_size = 256 * h * w

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.flattened_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
