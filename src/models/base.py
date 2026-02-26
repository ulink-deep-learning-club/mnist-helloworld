from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Any


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models."""

    MODEL_TYPES = ["classification", "siamese", "autoencoder", "triplet"]

    def __init__(self, num_classes: int = 10, input_channels: int = 1, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Model paradigm type. Must be one of MODEL_TYPES."""
        pass

    @property
    def has_aux_loss(self) -> bool:
        """Whether the model has auxiliary loss (e.g., MoE balance loss)."""
        return False

    @abstractmethod
    def forward(self, x):
        """Forward pass."""
        pass

    @classmethod
    @abstractmethod
    def get_criterion(cls, **kwargs) -> nn.Module:
        """Return the appropriate loss function for this model type."""
        pass

    @classmethod
    @abstractmethod
    def get_metrics_tracker(cls, **kwargs):
        """Return the appropriate metrics tracker for this model type."""
        pass

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "name": self.__class__.__name__,
            "type": self.model_type,
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }
