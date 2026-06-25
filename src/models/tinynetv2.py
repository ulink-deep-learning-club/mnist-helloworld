import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

try:
    from .base import BaseModel
    from ..training.metrics import MetricsTracker
except ImportError:
    from base import BaseModel
    from training.metrics import MetricsTracker


def _orthogonal_loss(weight: torch.Tensor) -> torch.Tensor:
    """Encourage output channels of a conv kernel to be mutually orthogonal.

    Args:
        weight: Conv2d weight of shape (out_ch, in_ch, H, W).

    Returns:
        Scalar orthogonality loss.
    """
    out_ch = weight.size(0)
    if out_ch <= 1:
        return weight.new_zeros(())

    # (out_ch, in_ch * H * W)
    W = weight.view(out_ch, -1)
    # Normalize each filter to unit length
    W = F.normalize(W, p=2, dim=1)

    # Gram matrix: dot product between every pair of filters
    gram = W @ W.t()  # (out_ch, out_ch)

    # Target is identity matrix (orthogonal => off-diagonal == 0)
    eye = torch.eye(out_ch, device=W.device, dtype=W.dtype)
    loss = F.mse_loss(gram, eye)
    return loss


class TinyNetV2(BaseModel):
    """Tiny modern CNN with an auxiliary orthogonality loss on conv kernels.

    Returns ``(output, orth_loss)`` during training so that the Trainer
    automatically mixes the auxiliary term into the total loss (same pattern
    used by MoE models).
    """

    @property
    def model_type(self) -> str:
        return "classification"

    @property
    def has_aux_loss(self) -> bool:
        return True

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

        self.conv1 = nn.Conv2d(
            input_channels, 4, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(4)

        self.conv2 = nn.Conv2d(
            4, 8, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(8)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)

        if self.training:
            orth_loss = _orthogonal_loss(self.conv1.weight) + _orthogonal_loss(
                self.conv2.weight
            )
            return logits, orth_loss

        return logits
