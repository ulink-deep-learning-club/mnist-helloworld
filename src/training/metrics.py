import torch
from typing import Dict, List
import time


class MetricsTracker:
    """Track training and validation metrics."""

    def __init__(self, **kwargs):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.losses: List[float] = []
        self.correct = 0
        self.total = 0
        self.start_time = time.time()

    def update(self, loss: float, outputs: torch.Tensor, targets: torch.Tensor):
        """Update metrics with batch results."""
        self.losses.append(loss)

        _, predicted = torch.max(outputs.data, 1)
        self.total += targets.size(0)
        self.correct += (predicted == targets).sum().item()

    def get_accuracy(self) -> float:
        """Get current accuracy."""
        if self.total == 0:
            return 0.0
        return 100.0 * self.correct / self.total

    def get_average_loss(self) -> float:
        """Get average loss."""
        if not self.losses:
            return 0.0
        return sum(self.losses) / len(self.losses)

    def get_metrics(self) -> Dict[str, float]:
        """Get all metrics as dictionary."""
        return {
            "loss": self.get_average_loss(),
            "accuracy": self.get_accuracy(),
            "total_samples": self.total,
        }


class TripletMetricsTracker(MetricsTracker):
    """Track triplet training metrics."""

    def __init__(self, margin: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.positive_distances: List[float] = []
        self.negative_distances: List[float] = []
        self.valid_triplets = 0

    def reset(self):
        """Reset all metrics."""
        super().reset()
        self.positive_distances = []
        self.negative_distances = []
        self.valid_triplets = 0

    def update(
        self,
        loss: float,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ):
        """Update metrics with triplet results."""
        self.losses.append(loss)

        # Calculate distances
        pos_dist = torch.nn.functional.pairwise_distance(anchor, positive, p=2)
        neg_dist = torch.nn.functional.pairwise_distance(anchor, negative, p=2)

        # Track distances
        self.positive_distances.extend(pos_dist.detach().cpu().numpy())
        self.negative_distances.extend(neg_dist.detach().cpu().numpy())

        # Count valid triplets (where pos_dist < neg_dist)
        valid = (pos_dist < neg_dist).sum().item()
        self.valid_triplets += valid
        self.total += anchor.size(0)

    def get_accuracy(self) -> float:
        """Get triplet accuracy (fraction of valid triplets)."""
        if self.total == 0:
            return 0.0
        return 100.0 * self.valid_triplets / self.total

    def get_average_positive_distance(self) -> float:
        """Get average positive distance."""
        if not self.positive_distances:
            return 0.0
        return sum(self.positive_distances) / len(self.positive_distances)

    def get_average_negative_distance(self) -> float:
        """Get average negative distance."""
        if not self.negative_distances:
            return 0.0
        return sum(self.negative_distances) / len(self.negative_distances)

    def get_metrics(self) -> Dict[str, float]:
        """Get all triplet metrics as dictionary."""
        return {
            "loss": self.get_average_loss(),
            "accuracy": self.get_accuracy(),
            "pos_dist": self.get_average_positive_distance(),
            "neg_dist": self.get_average_negative_distance(),
            "total_samples": self.total,
        }


class MetricsTrackerFactory:
    """Factory for creating appropriate metrics tracker."""

    @staticmethod
    def create(tracker_type: str = "classification", **kwargs) -> MetricsTracker:
        """Create a metrics tracker based on type."""
        if tracker_type == "classification":
            return MetricsTracker()
        elif tracker_type == "triplet":
            margin = kwargs.get("margin", 1.0)
            return TripletMetricsTracker(margin=margin)
        else:
            raise ValueError(f"Unknown tracker type: {tracker_type}")
