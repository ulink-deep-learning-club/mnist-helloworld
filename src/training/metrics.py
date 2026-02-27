import torch
from typing import Dict, List, Optional
import time
import json
import os


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

    def update( # pyright: ignore[reportIncompatibleMethodOverride]
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


class MoEMetricsTracker:
    """Track MoE-specific metrics: balance loss, expert_freq, expert_prob."""

    def __init__(self, num_experts: int = 8, save_path: Optional[str] = None, **kwargs):
        self.num_experts = num_experts
        self.save_path = save_path
        self.epoch_records: List[Dict] = []
        self.current_epoch = 0
        self.full_reset()

    def reset(self):
        """Reset batch-level metrics (call at start of epoch)."""
        self.balance_losses: List[float] = []
        self.expert_freq_records: List[List[float]] = []
        self.expert_prob_records: List[List[float]] = []

    def full_reset(self):
        """Reset all metrics including epoch history (call at start of training)."""
        self.reset()
        self.epoch_records = []
        self.current_epoch = 0

    def update(
        self,
        balance_loss: float,
        expert_freq: Optional[torch.Tensor] = None,
        expert_prob: Optional[torch.Tensor] = None,
    ):
        """Update metrics with batch results."""
        self.balance_losses.append(balance_loss)

        if expert_freq is not None:
            freq = expert_freq.detach().cpu().numpy().tolist()
            self.expert_freq_records.append(freq)

        if expert_prob is not None:
            prob = expert_prob.detach().cpu().numpy().tolist()
            self.expert_prob_records.append(prob)

    def get_average_balance_loss(self) -> float:
        """Get average balance loss."""
        if not self.balance_losses:
            return 0.0
        return sum(self.balance_losses) / len(self.balance_losses)

    def get_average_expert_freq(self) -> List[float]:
        """Get average expert frequency across all batches."""
        if not self.expert_freq_records:
            return [0.0] * self.num_experts
        num_batches = len(self.expert_freq_records)
        avg_freq = [
            sum(batch[i] for batch in self.expert_freq_records) / num_batches
            for i in range(self.num_experts)
        ]
        return avg_freq

    def get_average_expert_prob(self) -> List[float]:
        """Get average expert probability across all batches."""
        if not self.expert_prob_records:
            return [0.0] * self.num_experts
        num_batches = len(self.expert_prob_records)
        avg_prob = [
            sum(batch[i] for batch in self.expert_prob_records) / num_batches
            for i in range(self.num_experts)
        ]
        return avg_prob

    def get_metrics(self) -> Dict[str, float | List[float]]:
        """Get all MoE metrics as dictionary."""
        metrics: Dict[str, float | List[float]] = {
            "balance_loss": self.get_average_balance_loss(),
        }
        metrics["expert_freq_avg"] = self.get_average_expert_freq()
        metrics["expert_prob_avg"] = self.get_average_expert_prob()
        return metrics

    def get_last_expert_stats(self) -> Dict:
        """Get expert stats from last batch."""
        return {
            "expert_freq": self.expert_freq_records[-1]
            if self.expert_freq_records
            else [],
            "expert_prob": self.expert_prob_records[-1]
            if self.expert_prob_records
            else [],
        }

    def save_epoch(self, epoch: int):
        """Save epoch metrics to records."""
        record = {
            "epoch": epoch,
            "balance_loss": self.get_average_balance_loss(),
            "expert_freq_avg": self.get_average_expert_freq(),
            "expert_prob_avg": self.get_average_expert_prob(),
        }
        self.epoch_records.append(record)
        self.current_epoch = epoch

    def save_to_json(self, filepath: Optional[str] = None):
        """Save all epoch records to JSON file."""
        path = filepath or self.save_path
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.epoch_records, f, indent=2)
