"""Test training components (metrics, checkpoint, experiment)."""

import os
import tempfile
import json

import pytest
import torch
import torch.nn as nn

from src.training import MetricsTracker, CheckpointManager, ExperimentManager
from src.training.metrics import TripletMetricsTracker, MoEMetricsTracker


class TestMetricsTracker:
    """Standard classification metrics tracking."""

    def test_initial_state(self):
        tracker = MetricsTracker()
        assert tracker.get_average_loss() == 0.0
        assert tracker.get_accuracy() == 0.0

    def test_update_and_metrics(self):
        tracker = MetricsTracker()
        # Simulate 2 correct out of 4
        outputs = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
        targets = torch.tensor([0, 1, 0, 1])
        tracker.update(loss=0.5, outputs=outputs, targets=targets)
        metrics = tracker.get_metrics()
        assert metrics["accuracy"] == 100.0  # 4/4 correct
        assert metrics["loss"] == 0.5

    def test_reset(self):
        tracker = MetricsTracker()
        outputs = torch.tensor([[1.0, 0.0]])
        targets = torch.tensor([0])
        tracker.update(loss=0.1, outputs=outputs, targets=targets)
        tracker.reset()
        assert tracker.get_average_loss() == 0.0
        assert tracker.get_accuracy() == 0.0


class TestTripletMetricsTracker:
    """Triplet metrics tracking."""

    def test_update_and_metrics(self):
        tracker = TripletMetricsTracker(margin=1.0)
        anchor = torch.randn(4, 128)
        positive = anchor + 0.1  # Close to anchor
        negative = anchor + 2.0  # Far from anchor
        tracker.update(loss=0.5, anchor=anchor, positive=positive, negative=negative)
        metrics = tracker.get_metrics()
        assert "pos_dist" in metrics
        assert "neg_dist" in metrics
        assert metrics["total_samples"] == 4


class TestMoEMetricsTracker:
    """MoE-specific metrics tracking."""

    def test_update_and_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "moe_metrics.json")
            tracker = MoEMetricsTracker(num_experts=4, save_path=save_path)
            freq = torch.tensor([0.25, 0.25, 0.25, 0.25])
            prob = torch.tensor([0.25, 0.25, 0.25, 0.25])
            tracker.update(balance_loss=0.01, expert_freq=freq, expert_prob=prob)
            tracker.save_epoch(1)
            tracker.save_to_json()
            assert os.path.exists(save_path)
            with open(save_path) as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]["epoch"] == 1


class TestCheckpointManager:
    """Checkpoint saving and loading."""

    def test_save_and_load_latest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoints_dir=tmpdir, save_frequency=1)
            model = nn.Linear(10, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            cm.save_latest_checkpoint(model, optimizer, epoch=5, loss=0.5, accuracy=85.0)
            latest_path = os.path.join(tmpdir, "latest_checkpoint.pt")
            assert os.path.exists(latest_path)

            # Load
            new_model = nn.Linear(10, 2)
            info = cm.load_checkpoint(latest_path, new_model, strict=True)
            assert info["epoch"] == 5
            assert info["accuracy"] == 85.0
            assert info["fully_restored"]

    def test_best_model_saving(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoints_dir=tmpdir)
            model = nn.Linear(10, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            # First save (should save)
            saved = cm.save_best_model(model, optimizer, epoch=1, loss=0.5, accuracy=80.0)
            assert saved
            assert cm.best_accuracy == 80.0

            # Worse accuracy (should NOT save)
            saved = cm.save_best_model(model, optimizer, epoch=2, loss=0.6, accuracy=75.0)
            assert not saved
            assert cm.best_accuracy == 80.0

            # Better accuracy (should save)
            saved = cm.save_best_model(model, optimizer, epoch=3, loss=0.3, accuracy=90.0)
            assert saved
            assert cm.best_accuracy == 90.0

    def test_non_strict_loading(self):
        """Non-strict loading should handle incompatible layers gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(checkpoints_dir=tmpdir)
            model = nn.Linear(10, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            cm.save_latest_checkpoint(model, optimizer, epoch=1, loss=0.5, accuracy=80.0)

            # Load into a different architecture
            different_model = nn.Linear(10, 5)  # Different output dim
            info = cm.load_checkpoint(
                os.path.join(tmpdir, "latest_checkpoint.pt"),
                different_model,
                strict=False,
            )
            assert not info["fully_restored"]


class TestExperimentManager:
    """Experiment directory management."""

    def test_new_experiment_creates_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            em = ExperimentManager(base_dir=tmpdir)
            assert os.path.exists(em.experiment_dir)
            assert os.path.exists(em.checkpoints_dir)
            assert em.experiment_dir.startswith(tmpdir)

    def test_experiment_properties(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            em = ExperimentManager(base_dir=tmpdir)
            assert em.log_file.endswith("training_log.txt")
            assert em.plot_file.endswith("training_curves.png")
            assert em.config_file.endswith("config.yaml")

    def test_resume_experiment_requires_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                ExperimentManager(base_dir=tmpdir, resume_exp="exp99")
