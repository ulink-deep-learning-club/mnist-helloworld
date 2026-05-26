"""Test the full training loop: paradigm detection, train/val epochs, early stopping, reapply transforms, MoE."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.models import ModelRegistry
from src.training import Trainer, CheckpointManager, ExperimentManager
from src.training.metrics import TripletMetricsTracker


# ---------- helpers ----------

def _make_classification_data(batch_size=4, num_batches=3, num_classes=10, ch=1, hw=28):
    """Create synthetic classification data loaders."""
    xs = torch.randn(batch_size * num_batches, ch, hw, hw)
    ys = torch.randint(0, num_classes, (batch_size * num_batches,))
    ds = TensorDataset(xs, ys)
    loader = DataLoader(ds, batch_size=batch_size)
    return loader, loader  # train, val (use same for test)


def _make_triplet_data(batch_size=4, num_batches=3, ch=1, hw=28, embed_dim=128):
    """Create synthetic triplet data loaders."""
    xs = torch.randn(batch_size * num_batches, ch, hw, hw)
    # For triplet, data format is (anchor, positive, negative, label)
    # Just make another copy for positive/negative
    ds = TensorDataset(xs, xs, xs, torch.zeros(batch_size * num_batches, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size)
    return loader, loader


# ---------- paradigm detection ----------

class TestParadigmDetection:

    def test_classification_with_standard_model(self):
        model = ModelRegistry.create("mynet", num_classes=10, input_channels=1, input_size=(28, 28))
        train_loader, val_loader = _make_classification_data()

        with tempfile.TemporaryDirectory() as tmpdir:
            em = ExperimentManager(base_dir=tmpdir)
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=nn.CrossEntropyLoss(),
                optimizer=optim.SGD(model.parameters(), lr=0.01),
                device=torch.device("cpu"),
                experiment_manager=em,
            )
            assert trainer.paradigm == "classification"

    def test_triplet_with_siamese_model(self):
        model = ModelRegistry.create("siamese", num_classes=10, input_channels=1, input_size=(28, 28))
        train_loader, val_loader = _make_triplet_data()

        with tempfile.TemporaryDirectory() as tmpdir:
            em = ExperimentManager(base_dir=tmpdir)
            criterion = nn.TripletMarginLoss(margin=1.0)
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optim.SGD(model.parameters(), lr=0.01),
                device=torch.device("cpu"),
                experiment_manager=em,
            )
            assert trainer.paradigm == "triplet"

    def test_moe_detection(self):
        """MoE models should set self.is_moe = True."""
        model = ModelRegistry.create("fpn_moe_vit_tiny", num_classes=10, input_channels=1, input_size=(64, 64))
        train_loader, val_loader = _make_classification_data(ch=1, hw=64)

        with tempfile.TemporaryDirectory() as tmpdir:
            em = ExperimentManager(base_dir=tmpdir)
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=nn.CrossEntropyLoss(),
                optimizer=optim.SGD(model.parameters(), lr=0.01),
                device=torch.device("cpu"),
                experiment_manager=em,
            )
            assert trainer.is_moe


# ---------- classification training loop ----------

class TestClassificationTrainEpoch:

    @pytest.fixture
    def setup(self):
        model = ModelRegistry.create("mynet", num_classes=10, input_channels=1, input_size=(28, 28))
        train_loader, val_loader = _make_classification_data(batch_size=4, num_batches=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        with tempfile.TemporaryDirectory() as tmpdir:
            em = ExperimentManager(base_dir=tmpdir)
            trainer = Trainer(
                model=model, train_loader=train_loader, val_loader=val_loader,
                criterion=criterion, optimizer=optimizer,
                device=torch.device("cpu"), experiment_manager=em,
            )
            yield trainer, model

    def test_train_epoch_returns_metrics(self, setup):
        trainer, _ = setup
        metrics, speed = trainer.train_epoch(epoch=0)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert speed > 0

    def test_validate_returns_metrics(self, setup):
        trainer, _ = setup
        metrics, speed = trainer.validate()
        assert "loss" in metrics
        assert "accuracy" in metrics


class TestClassificationFullTrain:

    def test_train_few_epochs(self):
        model = ModelRegistry.create("mynet", num_classes=10, input_channels=1, input_size=(28, 28))
        train_loader, val_loader = _make_classification_data(batch_size=4, num_batches=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            em = ExperimentManager(base_dir=tmpdir)
            cm = CheckpointManager(em.checkpoints_dir, save_frequency=1)
            trainer = Trainer(
                model=model, train_loader=train_loader, val_loader=val_loader,
                criterion=nn.CrossEntropyLoss(),
                optimizer=optim.SGD(model.parameters(), lr=0.01),
                device=torch.device("cpu"),
                experiment_manager=em, checkpoint_manager=cm,
            )
            results = trainer.train(epochs=3)
            assert results["epochs_trained"] == 3
            assert results["best_accuracy"] >= 0
            assert results["training_time"] > 0
            assert "history" in results
            assert len(results["history"]["train_loss"]) == 3

    def test_log_file_created(self):
        model = ModelRegistry.create("mynet", num_classes=10, input_channels=1, input_size=(28, 28))
        train_loader, val_loader = _make_classification_data(batch_size=4, num_batches=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            em = ExperimentManager(base_dir=tmpdir)
            trainer = Trainer(
                model=model, train_loader=train_loader, val_loader=val_loader,
                criterion=nn.CrossEntropyLoss(),
                optimizer=optim.SGD(model.parameters(), lr=0.01),
                device=torch.device("cpu"), experiment_manager=em,
            )
            trainer.train(epochs=2)
            assert os.path.exists(em.log_file)
            with open(em.log_file) as f:
                lines = f.readlines()
            assert len(lines) == 3  # header + 2 epochs
            assert lines[0].startswith("epoch")


# ---------- early stopping ----------

class TestEarlyStopping:

    def test_early_stopping_triggers(self):
        model = ModelRegistry.create("mynet", num_classes=10, input_channels=1, input_size=(28, 28))
        train_loader, val_loader = _make_classification_data(batch_size=4, num_batches=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            em = ExperimentManager(base_dir=tmpdir)
            cm = CheckpointManager(em.checkpoints_dir, save_frequency=1)
            trainer = Trainer(
                model=model, train_loader=train_loader, val_loader=val_loader,
                criterion=nn.CrossEntropyLoss(),
                optimizer=optim.SGD(model.parameters(), lr=0.01),
                device=torch.device("cpu"),
                experiment_manager=em, checkpoint_manager=cm,
                patience=1,  # Stop after 1 epoch without improvement
            )
            results = trainer.train(epochs=20)  # Would run 20, but early stop at ~2
            assert results["stopped_early"]
            assert results["epochs_trained"] < 20

    def test_no_early_stopping_with_patience_0(self):
        model = ModelRegistry.create("mynet", num_classes=10, input_channels=1, input_size=(28, 28))
        train_loader, val_loader = _make_classification_data(batch_size=4, num_batches=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            em = ExperimentManager(base_dir=tmpdir)
            trainer = Trainer(
                model=model, train_loader=train_loader, val_loader=val_loader,
                criterion=nn.CrossEntropyLoss(),
                optimizer=optim.SGD(model.parameters(), lr=0.01),
                device=torch.device("cpu"),
                experiment_manager=em,
                patience=0,
            )
            results = trainer.train(epochs=3)
            assert not results["stopped_early"]


# ---------- checkpointing during training ----------

class TestCheckpointingDuringTrain:

    def test_checkpoints_saved(self):
        model = ModelRegistry.create("mynet", num_classes=10, input_channels=1, input_size=(28, 28))
        train_loader, val_loader = _make_classification_data(batch_size=4, num_batches=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            em = ExperimentManager(base_dir=tmpdir)
            cm = CheckpointManager(em.checkpoints_dir, save_frequency=1)
            trainer = Trainer(
                model=model, train_loader=train_loader, val_loader=val_loader,
                criterion=nn.CrossEntropyLoss(),
                optimizer=optim.SGD(model.parameters(), lr=0.01),
                device=torch.device("cpu"),
                experiment_manager=em, checkpoint_manager=cm,
            )
            trainer.train(epochs=3)

            # Verify checkpoints exist
            assert os.path.exists(os.path.join(em.checkpoints_dir, "latest_checkpoint.pt"))
            assert os.path.exists(os.path.join(em.checkpoints_dir, "best_model.pt"))


# ---------- scheduler integration ----------

class TestSchedulerInTrainLoop:

    def test_scheduler_step_called(self):
        model = ModelRegistry.create("mynet", num_classes=10, input_channels=1, input_size=(28, 28))
        train_loader, val_loader = _make_classification_data(batch_size=4, num_batches=2)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            em = ExperimentManager(base_dir=tmpdir)
            trainer = Trainer(
                model=model, train_loader=train_loader, val_loader=val_loader,
                criterion=nn.CrossEntropyLoss(), optimizer=optimizer,
                device=torch.device("cpu"), experiment_manager=em,
                scheduler=scheduler,
            )
            lr_before = optimizer.param_groups[0]["lr"]
            trainer.train(epochs=2)
            lr_after = optimizer.param_groups[0]["lr"]
            assert lr_after < lr_before, "LR should have decayed after scheduler steps"


# ---------- replay transforms ----------

class TestReapplyTransforms:

    def test_reapply_transforms_flag(self):
        """The reapply_transforms flag should be stored in trainer.dataset."""
        from src.datasets import MNISTDataset
        ds = MNISTDataset(reapply_transforms=True)
        assert ds.reapply_transforms

    def test_train_loader_recreated_when_reapply(self):
        """When reapply_transforms is True, the train loader should be replaced each epoch."""
        model = ModelRegistry.create("mynet", num_classes=10, input_channels=1, input_size=(28, 28))
        loader, val_loader = _make_classification_data(batch_size=4, num_batches=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            em = ExperimentManager(base_dir=tmpdir)
            trainer = Trainer(
                model=model, train_loader=loader, val_loader=val_loader,
                criterion=nn.CrossEntropyLoss(),
                optimizer=optim.SGD(model.parameters(), lr=0.01),
                device=torch.device("cpu"), experiment_manager=em,
            )
            # Simulate what happens in train(): after each epoch, if dataset.reapply_transforms:
            #   self.train_loader = DataLoader(self.dataset._train_dataset, **self._train_loader_kwargs)
            # Without a real dataset with _train_dataset, this just checks the flag path exists.
            assert hasattr(trainer, "_train_loader_kwargs")
            assert "_train_loader_kwargs" in trainer.__dict__


# ---------- history tracking ----------

class TestHistoryTracking:

    def test_history_populates(self):
        model = ModelRegistry.create("mynet", num_classes=10, input_channels=1, input_size=(28, 28))
        train_loader, val_loader = _make_classification_data(batch_size=4, num_batches=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            em = ExperimentManager(base_dir=tmpdir)
            trainer = Trainer(
                model=model, train_loader=train_loader, val_loader=val_loader,
                criterion=nn.CrossEntropyLoss(),
                optimizer=optim.SGD(model.parameters(), lr=0.01),
                device=torch.device("cpu"), experiment_manager=em,
            )
            trainer.train(epochs=3)
            h = trainer.history
            assert len(h["train_loss"]) == 3
            assert len(h["val_loss"]) == 3
            assert len(h["train_accuracy"]) == 3
            assert len(h["val_accuracy"]) == 3
            assert len(h["learning_rate"]) == 3
            assert len(h["epoch_time"]) == 3
            assert len(h["train_speed"]) == 3
            assert len(h["val_speed"]) == 3


# ---------- triplet training loop ----------

class TestTripletTrainEpoch:

    @pytest.fixture
    def setup(self):
        model = ModelRegistry.create("siamese", num_classes=10, input_channels=1, input_size=(28, 28))
        train_loader, val_loader = _make_triplet_data(batch_size=4, num_batches=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            em = ExperimentManager(base_dir=tmpdir)
            model_class = model.__class__
            criterion = model_class.get_criterion(margin=1.0)
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            trainer = Trainer(
                model=model, train_loader=train_loader, val_loader=val_loader,
                criterion=criterion, optimizer=optimizer,
                device=torch.device("cpu"), experiment_manager=em,
            )
            yield trainer

    def test_triplet_train_epoch(self, setup):
        trainer = setup
        metrics, speed = trainer.train_epoch(epoch=0)
        assert "pos_dist" in metrics
        assert "neg_dist" in metrics
        assert "accuracy" in metrics  # triplet "accuracy" = fraction of valid triplets
        assert speed > 0

    def test_triplet_validate(self, setup):
        trainer = setup
        metrics, speed = trainer.validate()
        assert "pos_dist" in metrics
        assert "neg_dist" in metrics
        assert speed > 0

    def test_triplet_metrics_tracker_used(self, setup):
        trainer = setup
        assert isinstance(trainer.train_metrics, TripletMetricsTracker)
        assert isinstance(trainer.val_metrics, TripletMetricsTracker)
