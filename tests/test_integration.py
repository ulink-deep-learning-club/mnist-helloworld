"""Integration tests: end-to-end workflows combining config, dataset, model, trainer, checkpoint."""

import os
import tempfile
import copy

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config, create_config_parser, get_default_config
from src.models import ModelRegistry
from src.training import Trainer, CheckpointManager, ExperimentManager
from train import create_optimizer, create_scheduler, freeze_layers


# ========================================================================
# Helpers
# ========================================================================

def make_config(**overrides):
    """Build a Config from overrides on top of defaults."""
    d = copy.deepcopy(get_default_config())
    for section, values in overrides.items():
        if section in d and isinstance(d[section], dict):
            d[section].update(values)
        else:
            d[section] = values
    return Config(d)


def synthetic_classification_loader(batch_size=4, num_batches=3, num_classes=10, ch=1, hw=28):
    xs = torch.randn(batch_size * num_batches, ch, hw, hw)
    ys = torch.randint(0, num_classes, (batch_size * num_batches,))
    ds = TensorDataset(xs, ys)
    return DataLoader(ds, batch_size=batch_size)


def synthetic_triplet_loader(batch_size=4, num_batches=3, ch=1, hw=28):
    xs = torch.randn(batch_size * num_batches, ch, hw, hw)
    ds = TensorDataset(xs, xs, xs, torch.zeros(batch_size * num_batches, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size)


# ========================================================================
# 1.  End-to-end: Config → Model → Optimizer → Trainer → Train → Checkpoint
# ========================================================================

class TestFullPipeline:
    """The golden path: a complete train.py-style pipeline on synthetic data."""

    def test_config_to_train_to_checkpoint(self):
        """Simulate the full train.py main() flow end-to-end."""
        # ---- Config ----
        config = make_config(
            model={"name": "mynet", "num_classes": 10, "input_channels": 1, "input_size": [28, 28]},
            training={"epochs": 3, "batch_size": 4},
            optimization={"optimizer": "adamw", "learning_rate": 1e-3, "weight_decay": 0},
            checkpointing={"save_frequency": 1},
        )

        # ---- Model ----
        model = ModelRegistry.create(
            config.model["name"],
            num_classes=config.model["num_classes"],
            input_channels=config.model.get("input_channels", 1),
            input_size=tuple(config.model.get("input_size", [28, 28])),
        )

        # ---- Data ----
        loader = synthetic_classification_loader(
            batch_size=config.training["batch_size"],
            num_classes=config.model["num_classes"],
        )

        # ---- Optimizer / Criterion ----
        optimizer = create_optimizer(model, config)
        model_class = ModelRegistry.get(config.model["name"])
        criterion = model_class.get_criterion()

        # ---- Trainer ----
        with tempfile.TemporaryDirectory() as tmpdir:
            em = ExperimentManager(base_dir=tmpdir)
            cm = CheckpointManager(em.checkpoints_dir, save_frequency=config.checkpointing["save_frequency"])
            trainer = Trainer(
                model=model,
                train_loader=loader,
                val_loader=loader,
                criterion=criterion,
                optimizer=optimizer,
                device=torch.device("cpu"),
                experiment_manager=em,
                checkpoint_manager=cm,
            )

            # ---- Train ----
            results = trainer.train(epochs=config.training["epochs"])

            # ---- Verify ----
            assert results["epochs_trained"] == 3
            assert results["best_accuracy"] >= 0
            assert results["training_time"] > 0
            assert os.path.exists(os.path.join(em.checkpoints_dir, "latest_checkpoint.pt"))
            assert os.path.exists(os.path.join(em.checkpoints_dir, "best_model.pt"))
            assert os.path.exists(em.log_file)

            # Log file has correct content
            with open(em.log_file) as f:
                lines = f.readlines()
            assert len(lines) == 4  # header + 3 epochs
            assert lines[0].startswith("epoch")


# ========================================================================
# 2.  Resume training — continuity check
# ========================================================================

class TestResumeIntegration:
    """Train, save checkpoint, resume, verify epoch continuity."""

    def test_resume_continues_from_checkpoint(self):
        config = make_config(
            model={"name": "mynet"},
            training={"epochs": 5, "batch_size": 4},
            optimization={"optimizer": "sgd", "learning_rate": 0.01, "weight_decay": 0, "momentum": 0.9},
            checkpointing={"save_frequency": 1},
        )
        model = ModelRegistry.create("mynet", num_classes=10, input_channels=1, input_size=(28, 28))
        loader = synthetic_classification_loader(batch_size=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            # ---- Phase 1: Train for 2 epochs ----
            em1 = ExperimentManager(base_dir=tmpdir)
            cm1 = CheckpointManager(em1.checkpoints_dir, save_frequency=1)
            opt1 = create_optimizer(model, config)
            trainer1 = Trainer(
                model=model, train_loader=loader, val_loader=loader,
                criterion=nn.CrossEntropyLoss(), optimizer=opt1,
                device=torch.device("cpu"), experiment_manager=em1, checkpoint_manager=cm1,
            )
            trainer1.train(epochs=2)

            # ---- Phase 2: Resume into same experiment with fresh Trainer ----
            model2 = ModelRegistry.create("mynet", num_classes=10, input_channels=1, input_size=(28, 28))
            em2 = ExperimentManager(base_dir=tmpdir, resume_exp=os.path.basename(em1.experiment_dir))
            cm2 = CheckpointManager(em2.checkpoints_dir, save_frequency=1)
            opt2 = create_optimizer(model2, config)

            # Load checkpoint
            ckpt_path = os.path.join(em2.checkpoints_dir, "latest_checkpoint.pt")
            assert os.path.exists(ckpt_path)
            cm2.load_checkpoint(ckpt_path, model2, opt2, strict=True)

            trainer2 = Trainer(
                model=model2, train_loader=loader, val_loader=loader,
                criterion=nn.CrossEntropyLoss(), optimizer=opt2,
                device=torch.device("cpu"), experiment_manager=em2, checkpoint_manager=cm2,
            )
            history = trainer2.load_history_from_log()
            assert history >= 2, f"Expected at least 2 epochs of history, got {history}"

            # Train 3 more epochs
            results = trainer2.train(epochs=5, start_epoch=history)
            assert results["epochs_trained"] == 5 - history


# ========================================================================
# 3.  Fork training — load from source, save to new experiment
# ========================================================================

class TestForkIntegration:
    """Fork from a trained experiment into a new one."""

    def test_fork_creates_new_experiment(self):
        config = make_config(
            model={"name": "mynet"},
            training={"epochs": 2, "batch_size": 4},
            optimization={"optimizer": "sgd", "learning_rate": 0.01, "weight_decay": 0, "momentum": 0.9},
            checkpointing={"save_frequency": 1},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # ---- Train source ----
            model = ModelRegistry.create("mynet", num_classes=10, input_channels=1, input_size=(28, 28))
            loader = synthetic_classification_loader(batch_size=4)
            em_src = ExperimentManager(base_dir=tmpdir)
            cm_src = CheckpointManager(em_src.checkpoints_dir, save_frequency=1)
            opt = create_optimizer(model, config)
            trainer = Trainer(
                model=model, train_loader=loader, val_loader=loader,
                criterion=nn.CrossEntropyLoss(), optimizer=opt,
                device=torch.device("cpu"), experiment_manager=em_src, checkpoint_manager=cm_src,
            )
            trainer.train(epochs=2)

            exp_name = os.path.basename(em_src.experiment_dir)

            # ---- Fork: create new experiment from source ----
            model_fork = ModelRegistry.create("mynet", num_classes=10, input_channels=1, input_size=(28, 28))
            em_fork = ExperimentManager(base_dir=tmpdir, fork_exp=exp_name)
            cm_fork = CheckpointManager(em_fork.checkpoints_dir, save_frequency=1)
            opt_fork = create_optimizer(model_fork, config)

            # Load from source checkpoints dir
            source_ckpt = os.path.join(em_src.checkpoints_dir, "latest_checkpoint.pt")
            cm_fork.load_checkpoint(source_ckpt, model_fork, opt_fork, strict=True)

            trainer_fork = Trainer(
                model=model_fork, train_loader=loader, val_loader=loader,
                criterion=nn.CrossEntropyLoss(), optimizer=opt_fork,
                device=torch.device("cpu"), experiment_manager=em_fork, checkpoint_manager=cm_fork,
            )
            results = trainer_fork.train(epochs=3)

            # Fork should have its own experiment dir (different from source)
            assert em_fork.experiment_dir != em_src.experiment_dir
            assert results["epochs_trained"] == 3


# ========================================================================
# 4.  Layer freezing + training integration
# ========================================================================

class TestFreezeAndTrain:
    """Freeze layers, train, verify frozen params stay unchanged."""

    def test_frozen_weights_unchanged_after_training(self):
        config = make_config(
            model={"name": "mynet"},
            training={"epochs": 2, "batch_size": 4},
            optimization={"optimizer": "sgd", "learning_rate": 0.1, "weight_decay": 0, "momentum": 0},
            checkpointing={"save_frequency": 1},
        )
        model = ModelRegistry.create("mynet", num_classes=10, input_channels=1, input_size=(28, 28))
        loader = synthetic_classification_loader(batch_size=4)

        # Record weights before training
        weight_snapshots = {}
        for name, param in model.named_parameters():
            if "features" in name:  # Freeze feature extractor
                weight_snapshots[name] = param.data.clone()

        # Freeze feature extractor (name pattern)
        freeze_layers(model, ["features"], logger=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            em = ExperimentManager(base_dir=tmpdir)
            opt = create_optimizer(model, config)
            trainer = Trainer(
                model=model, train_loader=loader, val_loader=loader,
                criterion=nn.CrossEntropyLoss(), optimizer=opt,
                device=torch.device("cpu"), experiment_manager=em,
            )
            trainer.train(epochs=2)

        # Verify frozen weights are identical
        for name, param in model.named_parameters():
            if name in weight_snapshots:
                assert torch.equal(param.data, weight_snapshots[name]), \
                    f"Frozen param {name} changed after training"


# ========================================================================
# 5.  Cross-optimizer training integration
# ========================================================================

class TestAllOptimizersTrainStep:
    """Each optimizer should complete a full forward + backward + step."""

    @pytest.mark.parametrize("opt_name", ["adamw", "adam", "sgd", "muon", "muon_with_aux_adam"])
    def test_optimizer_completes_training_step(self, opt_name):
        config = make_config(
            model={"name": "mynet"},
            training={"epochs": 1, "batch_size": 4},
            optimization={
                "optimizer": opt_name, "learning_rate": 1e-3, "weight_decay": 0,
                "momentum": 0.9, "muon_momentum": 0.95, "muon_ns_steps": 5,
                "adam_lr": 3e-4, "adam_betas": [0.9, 0.95],
            },
        )
        model = ModelRegistry.create("mynet", num_classes=10, input_channels=1, input_size=(28, 28))
        loader = synthetic_classification_loader(batch_size=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            em = ExperimentManager(base_dir=tmpdir)
            opt = create_optimizer(model, config)
            criterion = nn.CrossEntropyLoss()
            trainer = Trainer(
                model=model, train_loader=loader, val_loader=loader,
                criterion=criterion, optimizer=opt,
                device=torch.device("cpu"), experiment_manager=em,
            )
            results = trainer.train(epochs=1)
            assert results["epochs_trained"] == 1
            assert results["best_accuracy"] >= 0


# ========================================================================
# 6.  Multiple epochs with history growth
# ========================================================================

class TestHistoryGrowth:
    """History should accumulate across multiple train() calls (simulating resume)."""

    def test_history_appends_across_calls(self):
        model = ModelRegistry.create("mynet", num_classes=10, input_channels=1, input_size=(28, 28))
        loader = synthetic_classification_loader(batch_size=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            em = ExperimentManager(base_dir=tmpdir)
            opt = optim.SGD(model.parameters(), lr=0.01)
            trainer = Trainer(
                model=model, train_loader=loader, val_loader=loader,
                criterion=nn.CrossEntropyLoss(), optimizer=opt,
                device=torch.device("cpu"), experiment_manager=em,
            )

            trainer.train(epochs=2)
            assert len(trainer.history["train_loss"]) == 2

            # Second call (resume-style)
            trainer.train(epochs=4, start_epoch=2)
            assert len(trainer.history["train_loss"]) == 4


# ========================================================================
# 7.  CLI argument → Config → Trainer consistency
# ========================================================================

class TestCLIToTrainer:
    """Parse CLI args, build config, create trainer, verify settings propagated."""

    def test_cli_args_flow_to_trainer(self):
        parser = create_config_parser()
        args = parser.parse_args([
            "--dataset", "cifar10",
            "--model", "alexnet",
            "--epochs", "5",
            "--batch-size", "8",
            "--learning-rate", "5e-4",
            "--optimizer", "sgd",
            "--save-frequency", "2",
        ])
        config = Config.from_args(args)

        # Verify config propagation
        assert config.dataset["name"] == "cifar10"
        assert config.model["name"] == "alexnet"
        assert config.training["epochs"] == 5
        assert config.optimization["learning_rate"] == 5e-4
        assert config.checkpointing["save_frequency"] == 2

        # Create model and trainer
        model = ModelRegistry.create(
            config.model["name"],
            num_classes=config.model["num_classes"],
            input_channels=3,
            input_size=(32, 32),
        )
        loader = synthetic_classification_loader(batch_size=8, ch=3, hw=32)

        with tempfile.TemporaryDirectory() as tmpdir:
            em = ExperimentManager(base_dir=tmpdir)
            opt = create_optimizer(model, config)
            assert isinstance(opt, optim.SGD)
            assert opt.param_groups[0]["lr"] == 5e-4

            trainer = Trainer(
                model=model, train_loader=loader, val_loader=loader,
                criterion=nn.CrossEntropyLoss(), optimizer=opt,
                device=torch.device("cpu"), experiment_manager=em,
            )
            assert trainer.patience == 0  # default
            assert not trainer.use_amp

            results = trainer.train(epochs=2)
            assert results["epochs_trained"] == 2


# ========================================================================
# 8.  Scheduler + training integration
# ========================================================================

class TestSchedulerIntegration:
    """Scheduler should actually change LR across epochs during training."""

    @pytest.mark.parametrize("sched_name", ["step", "cosine", "exponential"])
    def test_lr_decays_across_epochs(self, sched_name):
        config = make_config(
            model={"name": "mynet"},
            training={"epochs": 5, "batch_size": 4},
            optimization={
                "optimizer": "sgd", "learning_rate": 0.1, "weight_decay": 0, "momentum": 0,
                "scheduler": sched_name,
                "scheduler_step_size": 2, "scheduler_gamma": 0.5,
                "scheduler_t_max": 5, "scheduler_eta_min": 1e-6,
            },
        )
        model = ModelRegistry.create("mynet", num_classes=10, input_channels=1, input_size=(28, 28))
        loader = synthetic_classification_loader(batch_size=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            em = ExperimentManager(base_dir=tmpdir)
            opt = create_optimizer(model, config)
            sched = create_scheduler(opt, config)
            assert sched is not None

            trainer = Trainer(
                model=model, train_loader=loader, val_loader=loader,
                criterion=nn.CrossEntropyLoss(), optimizer=opt,
                device=torch.device("cpu"), experiment_manager=em,
                scheduler=sched,
            )
            lrs_before = [pg["lr"] for pg in opt.param_groups]
            trainer.train(epochs=5)
            lrs_after = [pg["lr"] for pg in opt.param_groups]

            for before, after in zip(lrs_before, lrs_after):
                assert after < before, f"LR should decay with {sched_name} scheduler"
