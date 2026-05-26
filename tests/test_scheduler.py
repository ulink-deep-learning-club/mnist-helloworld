"""Test all 5 LR scheduler creation paths + scheduler step behavior."""

import pytest
import torch.nn as nn
import torch.optim as optim

from train import create_scheduler
from src.config import Config


@pytest.fixture
def model_and_optimizer():
    model = nn.Linear(10, 2)
    opt = optim.SGD(model.parameters(), lr=0.1)
    return model, opt


def make_config(**overrides):
    d = {
        "dataset": {"name": "mnist"},
        "model": {"name": "mynet"},
        "training": {"epochs": 5, "batch_size": 4},
        "optimization": {
            "optimizer": "sgd",
            "learning_rate": 0.1,
            "weight_decay": 0,
            "momentum": 0,
            "scheduler": "none",
            "scheduler_step_size": 2,
            "scheduler_gamma": 0.5,
            "scheduler_t_max": 5,
            "scheduler_eta_min": 1e-6,
            "scheduler_patience": 2,
            "scheduler_factor": 0.5,
        },
        "checkpointing": {"save_frequency": 10},
    }
    for section, values in overrides.items():
        if section in d and isinstance(d[section], dict):
            d[section].update(values)
        else:
            d[section] = values
    return Config(d)


class TestCreateScheduler:

    def test_none(self, model_and_optimizer):
        _, opt = model_and_optimizer
        config = make_config(optimization={"scheduler": "none"})
        sched = create_scheduler(opt, config)
        assert sched is None

    def test_step(self, model_and_optimizer):
        _, opt = model_and_optimizer
        config = make_config(optimization={"scheduler": "step", "scheduler_step_size": 2, "scheduler_gamma": 0.5})
        sched = create_scheduler(opt, config)
        assert isinstance(sched, optim.lr_scheduler.StepLR)
        assert sched.step_size == 2
        assert sched.gamma == 0.5

    def test_cosine(self, model_and_optimizer):
        _, opt = model_and_optimizer
        config = make_config(optimization={"scheduler": "cosine", "scheduler_t_max": 10, "scheduler_eta_min": 1e-6})
        sched = create_scheduler(opt, config)
        assert isinstance(sched, optim.lr_scheduler.CosineAnnealingLR)
        assert sched.T_max == 10

    def test_plateau(self, model_and_optimizer):
        _, opt = model_and_optimizer
        config = make_config(optimization={"scheduler": "plateau", "scheduler_patience": 3, "scheduler_factor": 0.3})
        sched = create_scheduler(opt, config)
        assert isinstance(sched, optim.lr_scheduler.ReduceLROnPlateau)
        assert sched.patience == 3
        assert sched.factor == 0.3
        assert sched.mode == "max"  # hardcoded in create_scheduler

    def test_exponential(self, model_and_optimizer):
        _, opt = model_and_optimizer
        config = make_config(optimization={"scheduler": "exponential", "scheduler_gamma": 0.95})
        sched = create_scheduler(opt, config)
        assert isinstance(sched, optim.lr_scheduler.ExponentialLR)
        assert sched.gamma == 0.95

    def test_unknown_raises(self, model_and_optimizer):
        _, opt = model_and_optimizer
        config = make_config(optimization={"scheduler": "invalid"})
        with pytest.raises(ValueError, match="Unknown scheduler"):
            create_scheduler(opt, config)


class TestSchedulerStepBehavior:
    """Verify scheduler stepping matches trainer.py behavior."""

    def test_step_lr_decays(self, model_and_optimizer):
        _, opt = model_and_optimizer
        config = make_config(optimization={"scheduler": "step", "scheduler_step_size": 1, "scheduler_gamma": 0.5})
        sched = create_scheduler(opt, config)
        lr_before = opt.param_groups[0]["lr"]
        sched.step()
        lr_after = opt.param_groups[0]["lr"]
        assert lr_after == lr_before * 0.5

    def test_cosine_annealing(self, model_and_optimizer):
        _, opt = model_and_optimizer
        config = make_config(optimization={"scheduler": "cosine", "scheduler_t_max": 10, "scheduler_eta_min": 0})
        sched = create_scheduler(opt, config)
        # After T_max steps, LR should be at eta_min
        for _ in range(10):
            sched.step()
        assert opt.param_groups[0]["lr"] == 0.0

    def test_plateau_on_loss_metric(self, model_and_optimizer):
        """Plateau scheduler steps on val_loss from trainer.py."""
        _, opt = model_and_optimizer
        config = make_config(optimization={"scheduler": "plateau", "scheduler_factor": 0.5, "scheduler_patience": 1})
        sched = create_scheduler(opt, config)
        lr_before = opt.param_groups[0]["lr"]

        # Simulate trainer behavior: sched.step(val_metrics["loss"])
        # patience=1 means LR decays after 2 consecutive non-improving steps
        for _ in range(3):
            sched.step(1.0)  # no improvement
        lr_after = opt.param_groups[0]["lr"]
        assert lr_after < lr_before

    def test_trainer_scheduler_step_logic(self, model_and_optimizer):
        """Replicate the trainer.py scheduler step logic."""
        _, opt = model_and_optimizer
        config = make_config(optimization={"scheduler": "step", "scheduler_step_size": 2, "scheduler_gamma": 0.5})
        sched = create_scheduler(opt, config)

        initial_lr = opt.param_groups[0]["lr"]

        # Simulate trainer loop with epoch-based stepping
        for epoch in range(5):
            if sched is not None:
                if isinstance(sched, optim.lr_scheduler.ReduceLROnPlateau):
                    sched.step(0.5)  # val_loss
                else:
                    sched.step()

        # After 5 epochs with step_size=2, gamma=0.5: LR stepped at epoch 2 and 4
        assert opt.param_groups[0]["lr"] == initial_lr * 0.5 * 0.5
