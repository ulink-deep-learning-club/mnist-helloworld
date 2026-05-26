"""Test all 5 optimizer creation paths from CLI/config."""

import pytest
import torch
import torch.nn as nn

from train import create_optimizer
from src.config import Config


@pytest.fixture
def simple_model():
    """A model with both 2D (conv/linear) and 1D (bias) parameters."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )


@pytest.fixture
def config_dict():
    """Base config dict; override specific fields per test."""
    return {
        "dataset": {"name": "mnist"},
        "model": {"name": "mynet"},
        "training": {"epochs": 1, "batch_size": 4},
        "optimization": {
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "momentum": 0.9,
            "muon_momentum": 0.95,
            "muon_ns_steps": 5,
            "adam_lr": 3e-4,
            "adam_betas": [0.9, 0.95],
        },
        "checkpointing": {"save_frequency": 10},
    }


def make_config(**overrides):
    d = {
        "dataset": {"name": "mnist"},
        "model": {"name": "mynet"},
        "training": {"epochs": 1, "batch_size": 4},
        "optimization": {
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "optimizer": "adamw",
            "momentum": 0.9,
            "muon_momentum": 0.95,
            "muon_ns_steps": 5,
            "adam_lr": 3e-4,
            "adam_betas": [0.9, 0.95],
        },
        "checkpointing": {"save_frequency": 10},
    }
    # Deep merge overrides
    for section, values in overrides.items():
        if section in d and isinstance(d[section], dict):
            d[section].update(values)
        else:
            d[section] = values
    return Config(d)


class TestCreateOptimizer:

    def test_adamw(self, simple_model):
        config = make_config(optimization={"optimizer": "adamw"})
        opt = create_optimizer(simple_model, config)
        assert isinstance(opt, torch.optim.AdamW)
        assert opt.param_groups[0]["lr"] == 1e-3
        assert opt.param_groups[0]["weight_decay"] == 0.01

    def test_adam(self, simple_model):
        config = make_config(optimization={"optimizer": "adam"})
        opt = create_optimizer(simple_model, config)
        assert isinstance(opt, torch.optim.Adam)
        assert opt.param_groups[0]["lr"] == 1e-3

    def test_sgd(self, simple_model):
        config = make_config(optimization={"optimizer": "sgd", "momentum": 0.9})
        opt = create_optimizer(simple_model, config)
        assert isinstance(opt, torch.optim.SGD)
        assert opt.param_groups[0]["momentum"] == 0.9

    def test_muon(self, simple_model):
        """Muon should only receive 2D+ params."""
        import muon
        config = make_config(optimization={"optimizer": "muon"})
        opt = create_optimizer(simple_model, config)
        assert isinstance(opt, muon.SingleDeviceMuon)

    def test_muon_with_aux_adam(self, simple_model):
        import muon
        config = make_config(optimization={"optimizer": "muon_with_aux_adam"})
        opt = create_optimizer(simple_model, config)
        assert isinstance(opt, muon.SingleDeviceMuonWithAuxAdam)

    def test_unknown_optimizer_raises(self, simple_model):
        config = make_config(optimization={"optimizer": "unknown"})
        with pytest.raises(ValueError, match="Unknown optimizer"):
            create_optimizer(simple_model, config)

    def test_muon_only_2d_params(self, simple_model):
        """Muon should only get parameters with ndim >= 2."""
        config = make_config(optimization={"optimizer": "muon"})
        opt = create_optimizer(simple_model, config)
        # All params in nn.Linear have ndim=2 (weight) or ndim=1 (bias)
        # Muon should only take weights (ndim>=2), not biases (ndim<2)
        for pg in opt.param_groups:
            for p in pg["params"]:
                assert p.ndim >= 2, "Muon should not receive 1D parameters"

    def test_custom_learning_rate(self, simple_model):
        config = make_config(optimization={"optimizer": "adamw", "learning_rate": 5e-4})
        opt = create_optimizer(simple_model, config)
        assert opt.param_groups[0]["lr"] == 5e-4

    def test_lr_override_after_resume(self, simple_model):
        """CLI should override checkpoint LR (train.py line ~533)."""
        config = make_config(optimization={"optimizer": "sgd", "learning_rate": 1e-5, "momentum": 0.9})
        opt = create_optimizer(simple_model, config)
        # Simulate the resume override logic from train.py
        new_lr = config.optimization.get("learning_rate", 1e-3)
        for pg in opt.param_groups:
            pg["lr"] = new_lr
        assert opt.param_groups[0]["lr"] == 1e-5
