"""Test configuration loading and consistency."""

import os
import tempfile


from src.config import Config, load_config, create_config_parser, get_default_config


class TestDefaultConfig:
    """Default configuration values should be consistent."""

    def test_default_values(self):
        cfg = get_default_config()
        assert cfg["dataset"]["name"] == "mnist"
        assert cfg["model"]["name"] == "mynet"
        assert cfg["model"]["num_classes"] == 10
        assert cfg["model"]["input_channels"] == 1
        assert cfg["training"]["epochs"] == 20
        assert cfg["training"]["batch_size"] == 64
        assert cfg["optimization"]["learning_rate"] == 1e-3
        assert cfg["optimization"]["optimizer"] == "adamw"
        assert cfg["optimization"]["weight_decay"] == 0.01
        assert cfg["checkpointing"]["save_frequency"] == 10

    def test_load_without_args_returns_default(self):
        config = load_config()
        assert config.dataset["name"] == "mnist"
        assert config.model["name"] == "mynet"


class TestConfigFromYAML:
    """Config should load correctly from YAML files."""

    def test_load_yaml(self):
        yaml_content = """
dataset:
  name: cifar10
  root: ./data
model:
  name: alexnet
  num_classes: 10
training:
  epochs: 5
  batch_size: 16
optimization:
  learning_rate: 0.01
  optimizer: sgd
checkpointing:
  save_frequency: 2
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            tmp_path = f.name

        try:
            config = Config.from_yaml(tmp_path)
            assert config.dataset["name"] == "cifar10"
            assert config.model["name"] == "alexnet"
            assert config.training["epochs"] == 5
            assert config.optimization["learning_rate"] == 0.01
            assert config.checkpointing["save_frequency"] == 2
        finally:
            os.unlink(tmp_path)

    def test_yaml_with_all_fields(self):
        """Full YAML config with all optional fields should load cleanly."""
        yaml_content = """
dataset:
  name: subset_631
  root: ./data
  download: true
model:
  name: fpn_vit
  num_classes: 631
  input_channels: 1
  input_size: [64, 64]
training:
  epochs: 100
  batch_size: 32
  num_workers: 4
optimization:
  learning_rate: 1e-4
  optimizer: adamw
  weight_decay: 0.01
  scheduler: cosine
  scheduler_t_max: 100
  scheduler_eta_min: 1e-6
  muon_momentum: 0.95
  muon_ns_steps: 5
checkpointing:
  checkpoint_dir: checkpoints
  save_frequency: 10
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            tmp_path = f.name

        try:
            config = Config.from_yaml(tmp_path)
            assert config.dataset["name"] == "subset_631"
            assert config.model["name"] == "fpn_vit"
            assert config.optimization["scheduler"] == "cosine"
            assert config.optimization["scheduler_t_max"] == 100
        finally:
            os.unlink(tmp_path)


class TestConfigFromArgs:
    """Config from CLI args should correctly populate all fields."""

    def test_minimal_args(self):
        parser = create_config_parser()
        args = parser.parse_args([])  # All defaults
        config = Config.from_args(args)

        assert config.dataset["name"] == "mnist"
        assert config.model["name"] == "mynet"
        assert config.training["epochs"] == 20
        assert config.optimization["learning_rate"] == 1e-3
        assert config.checkpointing["save_frequency"] == 10  # Should match CLI default

    def test_custom_args(self):
        parser = create_config_parser()
        args = parser.parse_args([
            "--dataset", "cifar10",
            "--model", "alexnet",
            "--epochs", "50",
            "--learning-rate", "1e-4",
            "--optimizer", "sgd",
            "--batch-size", "128",
        ])
        config = Config.from_args(args)

        assert config.dataset["name"] == "cifar10"
        assert config.model["name"] == "alexnet"
        assert config.training["epochs"] == 50
        assert config.optimization["learning_rate"] == 1e-4
        assert config.optimization["optimizer"] == "sgd"

    def test_embedding_dim_only_in_model_not_optimization(self):
        """embedding_dim should only appear in model config, not optimization."""
        parser = create_config_parser()
        args = parser.parse_args([])
        config = Config.from_args(args)

        assert "embedding_dim" in config.model, "embedding_dim should be in model config"
        assert "embedding_dim" not in config.optimization, "embedding_dim should NOT be in optimization config"


class TestConfigParser:
    """Argument parser should have correct choices and defaults."""

    def test_optimizer_choices(self):
        parser = create_config_parser()
        opt_action = [a for a in parser._actions if a.dest == "optimizer"][0]
        expected = ["adamw", "adam", "sgd", "muon", "muon_with_aux_adam"]
        assert list(opt_action.choices) == expected

    def test_scheduler_choices(self):
        parser = create_config_parser()
        sched_action = [a for a in parser._actions if a.dest == "scheduler"][0]
        expected = ["none", "step", "cosine", "plateau", "exponential"]
        assert list(sched_action.choices) == expected

    def test_save_frequency_default(self):
        """CLI default for save-frequency should be 10 (matching config.yaml)."""
        parser = create_config_parser()
        action = [a for a in parser._actions if a.dest == "save_frequency"][0]
        assert action.default == 10, f"Expected default=10, got {action.default}"
