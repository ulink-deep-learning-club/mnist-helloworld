"""Test configuration loading and consistency."""

import os
import tempfile

from src.config import Config, load_config, create_config_parser, get_default_config
from src.config.config import (
    _deep_merge,
    _coerce_numeric_types,
    _NUMERIC_SCHEMA,
    _build_user_cli_dict,
)


# ========================================================================
# Unit: _deep_merge
# ========================================================================

class TestDeepMerge:
    """_deep_merge produces correct merged dicts."""

    def test_basic_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"opt": {"lr": 1e-3, "wd": 0.01}}
        override = {"opt": {"lr": 1e-4}}
        result = _deep_merge(base, override)
        assert result == {"opt": {"lr": 1e-4, "wd": 0.01}}

    def test_override_with_non_dict_replaces_entire_value(self):
        base = {"opt": {"lr": 1e-3, "wd": 0.01}}
        override = {"opt": "string"}
        result = _deep_merge(base, override)
        assert result == {"opt": "string"}

    def test_deepcopy_isolation(self):
        base = {"inner": {"x": 1}}
        override = {"inner": {"x": 2}}
        result = _deep_merge(base, override)
        # Mutating result should not affect base
        result["inner"]["y"] = 3
        assert "y" not in base["inner"]

    def test_empty_override(self):
        base = {"a": 1, "b": {"c": 2}}
        result = _deep_merge(base, {})
        assert result == base
        # Should be a deep copy
        result["b"]["c"] = 99
        assert base["b"]["c"] == 2

    def test_list_values_not_recursively_merged(self):
        """Lists should be replaced, not element-wise merged."""
        base = {"betas": [0.9, 0.95]}
        override = {"betas": [0.8, 0.9]}
        result = _deep_merge(base, override)
        assert result == {"betas": [0.8, 0.9]}


# ========================================================================
# Unit: _coerce_numeric_types
# ========================================================================

class TestNumericCoercion:
    """_coerce_numeric_types converts string values to correct types."""

    def test_string_learning_rate_to_float(self):
        d = {"optimization": {"learning_rate": "1e-3"}}
        result = _coerce_numeric_types(d)
        val = result["optimization"]["learning_rate"]
        assert val == 0.001
        assert isinstance(val, float)

    def test_string_epochs_to_int(self):
        d = {"training": {"epochs": "20"}}
        result = _coerce_numeric_types(d)
        assert result["training"]["epochs"] == 20
        assert isinstance(result["training"]["epochs"], int)

    def test_already_correct_type_unchanged(self):
        d = {"optimization": {"learning_rate": 0.001, "weight_decay": 0.01}}
        result = _coerce_numeric_types(d)
        assert result["optimization"]["learning_rate"] == 0.001
        assert isinstance(result["optimization"]["learning_rate"], float)
        assert result["optimization"]["weight_decay"] == 0.01

    def test_string_momentum_to_float(self):
        d = {"optimization": {"momentum": "0.9"}}
        result = _coerce_numeric_types(d)
        assert result["optimization"]["momentum"] == 0.9
        assert isinstance(result["optimization"]["momentum"], float)

    def test_invalid_string_left_untouched(self):
        """Non-numeric strings in numeric fields should be left as-is."""
        d = {"optimization": {"learning_rate": "not-a-number"}}
        result = _coerce_numeric_types(d)
        assert result["optimization"]["learning_rate"] == "not-a-number"

    def test_string_scheduler_step_size_to_int(self):
        d = {"optimization": {"scheduler_step_size": "10"}}
        result = _coerce_numeric_types(d)
        assert result["optimization"]["scheduler_step_size"] == 10
        assert isinstance(result["optimization"]["scheduler_step_size"], int)

    def test_string_adam_betas_to_float_list(self):
        d = {"optimization": {"adam_betas": ["0.8", "0.9"]}}
        result = _coerce_numeric_types(d)
        assert result["optimization"]["adam_betas"] == [0.8, 0.9]
        assert all(isinstance(x, float) for x in result["optimization"]["adam_betas"])

    def test_unknown_field_not_touched(self):
        d = {"custom": {"field": "string_value"}}
        result = _coerce_numeric_types(d)
        assert result["custom"]["field"] == "string_value"

    def test_missing_optional_field_no_error(self):
        d = {"optimization": {"learning_rate": 1e-3}}
        result = _coerce_numeric_types(d)
        # Missing optional numeric fields should not cause errors
        # (e.g., scheduler_patience not present)
        assert "scheduler_patience" not in result["optimization"]

    def test_full_config_dict_preserves_all_types(self):
        """After coercion, a full config dict should have all correct types."""
        d = get_default_config()
        # Insert some string values to simulate YAML quirks
        d["optimization"]["learning_rate"] = "1e-3"
        d["optimization"]["momentum"] = "0.9"
        d["training"]["epochs"] = "20"

        result = _coerce_numeric_types(d)
        assert isinstance(result["optimization"]["learning_rate"], float)
        assert isinstance(result["optimization"]["momentum"], float)
        assert isinstance(result["training"]["epochs"], int)

    def test_numeric_schema_completeness(self):
        """Every entry in _NUMERIC_SCHEMA should correspond to a real path
        in get_default_config() or from_args()."""
        cfg = get_default_config()
        for dotted_path in _NUMERIC_SCHEMA:
            parts = dotted_path.split(".")
            parent = cfg
            for part in parts:
                assert part in parent, (
                    f"_NUMERIC_SCHEMA path '{dotted_path}' "
                    f"not found in get_default_config()"
                )
                parent = parent[part]


# ========================================================================
# Unit: _build_user_cli_dict
# ========================================================================

class TestBuildUserCliDict:
    """_build_user_cli_dict extracts only explicitly provided CLI args."""

    def test_all_defaults_returns_empty(self):
        parser = create_config_parser()
        args = parser.parse_args([])
        overrides = _build_user_cli_dict(args)
        assert overrides == {}

    def test_single_override_returns_sparse_dict(self):
        parser = create_config_parser()
        args = parser.parse_args(["--epochs", "50"])
        overrides = _build_user_cli_dict(args)
        assert overrides == {"training": {"epochs": 50}}
        # Only the changed field should be present, not the entire section
        assert "batch_size" not in overrides.get("training", {})

    def test_multiple_overrides(self):
        parser = create_config_parser()
        args = parser.parse_args([
            "--epochs", "50",
            "--learning-rate", "1e-4",
            "--optimizer", "sgd",
        ])
        overrides = _build_user_cli_dict(args)
        assert overrides["training"]["epochs"] == 50
        assert overrides["optimization"]["learning_rate"] == 1e-4
        assert overrides["optimization"]["optimizer"] == "sgd"
        # batch_size should NOT be included since it's at default
        assert "batch_size" not in overrides.get("training", {})

    def test_model_override(self):
        parser = create_config_parser()
        args = parser.parse_args(["--model", "lenet"])
        overrides = _build_user_cli_dict(args)
        assert overrides == {"model": {"name": "lenet"}}


# ========================================================================
# Integration: load_config priority chain
# ========================================================================

class TestLoadConfigPriority:
    """load_config correctly applies the 4-tier priority."""

    def test_tier1_hardcoded_defaults_loaded_when_no_yaml_and_no_args(self):
        """Calling load_config() without arguments still loads hardcoded defaults
        merged with the project's default config.yaml."""
        config = load_config()
        # Values from hardcoded defaults
        assert config.dataset["name"] == "mnist"
        assert config.model["name"] == "mynet"
        assert config.training["epochs"] == 20
        # Numeric types must be correct (not strings)
        assert isinstance(config.optimization["learning_rate"], float)

    def test_tier3_user_yaml_overrides_default_yaml(self):
        """A user-specified YAML overrides the default config.yaml at project root."""
        user_yaml = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        user_yaml.write("""
model:
  name: lenet
training:
  epochs: 5
""")
        user_yaml.close()
        try:
            config = load_config(config_path=user_yaml.name)
            # Should pick up user values over default config.yaml
            assert config.model["name"] == "lenet"
            assert config.training["epochs"] == 5
            # But keep default yaml values that weren't overridden
            assert config.dataset["name"] == "mnist"
            assert config.optimization["learning_rate"] == 1e-3
        finally:
            os.unlink(user_yaml.name)

    def test_tier4_cli_overrides_user_yaml(self):
        """Explicit CLI args override values from a user-specified YAML."""
        parser = create_config_parser()
        user_yaml = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        user_yaml.write("""
training:
  epochs: 100
optimization:
  learning_rate: 0.01
""")
        user_yaml.close()
        try:
            raw_args = [
                "--config", user_yaml.name,
                "--epochs", "50",
            ]
            args = parser.parse_args(raw_args)
            config = load_config(config_path=args.config, args=args)
            # CLI --epochs 50 should override YAML's epochs: 100
            assert config.training["epochs"] == 50
            # YAML's learning_rate: 0.01 should survive (no CLI override)
            assert config.optimization["learning_rate"] == 0.01
        finally:
            os.unlink(user_yaml.name)

    def test_tier4_cli_does_not_leak_defaults_into_yaml(self):
        """CLI defaults should NOT override user YAML values."""
        parser = create_config_parser()
        user_yaml = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        user_yaml.write("""
optimization:
  learning_rate: 0.05
  scheduler: cosine
  scheduler_t_max: 200
""")
        user_yaml.close()
        try:
            # No CLI args for learning_rate or scheduler — just --config
            raw_args = ["--config", user_yaml.name]
            args = parser.parse_args(raw_args)
            config = load_config(config_path=args.config, args=args)
            # YAML values must survive
            assert config.optimization["learning_rate"] == 0.05
            assert config.optimization["scheduler"] == "cosine"
            assert config.optimization["scheduler_t_max"] == 200
        finally:
            os.unlink(user_yaml.name)

    def test_all_tiers_combined(self):
        """Full chain: defaults → default yaml → user yaml → CLI."""
        parser = create_config_parser()
        user_yaml = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )
        user_yaml.write("""
training:
  epochs: 50
optimization:
  learning_rate: 0.01
""")
        user_yaml.close()
        try:
            raw_args = [
                "--config", user_yaml.name,
                "--optimizer", "sgd",
                "--momentum", "0.95",
            ]
            args = parser.parse_args(raw_args)
            config = load_config(config_path=args.config, args=args)
            # CLI overrides (--optimizer sgd, --momentum 0.95)
            assert config.optimization["optimizer"] == "sgd"
            assert config.optimization["momentum"] == 0.95
            # User YAML values that weren't CLI-overridden
            assert config.training["epochs"] == 50
            assert config.optimization["learning_rate"] == 0.01
            # Default config.yaml / hardcoded values that weren't overridden
            assert config.dataset["name"] == "mnist"
            assert config.model["name"] == "mynet"
            assert config.checkpointing["save_frequency"] == 10
        finally:
            os.unlink(user_yaml.name)


# ========================================================================
# Type safety: all numeric fields must be the correct Python type
# ========================================================================

class TestConfigTypeSafety:
    """All numeric config fields are the correct Python type from any source."""

    @staticmethod
    def _check_numeric_types(config: Config):
        """Assert all _NUMERIC_SCHEMA fields are the correct type."""
        for dotted_path, expected_types in _NUMERIC_SCHEMA.items():
            parts = dotted_path.split(".")
            cursor = config._config
            for part in parts:
                assert part in cursor, (
                    f"Path '{dotted_path}' not found in config"
                )
                cursor = cursor[part]
            assert isinstance(cursor, expected_types), (
                f"'{dotted_path}' expected {expected_types}, got "
                f"{type(cursor).__name__}: {cursor!r}"
            )

    def test_from_args_defaults_produce_correct_types(self):
        parser = create_config_parser()
        args = parser.parse_args([])
        config = Config.from_args(args)
        self._check_numeric_types(config)

    def test_from_args_custom_produce_correct_types(self):
        parser = create_config_parser()
        args = parser.parse_args([
            "--epochs", "50",
            "--learning-rate", "1e-4",
            "--weight-decay", "0.1",
            "--momentum", "0.95",
            "--batch-size", "128",
        ])
        config = Config.from_args(args)
        self._check_numeric_types(config)

    def test_from_yaml_produce_correct_types(self):
        """Even if YAML has string values, _coerce_numeric_types fixes them."""
        yaml_content = """
training:
  epochs: 50
  batch_size: 32
optimization:
  learning_rate: 1e-3
  weight_decay: 0.01
  momentum: 0.9
  scheduler_step_size: 10
  scheduler_gamma: 0.1
  scheduler_t_max: 100
  scheduler_eta_min: 1e-6
  scheduler_patience: 5
  scheduler_factor: 0.1
  muon_momentum: 0.95
  muon_ns_steps: 5
  adam_lr: 3e-4
  adam_betas: [0.9, 0.95]
checkpointing:
  save_frequency: 10
model:
  num_classes: 10
  input_size: [28, 28]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                         delete=False) as f:
            f.write(yaml_content)
            tmp_path = f.name

        try:
            config = load_config(config_path=tmp_path)
            self._check_numeric_types(config)
        finally:
            os.unlink(tmp_path)

    def test_load_config_without_args_produce_correct_types(self):
        config = load_config()
        self._check_numeric_types(config)

    def test_yaml_with_string_values_coerced_correctly(self):
        """YAML where numeric values are quoted as strings must still coerce."""
        yaml_content = """
optimization:
  learning_rate: "1e-3"
  weight_decay: "0.01"
  momentum: "0.9"
  adam_betas: ["0.9", "0.95"]
training:
  epochs: "50"
  batch_size: "32"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                         delete=False) as f:
            f.write(yaml_content)
            tmp_path = f.name

        try:
            config = load_config(config_path=tmp_path)
            self._check_numeric_types(config)
        finally:
            os.unlink(tmp_path)


# ========================================================================
# to_dict deep copy isolation
# ========================================================================

class TestToDict:
    """to_dict() returns a fully independent copy."""

    def test_mutation_does_not_affect_original(self):
        parser = create_config_parser()
        args = parser.parse_args(["--epochs", "50"])
        config = Config.from_args(args)
        d = config.to_dict()
        d["training"]["epochs"] = 999
        d["training"]["new_key"] = "should_not_leak"
        assert config.training["epochs"] == 50
        assert "new_key" not in config.training

    def test_deeply_nested_independence(self):
        parser = create_config_parser()
        args = parser.parse_args([])
        config = Config.from_args(args)
        d = config.to_dict()
        d["optimization"]["adam_betas"] = [0.0, 0.0]
        assert config.optimization["adam_betas"] == [0.9, 0.95]


# ========================================================================
# Existing tests — updated with tighter assertions
# ========================================================================

class TestDefaultConfig:
    """Default configuration values should be consistent and have correct types."""

    def test_default_values(self):
        cfg = get_default_config()
        assert cfg["dataset"]["name"] == "mnist"
        assert cfg["model"]["name"] == "mynet"
        assert cfg["model"]["num_classes"] == 10
        assert isinstance(cfg["model"]["num_classes"], int)
        assert cfg["model"]["input_channels"] == 1
        assert isinstance(cfg["model"]["input_channels"], int)
        assert cfg["training"]["epochs"] == 20
        assert isinstance(cfg["training"]["epochs"], int)
        assert cfg["training"]["batch_size"] == 64
        assert isinstance(cfg["training"]["batch_size"], int)
        assert cfg["optimization"]["learning_rate"] == 1e-3
        assert isinstance(cfg["optimization"]["learning_rate"], float)
        assert cfg["optimization"]["optimizer"] == "adamw"
        assert cfg["optimization"]["weight_decay"] == 0.01
        assert isinstance(cfg["optimization"]["weight_decay"], float)
        assert cfg["checkpointing"]["save_frequency"] == 10
        assert isinstance(cfg["checkpointing"]["save_frequency"], int)

    def test_newly_added_fields_present(self):
        """Fields added during refactoring must be present."""
        cfg = get_default_config()
        assert "momentum" in cfg["optimization"]
        assert cfg["optimization"]["momentum"] == 0.9
        assert "triplet_margin" in cfg["optimization"]
        assert cfg["optimization"]["triplet_margin"] == 1.0

    def test_load_without_args_returns_default(self):
        config = load_config()
        assert config.dataset["name"] == "mnist"
        assert config.model["name"] == "mynet"


class TestConfigFromArgs:
    """Config from CLI args should correctly populate all fields."""

    def test_minimal_args(self):
        parser = create_config_parser()
        args = parser.parse_args([])  # All defaults
        config = Config.from_args(args)

        assert config.dataset["name"] == "mnist"
        assert config.model["name"] == "mynet"
        assert config.training["epochs"] == 20
        assert config.training["shuffle_train"] is True
        assert config.optimization["learning_rate"] == 1e-3
        assert isinstance(config.optimization["learning_rate"], float)
        assert config.optimization["weight_decay"] == 0.01
        assert isinstance(config.optimization["weight_decay"], float)
        assert config.optimization["momentum"] == 0.9
        assert isinstance(config.optimization["momentum"], float)
        assert config.checkpointing["save_frequency"] == 10

    def test_custom_args(self):
        parser = create_config_parser()
        args = parser.parse_args([
            "--dataset", "cifar10",
            "--model", "alexnet",
            "--epochs", "50",
            "--learning-rate", "1e-4",
            "--optimizer", "sgd",
            "--batch-size", "128",
            "--weight-decay", "0.1",
            "--momentum", "0.95",
        ])
        config = Config.from_args(args)

        assert config.dataset["name"] == "cifar10"
        assert config.model["name"] == "alexnet"
        assert config.training["epochs"] == 50
        assert config.optimization["learning_rate"] == 1e-4
        assert isinstance(config.optimization["learning_rate"], float)
        assert config.optimization["optimizer"] == "sgd"
        assert config.optimization["weight_decay"] == 0.1
        assert isinstance(config.optimization["weight_decay"], float)
        assert config.optimization["momentum"] == 0.95
        assert isinstance(config.optimization["momentum"], float)

    def test_embedding_dim_only_in_model_not_optimization(self):
        """embedding_dim should only appear in model config, not optimization."""
        parser = create_config_parser()
        args = parser.parse_args([])
        config = Config.from_args(args)

        assert "embedding_dim" in config.model, \
            "embedding_dim should be in model config"
        assert "embedding_dim" not in config.optimization, \
            "embedding_dim should NOT be in optimization config"

    def test_device_default_is_auto(self):
        parser = create_config_parser()
        args = parser.parse_args([])
        assert args.device == "auto"


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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                         delete=False) as f:
            f.write(yaml_content)
            tmp_path = f.name

        try:
            config = load_config(config_path=tmp_path)
            assert config.dataset["name"] == "cifar10"
            assert config.model["name"] == "alexnet"
            assert config.training["epochs"] == 5
            assert config.optimization["learning_rate"] == 0.01
            assert isinstance(config.optimization["learning_rate"], float)
            assert config.checkpointing["save_frequency"] == 2
            assert isinstance(config.checkpointing["save_frequency"], int)
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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                         delete=False) as f:
            f.write(yaml_content)
            tmp_path = f.name

        try:
            config = load_config(config_path=tmp_path)
            assert config.dataset["name"] == "subset_631"
            assert config.model["name"] == "fpn_vit"
            assert config.optimization["scheduler"] == "cosine"
            assert config.optimization["scheduler_t_max"] == 100
            assert isinstance(config.optimization["scheduler_t_max"], int)
            assert config.optimization["learning_rate"] == 1e-4
            assert isinstance(config.optimization["learning_rate"], float)
        finally:
            os.unlink(tmp_path)


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
        assert action.default == 10, \
            f"Expected default=10, got {action.default}"

    def test_weight_decay_arg_exists(self):
        parser = create_config_parser()
        action = [a for a in parser._actions if a.dest == "weight_decay"][0]
        assert action.default == 0.01
        assert isinstance(action.default, float)

    def test_momentum_arg_exists(self):
        parser = create_config_parser()
        action = [a for a in parser._actions if a.dest == "momentum"][0]
        assert action.default == 0.9
        assert isinstance(action.default, float)

    def test_device_choices_include_auto(self):
        parser = create_config_parser()
        action = [a for a in parser._actions if a.dest == "device"][0]
        assert "auto" in action.choices
        assert action.default == "auto"
