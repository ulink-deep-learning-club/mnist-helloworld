"""Tests for AnnealingManager and its integration with config."""

import pytest
from src.training.annealing import AnnealingManager
from src.config import Config, load_config, create_config_parser


# ========================================================================
# Unit: AnnealingManager.get_tau
# ========================================================================

class TestGetTau:
    """get_tau returns linear values in [0, 1]."""

    def test_tau_starts_at_zero(self):
        mgr = AnnealingManager(epochs=10)
        assert mgr.get_tau(0, total_epochs=10) == 0.0

    def test_tau_reaches_one_at_anneal_epochs(self):
        mgr = AnnealingManager(epochs=10)
        assert mgr.get_tau(10, total_epochs=10) == 1.0

    def test_tau_clamps_at_one(self):
        mgr = AnnealingManager(epochs=10)
        assert mgr.get_tau(15, total_epochs=10) == 1.0

    def test_tau_linear_halfway(self):
        mgr = AnnealingManager(epochs=10)
        assert mgr.get_tau(5, total_epochs=10) == 0.5

    def test_tau_linear_progress(self):
        mgr = AnnealingManager(epochs=10)
        assert mgr.get_tau(3, total_epochs=10) == 0.3

    def test_epochs_none_uses_total_epochs(self):
        """When epochs is None, tau uses total_epochs as the window."""
        mgr = AnnealingManager(epochs=None)
        assert mgr.get_tau(0, total_epochs=20) == 0.0
        assert mgr.get_tau(10, total_epochs=20) == 0.5
        assert mgr.get_tau(20, total_epochs=20) == 1.0
        assert mgr.get_tau(25, total_epochs=20) == 1.0

    def test_no_epochs_constructor(self):
        """Default constructor sets epochs=None."""
        mgr = AnnealingManager()
        assert mgr.epochs is None
        assert mgr.get_tau(5, total_epochs=10) == 0.5


# ========================================================================
# Unit: AnnealingManager.from_config
# ========================================================================

class TestFromConfig:
    """from_config correctly parses config input."""

    def test_none_returns_none(self):
        assert AnnealingManager.from_config(None) is None

    def test_empty_dict_returns_manager_with_epochs_none(self):
        """{} means anneal over all epochs."""
        mgr = AnnealingManager.from_config({})
        assert mgr is not None
        assert mgr.epochs is None

    def test_dict_with_epochs(self):
        mgr = AnnealingManager.from_config({"epochs": 20})
        assert mgr is not None
        assert mgr.epochs == 20

    def test_dict_without_epochs_key(self):
        mgr = AnnealingManager.from_config({"other": "value"})
        assert mgr is not None
        assert mgr.epochs is None

    def test_int_value(self):
        mgr = AnnealingManager.from_config(20)
        assert mgr is not None
        assert mgr.epochs == 20

    def test_float_value(self):
        mgr = AnnealingManager.from_config(20.0)
        assert mgr is not None
        assert mgr.epochs == 20

    def test_epochs_zero_raises(self):
        with pytest.raises(ValueError, match="> 0"):
            AnnealingManager.from_config({"epochs": 0})
        with pytest.raises(ValueError, match="> 0"):
            AnnealingManager.from_config(0)

    def test_epochs_negative_raises(self):
        with pytest.raises(ValueError, match="> 0"):
            AnnealingManager.from_config({"epochs": -1})

    def test_bool_true_returns_manager(self):
        mgr = AnnealingManager.from_config(True)
        assert mgr is not None
        assert mgr.epochs is None

    def test_bool_false_returns_none(self):
        assert AnnealingManager.from_config(False) is None

    def test_unexpected_type_raises(self):
        with pytest.raises(TypeError, match="Unexpected"):
            AnnealingManager.from_config("string")


# ========================================================================
# Integration: config → AnnealingManager
# ========================================================================

class TestConfigToAnnealing:
    """CLI args and YAML config correctly produce AnnealingManager."""

    def test_no_annealing_arg_means_disabled(self):
        """Without --annealing, from_config should get None and return None."""
        parser = create_config_parser()
        args = parser.parse_args([])  # no --annealing
        config = Config.from_args(args)
        assert config.annealing is None
        assert AnnealingManager.from_config(config.annealing) is None

    def test_annealing_without_value_means_all_epochs(self):
        """--annealing (no value) should enable annealing over all epochs."""
        parser = create_config_parser()
        args = parser.parse_args(["--annealing"])
        config = Config.from_args(args)
        assert config.annealing == {}
        mgr = AnnealingManager.from_config(config.annealing)
        assert mgr is not None
        assert mgr.epochs is None  # all epochs

    def test_annealing_with_value_sets_epochs(self):
        parser = create_config_parser()
        args = parser.parse_args(["--annealing", "20"])
        config = Config.from_args(args)
        assert config.annealing == {"epochs": 20}
        mgr = AnnealingManager.from_config(config.annealing)
        assert mgr is not None
        assert mgr.epochs == 20

    def test_load_config_preserves_annealing_disabled(self):
        """load_config() without --annealing should leave annealing as None."""
        config = load_config()
        assert AnnealingManager.from_config(config.annealing) is None

    def test_yaml_annealing_with_epochs(self):
        """YAML with annealing.epochs should be parsed correctly."""
        import tempfile, os
        yaml_content = """
annealing:
  epochs: 15
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                         delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            config = load_config(config_path=path)
            mgr = AnnealingManager.from_config(config.annealing)
            assert mgr is not None
            assert mgr.epochs == 15
        finally:
            os.unlink(path)

    def test_yaml_empty_annealing_dict(self):
        """YAML with 'annealing: {}' enables over all epochs."""
        import tempfile, os
        yaml_content = """
annealing: {}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                         delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            config = load_config(config_path=path)
            mgr = AnnealingManager.from_config(config.annealing)
            assert mgr is not None
            assert mgr.epochs is None  # all epochs
        finally:
            os.unlink(path)

    def test_yaml_null_annealing_block(self):
        """YAML with 'annealing:' (null) leaves annealing as None (disabled)."""
        import tempfile, os
        yaml_content = """
annealing:
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                         delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            config = load_config(config_path=path)
            assert AnnealingManager.from_config(config.annealing) is None
        finally:
            os.unlink(path)

    def test_annealing_disabled_in_yaml(self):
        """YAML without annealing section leaves annealing as None."""
        import tempfile, os
        yaml_content = """
model:
  name: lenet
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                         delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            config = load_config(config_path=path)
            # No annealing section in YAML, and no --annealing CLI arg
            assert AnnealingManager.from_config(config.annealing) is None
        finally:
            os.unlink(path)
