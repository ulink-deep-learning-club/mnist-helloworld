import os
import argparse
from copy import deepcopy
from typing import Dict, Any, Optional, List

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base, returning a new dict.
    Values in override take precedence over values in base.
    """
    result = deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = deepcopy(val)
    return result


def get_available_datasets() -> List[str]:
    """Get list of available datasets from registry."""
    try:
        from ..datasets import DatasetRegistry

        return DatasetRegistry.list_available()
    except ImportError:
        # Fallback to hardcoded list if import fails
        return [
            "mnist",
            "cifar10",
            "subset_631",
            "subset_1000",
            "triplet_mnist",
            "balanced_triplet_mnist",
        ]


def get_available_models() -> List[str]:
    """Get list of available models from registry."""
    try:
        from ..models import ModelRegistry

        return ModelRegistry.list_available()
    except ImportError:
        # Fallback to hardcoded list if import fails
        return ["lenet", "mynet", "bottleneck_vit", "fpn_vit", "siamese"]


class Config:
    """Configuration management."""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        self.dataset = config_dict.get("dataset", {})
        self.model = config_dict.get("model", {})
        self.training = config_dict.get("training", {})
        self.optimization = config_dict.get("optimization", {})
        self.checkpointing = config_dict.get("checkpointing", {})
        self.annealing = config_dict.get("annealing")  # None → not enabled, {} → enabled over all epochs

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        config_dict = _load_yaml_file(yaml_path)
        return cls(config_dict)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """Create configuration from command line arguments."""
        config_dict = {
            "dataset": {"name": args.dataset, "root": args.data_root, "download": True},
            "model": {
                "name": args.model,
                "num_classes": args.num_classes,
                "input_size": [args.input_size, args.input_size],
                "embedding_dim": getattr(args, "embedding_dim", 256),
            },
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "shuffle_train": True,
            },
            "optimization": {
                "learning_rate": args.learning_rate,
                "optimizer": args.optimizer,
                "weight_decay": args.weight_decay,
                "momentum": args.momentum,
                "scheduler": args.scheduler,
                "scheduler_step_size": args.scheduler_step_size,
                "scheduler_gamma": args.scheduler_gamma,
                "scheduler_t_max": args.scheduler_t_max,
                "scheduler_eta_min": args.scheduler_eta_min,
                "scheduler_patience": args.scheduler_patience,
                "scheduler_factor": args.scheduler_factor,
                "triplet_margin": getattr(args, "triplet_margin", 1.0),
                "muon_momentum": args.muon_momentum,
                "muon_ns_steps": args.muon_ns_steps,
                "adam_lr": args.adam_lr,
                "adam_betas": args.adam_betas,
            },
            "checkpointing": {
                "checkpoint_dir": args.checkpoint_dir,
                "save_frequency": args.save_frequency,
            },
        }

        # Handle --annealing CLI flag
        annealing_val = getattr(args, "annealing", None)
        if annealing_val is not None:
            if annealing_val != -1:
                config_dict["annealing"] = {"epochs": annealing_val}
            else:
                config_dict["annealing"] = {}

        return cls(config_dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (deep copy, safe to mutate)."""
        return deepcopy(self._config)


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        "dataset": {"name": "mnist", "root": "./data", "download": True},
        "model": {
            "name": "mynet",
            "num_classes": 10,
            "input_channels": 1,
            "input_size": [28, 28],
        },
        "training": {
            "epochs": 20,
            "batch_size": 64,
            "num_workers": 4,
            "shuffle_train": True,
        },
        "optimization": {
            "learning_rate": 1e-3,
            "optimizer": "adamw",
            "weight_decay": 0.01,
            "scheduler": "none",
            "scheduler_step_size": 10,
            "scheduler_gamma": 0.1,
            "scheduler_t_max": 100,
            "scheduler_eta_min": 1e-6,
            "scheduler_patience": 5,
            "scheduler_factor": 0.1,
            "momentum": 0.9,
            "triplet_margin": 1.0,
            "muon_momentum": 0.95,
            "muon_ns_steps": 5,
            "adam_lr": 3e-4,
            "adam_betas": [0.9, 0.95],
        },
        "checkpointing": {"checkpoint_dir": "checkpoints", "save_frequency": 10},
    }


def create_config_parser() -> argparse.ArgumentParser:
    """Create argument parser for configuration."""
    parser = argparse.ArgumentParser(description="Train a neural network")

    # Get available choices dynamically
    available_datasets = get_available_datasets()
    available_models = get_available_models()

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=available_datasets,
        help=f"Dataset to use. Available: {', '.join(available_datasets)}",
    )
    parser.add_argument(
        "--data-root", type=str, default="./data", help="Root directory for datasets"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="mynet",
        choices=available_models,
        help=f"Model architecture. Available: {', '.join(available_models)}",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use for training (auto, cuda, cpu, mps)",
    )
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes")
    parser.add_argument(
        "--input-size",
        type=int,
        default=28,
        help="Input image size (height and width, will be squared)",
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )

    # Optimization arguments
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum (used by SGD optimizer)"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adam", "sgd", "muon", "muon_with_aux_adam"],
        help="Optimizer",
    )
    parser.add_argument(
        "--muon-momentum",
        type=float,
        default=0.95,
        help="Muon momentum parameter (only used for muon optimizers)",
    )
    parser.add_argument(
        "--muon-ns-steps",
        type=int,
        default=5,
        help="Newton-Schulz iteration steps for Muon (only used for muon optimizers)",
    )
    parser.add_argument(
        "--adam-lr",
        type=float,
        default=3e-4,
        help="Learning rate for Adam parts in MuonWithAuxAdam",
    )
    parser.add_argument(
        "--adam-betas",
        type=float,
        nargs=2,
        default=[0.9, 0.95],
        help="Adam betas for MuonWithAuxAdam (two values)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "step", "cosine", "plateau", "exponential"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--scheduler-step-size",
        type=int,
        default=10,
        help="Step size for StepLR scheduler",
    )
    parser.add_argument(
        "--scheduler-gamma",
        type=float,
        default=0.1,
        help="Gamma for learning rate decay",
    )
    parser.add_argument(
        "--scheduler-t-max",
        type=int,
        default=100,
        help="T_max for CosineAnnealingLR",
    )
    parser.add_argument(
        "--scheduler-eta-min",
        type=float,
        default=1e-6,
        help="Eta_min for CosineAnnealingLR",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=5,
        help="Patience for ReduceLROnPlateau",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.1,
        help="Factor for ReduceLROnPlateau",
    )
    parser.add_argument(
        "--triplet-margin",
        type=float,
        default=1.0,
        help="Margin for triplet loss (used by siamese models)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Embedding dimension for siamese/metric learning models",
    )

    # Checkpointing arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=10, help="Save frequency in epochs"
    )

    # Configuration file
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        metavar="EXP",
        help="Resume training from a specific experiment (e.g., exp1, exp2)",
    )

    # Fork training
    parser.add_argument(
        "--fork",
        type=str,
        metavar="EXP",
        help="Fork training from a specific experiment - loads checkpoint but saves to a new experiment",
    )

    # Layer freezing
    parser.add_argument(
        "--freeze",
        type=str,
        nargs="+",
        help=(
            "Freeze specific layers. Supports: "
            "(1) Layer IDs: --freeze 2-1 2-2; "
            "(2) ID ranges: --freeze 2-1:2-5; "
            "(3) Name patterns: --freeze features classifier"
        ),
    )

    # Data augmentation
    parser.add_argument(
        "--reapply-transforms",
        action="store_true",
        help="Reapply random transforms after each epoch for better generalization",
    )

    # Early stopping
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience (0 to disable). Stops training if no improvement after N epochs.",
    )

    # Parameter annealing
    parser.add_argument(
        "--annealing",
        nargs="?",
        const=-1,
        default=None,
        type=int,
        metavar="EPOCHS",
        help="Enable parameter annealing (tau 0→1 for model.on_annealing_step). "
             "Optionally specify number of epochs to anneal over, e.g. --annealing 20",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (torch, numpy, random)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic algorithms in PyTorch (may reduce performance)",
    )

    # Mixed precision training
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision training (FP16) for faster training on compatible GPUs",
    )

    # torch.compile
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for model optimization (requires PyTorch 2.x)",
    )

    return parser


_PARSER_DEFAULTS_CACHE: Optional[Dict[str, Any]] = None


def _get_parser_defaults() -> Dict[str, Any]:
    """Get the default values from the argument parser (cached)."""
    global _PARSER_DEFAULTS_CACHE
    if _PARSER_DEFAULTS_CACHE is None:
        parser = create_config_parser()
        _PARSER_DEFAULTS_CACHE = vars(parser.parse_args([]))
    return _PARSER_DEFAULTS_CACHE


def _build_user_cli_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """Build a sparse config dict containing only the CLI args that the user
    explicitly provided (i.e. differ from their default values).

    This ensures YAML config values are preserved for anything the user
    didn't explicitly specify on the command line.
    """
    defaults = _get_parser_defaults()
    # Determine which args the user explicitly provided (differ from default)
    explicit_args = {
        k: v for k, v in vars(args).items()
        if k not in defaults or v != defaults[k]
    }

    if not explicit_args:
        return {}

    # Build two full config dicts: one with defaults, one with user overrides
    default_ns = argparse.Namespace(**defaults)
    merged_ns = argparse.Namespace(**{**defaults, **explicit_args})

    default_config = Config.from_args(default_ns)._config
    merged_config = Config.from_args(merged_ns)._config

    # Recursively extract only the fields that differ from defaults
    def _diff_dict(base: dict, override: dict) -> dict:
        result = {}
        for key, val in override.items():
            if key not in base:
                result[key] = deepcopy(val)
            elif isinstance(val, dict) and isinstance(base[key], dict):
                nested = _diff_dict(base[key], val)
                if nested:
                    result[key] = nested
            elif val != base[key]:
                result[key] = deepcopy(val)
        return result

    return _diff_dict(default_config, merged_config)


def _load_yaml_file(path: str) -> dict:
    """Load a YAML file and return its contents as a dict."""
    if not YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required for YAML configuration files. Install with: pip install pyyaml"
        )
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Schema describing which config fields should be numeric and their target types.
# After merging YAML config, any string values in these paths are coerced.
_NUMERIC_SCHEMA: Dict[str, type | tuple[type, ...]] = {
    # training section
    "training.epochs": int,
    "training.batch_size": int,
    "training.num_workers": int,
    # optimization section
    "optimization.learning_rate": (float, int),
    "optimization.weight_decay": (float, int),
    "optimization.momentum": (float, int),
    "optimization.scheduler_step_size": int,
    "optimization.scheduler_gamma": (float, int),
    "optimization.scheduler_t_max": int,
    "optimization.scheduler_eta_min": (float, int),
    "optimization.scheduler_patience": int,
    "optimization.scheduler_factor": (float, int),
    "optimization.triplet_margin": (float, int),
    "optimization.muon_momentum": (float, int),
    "optimization.muon_ns_steps": int,
    "optimization.adam_lr": (float, int),
    "optimization.adam_betas": list,
    # checkpointing section
    "checkpointing.save_frequency": int,
    # model section
    "model.num_classes": int,
    "model.input_size": list,
}


def _coerce_numeric_types(d: dict) -> dict:
    """Convert string values in known numeric fields to the correct type.
    This is a safety net for YAML parsing quirks (e.g., "1e-3" parsed as string).
    """
    result = deepcopy(d)
    for dotted_path, target_type in _NUMERIC_SCHEMA.items():
        parts = dotted_path.split(".")
        parent = result
        for part in parts[:-1]:
            if part not in parent or not isinstance(parent[part], dict):
                parent = None
                break
            parent = parent[part]
        if parent is None:
            continue
        key = parts[-1]
        if key not in parent:
            continue
        val = parent[key]

        # --- List-type fields (adam_betas, input_size) ---
        if target_type is list:
            if isinstance(val, list):
                # Convert any string elements to float
                converted = [
                    float(item) if isinstance(item, str) else item
                    for item in val
                ]
                if converted != val:
                    parent[key] = converted
            # If val is not a list at all, leave it alone
            continue

        # Already the right type — skip
        if isinstance(val, target_type):
            continue

        # String that should be numeric — convert
        if isinstance(val, str):
            try:
                if target_type in (float, (float, int)):
                    parent[key] = float(val)
                elif target_type is int:
                    parent[key] = int(val)
            except (ValueError, TypeError):
                pass  # leave as-is, will fail downstream
    return result


def load_config(
    config_path: Optional[str] = None, args: Optional[argparse.Namespace] = None
) -> Config:
    """Load configuration with the following priority (highest wins):

    1. Hardcoded defaults (lowest priority)
    2. Default config.yaml at project root
    3. User-specified --config file
    4. Explicit CLI arguments (highest priority)

    Examples::

        python train.py                                    # defaults + config.yaml       # noqa
        python train.py --config myconfig.yaml              # defaults + config.yaml + myconfig.yaml  # noqa
        python train.py --epochs 50                         # defaults + config.yaml + CLI override  # noqa
        python train.py --config myconfig.yaml --epochs 50  # ... + myconfig.yaml + CLI override  # noqa
    """
    # Tier 1: hardcoded defaults
    config_dict = get_default_config()

    # Tier 2: default config.yaml at project root (if exists)
    default_yaml_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
    if os.path.exists(default_yaml_path):
        try:
            yaml_dict = _load_yaml_file(default_yaml_path)
            if yaml_dict:
                config_dict = _deep_merge(config_dict, yaml_dict)
        except Exception as e:
            # Warn but don't crash if default config.yaml is malformed
            import warnings
            warnings.warn(f"Failed to load default config.yaml: {e}")

    # Tier 3: user-specified --config file (if provided)
    if config_path:
        yaml_dict = _load_yaml_file(config_path)
        if yaml_dict:
            config_dict = _deep_merge(config_dict, yaml_dict)

    # Tier 4: CLI arguments (only non-default values override)
    if args:
        cli_overrides = _build_user_cli_dict(args)
        if cli_overrides:
            config_dict = _deep_merge(config_dict, cli_overrides)

    # Safety: coerce known numeric fields to correct types
    # (handles edge cases where YAML parses "1e-3" as a string)
    config_dict = _coerce_numeric_types(config_dict)

    return Config(config_dict)
