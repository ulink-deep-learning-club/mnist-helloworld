#!/usr/bin/env python3
"""
Modular Training Script for Neural Networks
Supports multiple datasets and model architectures
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

import yaml
from src.datasets import DatasetRegistry
from src.models import ModelRegistry
from src.training import Trainer, CheckpointManager, ExperimentManager
from src.config import create_config_parser, load_config
from src.utils import get_device, get_optimal_workers, setup_logger


def get_layer_id_mapping(model):
    """Generate a mapping from torchinfo-style layer IDs to module names.

    Returns a dict mapping "depth-idx" strings to module names.
    """
    id_to_name = {}
    depth_counters = {}

    def traverse_modules(module, prefix="", depth=1):
        if depth not in depth_counters:
            depth_counters[depth] = 0

        for name, child in module.named_children():
            depth_counters[depth] += 1
            idx = depth_counters[depth]
            layer_id = f"{depth}-{idx}"
            full_name = f"{prefix}.{name}" if prefix else name
            id_to_name[layer_id] = full_name

            # Recurse into children
            traverse_modules(child, full_name, depth + 1)

    traverse_modules(model)
    return id_to_name


def parse_freeze_spec(spec):
    """Parse a freeze specification.

    Supports:
    - Single layer ID: "2-1"
    - Range: "2-1:2-5" (freeze layers 2-1 through 2-5)
    - Name pattern: "features" (starts with)
    """
    if ":" in spec:
        # Range specification
        start, end = spec.split(":")
        return ("range", start.strip(), end.strip())
    elif "-" in spec and spec.split("-")[0].isdigit():
        # Single layer ID
        return ("id", spec.strip())
    else:
        # Name pattern
        return ("name", spec.strip())


def freeze_layers(model, freeze_specs, id_to_name=None):
    """Freeze model layers based on specifications.

    Args:
        model: The neural network model
        freeze_specs: List of freeze specifications (layer IDs, ranges, or name patterns)
        id_to_name: Optional mapping from layer IDs to module names

    Returns:
        Tuple of (frozen_count, modules_to_freeze)
    """
    if not freeze_specs:
        return 0, set()

    # Build ID mapping if not provided
    if id_to_name is None:
        id_to_name = get_layer_id_mapping(model)

    # Collect all module names to freeze
    modules_to_freeze = set()

    for spec in freeze_specs:
        spec_type, *rest = parse_freeze_spec(spec)

        if spec_type == "id":
            # Single layer ID
            layer_id = rest[0]
            if layer_id in id_to_name:
                modules_to_freeze.add(id_to_name[layer_id])
            else:
                print(f"Warning: Layer ID '{layer_id}' not found")

        elif spec_type == "range":
            # Range of layer IDs (same depth)
            start_id, end_id = rest
            start_depth, start_idx = map(int, start_id.split("-"))
            end_depth, end_idx = map(int, end_id.split("-"))

            if start_depth != end_depth:
                print(
                    f"Warning: Range {start_id}:{end_id} has different depths, using start depth"
                )

            depth = start_depth
            for idx in range(start_idx, end_idx + 1):
                layer_id = f"{depth}-{idx}"
                if layer_id in id_to_name:
                    modules_to_freeze.add(id_to_name[layer_id])

        elif spec_type == "name":
            # Name pattern
            pattern = rest[0]
            for layer_id, module_name in id_to_name.items():
                if module_name.startswith(pattern):
                    modules_to_freeze.add(module_name)

    # Freeze parameters in the selected modules
    frozen_count = 0
    frozen_modules = set()

    for name, param in model.named_parameters():
        for module_name in modules_to_freeze:
            if name.startswith(module_name):
                param.requires_grad = False
                frozen_count += 1
                frozen_modules.add(module_name)
                break

    return frozen_count, modules_to_freeze


def create_optimizer(model, config):
    """Create optimizer based on configuration."""
    optimizer_name = config.optimization.get("optimizer", "adamw").lower()
    learning_rate = config.optimization.get("learning_rate", 1e-3)
    weight_decay = config.optimization.get("weight_decay", 0.01)

    # Only pass trainable parameters to optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_name == "adamw":
        return optim.AdamW(
            trainable_params, lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "adam":
        return optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        momentum = config.optimization.get("momentum", 0.9)
        return optim.SGD(
            trainable_params,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def main():
    """Main training function."""
    # Parse arguments
    parser = create_config_parser()
    args = parser.parse_args()

    # Load configuration
    config = load_config(config_path=args.config, args=args)

    # Set up logging
    logger = setup_logger()
    logger.info("Starting modular training script")

    # Get device
    device, using_cpu = get_device()
    logger.info(f"Using device: {device}")

    # Get optimal workers
    train_workers, val_workers = get_optimal_workers(using_cpu)
    logger.info(
        f"Using {train_workers} workers for training, {val_workers} for validation"
    )

    # Create experiment manager (YOLO-style runs/expX)
    experiment_manager = ExperimentManager(base_dir="runs", resume_exp=args.resume)
    logger.info(experiment_manager.get_experiment_info())

    # Check if resuming
    resume_checkpoint = None
    start_epoch = 0
    if args.resume:
        checkpoint_path = os.path.join(
            experiment_manager.checkpoints_dir, "latest_checkpoint.pt"
        )
        if os.path.exists(checkpoint_path):
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            resume_checkpoint = checkpoint_path
        else:
            logger.warning(
                f"No checkpoint found at {checkpoint_path}, starting from scratch"
            )

    # Save config to experiment directory (if not resuming)
    # Note: Config is saved after dataset creation to get actual values

    # Create dataset
    logger.info(f"Creating dataset: {config.dataset['name']}")
    dataset = DatasetRegistry.create(
        config.dataset["name"],
        root=config.dataset["root"],
        download=config.dataset["download"],
    )

    # Get data loaders
    train_loader, val_loader = dataset.get_dataloaders(
        batch_size=config.training["batch_size"],
        num_workers=train_workers,
        shuffle_train=config.training.get("shuffle_train", True),
    )

    # Save config with actual dataset values (always update, even when resuming)
    config_dict = config.to_dict()
    config_dict["dataset"]["name"] = config.dataset["name"]
    config_dict["model"]["num_classes"] = dataset.num_classes
    config_dict["model"]["input_channels"] = dataset.input_channels
    config_dict["model"]["input_size"] = dataset.input_size
    with open(experiment_manager.config_file, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # Create checkpoint manager first (needed for resume)
    save_frequency = config.checkpointing.get("save_frequency", 10)
    checkpoint_manager = CheckpointManager(
        checkpoints_dir=experiment_manager.checkpoints_dir,
        save_frequency=save_frequency,
    )

    # Load checkpoint config if resuming
    model_config = None
    if resume_checkpoint:
        checkpoint_data = torch.load(resume_checkpoint, map_location="cpu")
        model_config = checkpoint_data.get("model_config", {})
        logger.info(f"Loaded model config from checkpoint: {model_config}")

    # Create model
    logger.info(f"Creating model: {config.model['name']}")

    # Use config from checkpoint if available, otherwise use current settings
    # Note: dataset properties are source of truth for input data characteristics
    if model_config and "num_classes" in model_config:
        # Resume with same architecture as checkpoint
        num_classes = model_config.get("num_classes", dataset.num_classes)
        # Always use dataset's actual input channels and size (source of truth)
        input_channels = dataset.input_channels
        input_size = dataset.input_size
        logger.info(
            f"Using model config from checkpoint: classes={num_classes}, channels={input_channels}, size={input_size}"
        )
    else:
        num_classes = dataset.num_classes
        input_channels = dataset.input_channels
        input_size = dataset.input_size

    model = ModelRegistry.create(
        config.model["name"],
        num_classes=num_classes,
        input_channels=input_channels,
        input_size=input_size,
    ).to(device)

    # Get layer ID mapping for freeze functionality
    id_to_name = get_layer_id_mapping(model)

    # Freeze layers if specified
    if args.freeze:
        frozen_count, frozen_modules = freeze_layers(model, args.freeze, id_to_name)
        logger.info(
            f"Frozen {frozen_count} parameter tensors in modules: {sorted(frozen_modules)}"
        )

    # Print model info
    model_info = model.get_model_info()
    info_str = ""
    padding_num = max(map(lambda x: len(x), [k for k in model_info.keys()]))
    for k, v in model_info.items():
        info_str += f"\n  {k.rjust(padding_num)}: {v}"
    logger.info(f"Model info: {info_str}")
    summary(model)

    # Create criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config)

    # Create trainer first
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        experiment_manager=experiment_manager,
        checkpoint_manager=checkpoint_manager,
    )

    # Load history and get start epoch if resuming
    if resume_checkpoint:
        start_epoch = trainer.load_history_from_log()
        logger.info(f"Loaded history from {start_epoch} epochs")

        # Load checkpoint weights (non-strict to allow partial loading like YOLO)
        try:
            checkpoint_info = checkpoint_manager.load_checkpoint(
                resume_checkpoint, model, optimizer, strict=False
            )
            # Use epoch from log file as it's more reliable
            checkpoint_manager.best_accuracy = checkpoint_info["accuracy"]
            logger.info(
                f"Resumed from epoch {start_epoch}, best accuracy: {checkpoint_info['accuracy']:.2f}%"
            )
        except RuntimeError as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.error(
                "The checkpoint may be incompatible with the current model architecture."
            )
            logger.error(
                "Please check if the model code has changed since the checkpoint was created."
            )
            raise

    # Start training
    logger.info(f"Starting training for {config.training['epochs']} epochs")

    try:
        results = trainer.train(
            epochs=config.training["epochs"], start_epoch=start_epoch
        )

        # Print summary
        logger.info(f"\nTraining completed!")
        logger.info(f"Experiment directory: {results['experiment_dir']}")
        logger.info(f"Epochs trained: {results['epochs_trained']}")
        logger.info(f"Best validation accuracy: {results['best_accuracy']:.2f}%")
        logger.info(f"Training time: {results['training_time']:.2f} seconds")

        # Save final model
        final_model_path = experiment_manager.checkpoints_dir + "/final_model.pth"
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Final model saved as {final_model_path}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

        # Save interrupted model
        interrupted_path = experiment_manager.checkpoints_dir + "/interrupted_model.pth"
        torch.save(model.state_dict(), interrupted_path)
        logger.info(f"Interrupted model saved as {interrupted_path}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
