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


def create_optimizer(model, config):
    """Create optimizer based on configuration."""
    optimizer_name = config.optimization.get("optimizer", "adamw").lower()
    learning_rate = config.optimization.get("learning_rate", 1e-3)
    weight_decay = config.optimization.get("weight_decay", 0.01)

    if optimizer_name == "adamw":
        return optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "adam":
        return optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "sgd":
        momentum = config.optimization.get("momentum", 0.9)
        return optim.SGD(
            model.parameters(),
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

    # Save config with actual dataset values (if not resuming)
    if not args.resume:
        config_dict = config.to_dict()
        config_dict["dataset"]["name"] = config.dataset["name"]
        config_dict["model"]["num_classes"] = dataset.num_classes
        config_dict["model"]["input_channels"] = dataset.input_channels
        config_dict["model"]["input_size"] = dataset.input_size
        with open(experiment_manager.config_file, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    # Create checkpoint manager first (needed for resume)
    checkpoint_manager = CheckpointManager(
        checkpoints_dir=experiment_manager.checkpoints_dir
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
