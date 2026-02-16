import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os
from typing import Optional, Dict, Any, List
from .metrics import MetricsTracker
from .checkpoint import CheckpointManager
from .experiment import ExperimentManager


class Trainer:
    """Modular training framework with experiment tracking."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        experiment_manager: ExperimentManager,
        checkpoint_manager: Optional[CheckpointManager] = None,
        scheduler: Optional[Any] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.experiment_manager = experiment_manager
        self.checkpoint_manager = checkpoint_manager
        self.scheduler = scheduler

        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()

        # Training history for plotting
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
            "epoch_time": [],
            "train_speed": [],
            "val_speed": [],
        }

        # Initialize log file only if it doesn't exist
        if not os.path.exists(self.experiment_manager.log_file):
            self._init_log_file()

    def _init_log_file(self):
        """Initialize training log file with headers."""
        log_path = self.experiment_manager.log_file
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            f.write(
                "epoch, train_loss, train_acc, val_loss, val_acc, lr, epoch_time(s), train_it/s, val_it/s\n"
            )

    def load_history_from_log(self) -> int:
        """Load training history from existing log file. Returns last epoch number."""
        log_path = self.experiment_manager.log_file
        if not os.path.exists(log_path):
            return 0

        last_epoch = 0
        with open(log_path, "r") as f:
            lines = f.readlines()
            # Skip header
            for line in lines[1:]:
                parts = line.strip().split(",")
                if len(parts) >= 9:
                    try:
                        self.history["train_loss"].append(float(parts[1].strip()))
                        self.history["train_accuracy"].append(float(parts[2].strip()))
                        self.history["val_loss"].append(float(parts[3].strip()))
                        self.history["val_accuracy"].append(float(parts[4].strip()))
                        self.history["learning_rate"].append(float(parts[5].strip()))
                        self.history["epoch_time"].append(
                            float(parts[6].strip().replace("s", ""))
                        )
                        self.history["train_speed"].append(float(parts[7].strip()))
                        self.history["val_speed"].append(float(parts[8].strip()))
                        last_epoch = int(parts[0].strip())
                    except (ValueError, IndexError):
                        continue
        return last_epoch

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict,
        lr: float,
        epoch_time: float,
        train_speed: float,
        val_speed: float,
    ):
        """Log epoch results to file."""
        log_path = self.experiment_manager.log_file
        with open(log_path, "a") as f:
            f.write(
                f"{epoch:4d}, "
                f"{train_metrics['loss']:8.4f}, "
                f"{train_metrics['accuracy']:6.2f}, "
                f"{val_metrics['loss']:8.4f}, "
                f"{val_metrics['accuracy']:6.2f}, "
                f"{lr:.2e}, "
                f"{epoch_time:6.1f}s, "
                f"{train_speed:6.2f}, "
                f"{val_speed:6.2f}\n"
            )

    def train_epoch(self, epoch: int) -> tuple[Dict[str, float], float]:
        """Train for one epoch. Returns metrics and speed (it/s)."""
        self.model.train()
        self.train_metrics.reset()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} - Training")
        start_time = time.time()
        num_batches = len(self.train_loader)

        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            self.train_metrics.update(loss.item(), outputs, labels)

            # Update progress bar
            metrics = self.train_metrics.get_metrics()
            pbar.set_postfix(
                {"Loss": f"{metrics['loss']:.4f}", "Acc": f"{metrics['accuracy']:.2f}%"}
            )

        elapsed_time = time.time() - start_time
        speed = num_batches / elapsed_time if elapsed_time > 0 else 0

        return self.train_metrics.get_metrics(), speed

    def validate(self) -> tuple[Dict[str, float], float]:
        """Validate the model. Returns metrics and speed (it/s)."""
        self.model.eval()
        self.val_metrics.reset()

        pbar = tqdm(self.val_loader, desc="Validating")
        start_time = time.time()
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                self.val_metrics.update(loss.item(), outputs, labels)

                # Update progress bar
                metrics = self.val_metrics.get_metrics()
                pbar.set_postfix(
                    {
                        "Loss": f"{metrics['loss']:.4f}",
                        "Acc": f"{metrics['accuracy']:.2f}%",
                    }
                )

        elapsed_time = time.time() - start_time
        speed = num_batches / elapsed_time if elapsed_time > 0 else 0

        return self.val_metrics.get_metrics(), speed

    def _get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def _plot_training_curves(self):
        """Plot and save training curves."""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            epochs = range(1, len(self.history["train_loss"]) + 1)

            # Plot 1: Loss curves
            ax = axes[0, 0]
            ax.plot(
                epochs,
                self.history["train_loss"],
                "b-",
                label="Train Loss",
                linewidth=2,
            )
            ax.plot(
                epochs, self.history["val_loss"], "r-", label="Val Loss", linewidth=2
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training and Validation Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 2: Accuracy curves
            ax = axes[0, 1]
            ax.plot(
                epochs,
                self.history["train_accuracy"],
                "b-",
                label="Train Acc",
                linewidth=2,
            )
            ax.plot(
                epochs, self.history["val_accuracy"], "r-", label="Val Acc", linewidth=2
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Training and Validation Accuracy")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 3: Learning rate
            ax = axes[1, 0]
            ax.plot(epochs, self.history["learning_rate"], "g-", linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Learning Rate")
            ax.set_title("Learning Rate Schedule")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)

            # Plot 4: Training speed
            ax = axes[1, 1]
            ax.plot(
                epochs, self.history["train_speed"], "b-", label="Train", linewidth=2
            )
            ax.plot(epochs, self.history["val_speed"], "r-", label="Val", linewidth=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Iterations/second")
            ax.set_title("Training Speed")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if os.path.exists(self.experiment_manager.plot_file):
                os.remove(self.experiment_manager.plot_file)

            plt.savefig(self.experiment_manager.plot_file, dpi=150)
            plt.close()

        except ImportError:
            print("Warning: matplotlib not available, skipping plot generation")

    def train(self, epochs: int, start_epoch: int = 0) -> Dict[str, Any]:
        """Train the model for specified epochs."""
        print(f"Starting training for {epochs} epochs")
        print(f"Experiment directory: {self.experiment_manager.experiment_dir}")
        start_time = time.time()

        best_accuracy = 0.0

        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            current_lr = self._get_current_lr()

            # Train
            train_metrics, train_speed = self.train_epoch(epoch)

            # Validate
            val_metrics, val_speed = self.validate()

            epoch_time = time.time() - epoch_start_time

            # Update history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_accuracy"].append(train_metrics["accuracy"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["learning_rate"].append(current_lr)
            self.history["epoch_time"].append(epoch_time)
            self.history["train_speed"].append(train_speed)
            self.history["val_speed"].append(val_speed)

            # Log to file
            self._log_epoch(
                epoch + 1,
                train_metrics,
                val_metrics,
                current_lr,
                epoch_time,
                train_speed,
                val_speed,
            )

            # Print epoch summary
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}% - "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.2f}% - "
                f"LR: {current_lr:.2e} - "
                f"Time: {epoch_time:.1f}s - "
                f"Speed: {train_speed:.2f} it/s"
            )

            # Update learning rate scheduler if present
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Save checkpoints
            if self.checkpoint_manager:
                # Save latest
                self.checkpoint_manager.save_latest_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch + 1,
                    val_metrics["loss"],
                    val_metrics["accuracy"],
                    {
                        "train_loss": train_metrics["loss"],
                        "train_accuracy": train_metrics["accuracy"],
                    },
                )

                # Save best
                is_best = self.checkpoint_manager.save_best_model(
                    self.model,
                    self.optimizer,
                    epoch + 1,
                    val_metrics["loss"],
                    val_metrics["accuracy"],
                    {
                        "train_loss": train_metrics["loss"],
                        "train_accuracy": train_metrics["accuracy"],
                    },
                )

                if is_best:
                    best_accuracy = val_metrics["accuracy"]
                    print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

                # Save epoch checkpoint periodically
                if (epoch + 1) % self.checkpoint_manager.save_frequency == 0:
                    self.checkpoint_manager.save_epoch_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch + 1,
                        val_metrics["loss"],
                        val_metrics["accuracy"],
                        {
                            "train_loss": train_metrics["loss"],
                            "train_accuracy": train_metrics["accuracy"],
                        },
                    )

            # Plot training curves
            self._plot_training_curves()

        training_time = time.time() - start_time

        return {
            "epochs_trained": epochs - start_epoch,
            "best_accuracy": best_accuracy,
            "training_time": training_time,
            "history": self.history,
            "experiment_dir": self.experiment_manager.experiment_dir,
        }
