import os
from datetime import datetime
from typing import Optional


class ExperimentManager:
    """Manage experiment directories in YOLO style (runs/exp1, runs/exp2, ...)."""

    def __init__(self, base_dir: str = "runs"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.experiment_dir = self._create_experiment_dir()

    def _create_experiment_dir(self) -> str:
        """Create a new experiment directory with auto-incrementing name."""
        existing_exps = [
            d
            for d in os.listdir(self.base_dir)
            if os.path.isdir(os.path.join(self.base_dir, d)) and d.startswith("exp")
        ]

        exp_numbers = []
        for exp in existing_exps:
            try:
                num = int(exp.replace("exp", ""))
                exp_numbers.append(num)
            except ValueError:
                continue

        next_exp = max(exp_numbers) + 1 if exp_numbers else 1
        exp_dir = os.path.join(self.base_dir, f"exp{next_exp}")
        os.makedirs(exp_dir, exist_ok=True)

        # Create subdirectories
        os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)

        return exp_dir

    @property
    def checkpoints_dir(self) -> str:
        """Get checkpoints directory path."""
        return os.path.join(self.experiment_dir, "checkpoints")

    @property
    def log_file(self) -> str:
        """Get training log file path."""
        return os.path.join(self.experiment_dir, "training_log.txt")

    @property
    def plot_file(self) -> str:
        """Get training plot file path."""
        return os.path.join(self.experiment_dir, "training_curves.png")

    @property
    def config_file(self) -> str:
        """Get config file path."""
        return os.path.join(self.experiment_dir, "config.yaml")

    def get_experiment_info(self) -> str:
        """Get experiment directory info."""
        return f"Experiment directory: {self.experiment_dir}"
