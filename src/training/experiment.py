import os
from typing import Optional


class ExperimentManager:
    """Manage experiment directories in YOLO style (runs/exp1, runs/exp2, ...)."""

    def __init__(self, base_dir: str = "runs", resume_exp: Optional[str] = None):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        if resume_exp:
            self.experiment_dir = self._load_experiment_dir(resume_exp)
        else:
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

    def _load_experiment_dir(self, exp_name: str) -> str:
        """Load an existing experiment directory."""
        exp_dir = os.path.join(self.base_dir, exp_name)
        if not os.path.exists(exp_dir):
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
        if not os.path.isdir(exp_dir):
            raise NotADirectoryError(f"Not a directory: {exp_dir}")

        # Ensure checkpoints directory exists
        checkpoints_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)

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
