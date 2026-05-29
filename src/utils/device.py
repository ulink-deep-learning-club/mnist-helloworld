import torch
import os
from typing import Tuple

def get_device(device_name: str | None = None) -> Tuple[torch.device, bool]:
    """Get the best available device and whether we're using CPU.

    Args:
        device_name: Optional device name to use ("cuda", "cpu", "mps").
                     If None, auto-detect the best available device.
    """
    using_cpu = False

    if device_name is not None and device_name.lower() != "auto":
        device_name = device_name.lower()
        if device_name == "cpu":
            using_cpu = True
            device = torch.device("cpu")
        elif device_name == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available on this system")
            device = torch.device("cuda")
        elif device_name == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS is not available on this system")
            device = torch.device("mps")
        else:
            raise ValueError(f"Unsupported device: {device_name}")
    else:
        # Auto-detect the best available device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            using_cpu = True
            device = torch.device('cpu')

    return device, using_cpu

def get_optimal_workers(using_cpu: bool) -> Tuple[int, int]:
    """Get optimal number of workers for train and validation loaders."""
    if using_cpu:
        return 1, 1

    num_cpus = os.cpu_count() or 1
    train_workers = min(8, max(2, num_cpus // 2))
    val_workers = min(4, max(1, train_workers // 2))

    return train_workers, val_workers
