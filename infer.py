#!/usr/bin/env python3
"""MNIST / Character classifier inference script.

Loads a trained model checkpoint from a ``runs/exp*`` experiment directory
and performs top-K inference on a user-supplied image.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# ---------------------------------------------------------------------------
# Project imports  —  rely on the same package structure as train.py
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.config import load_config
from src.datasets.utils import (
    get_character_test_transform,
    get_simple_test_transform,
    ResizePad,
)
from src.models.registry import ModelRegistry
from src.utils.device import get_device


# ---------------------------------------------------------------------------
# Inference transform helpers
# ---------------------------------------------------------------------------

def _mnist_test_transform(image_size: int, output_channels: int = 1) -> transforms.Compose:
    """MNIST test transform (no augmentation, only resize + normalise)."""
    mean = (0.1307,) * output_channels
    std = (0.3081,) * output_channels
    steps: list = []
    if output_channels != 1:
        steps.append(transforms.Grayscale(num_output_channels=output_channels))
    if image_size != 28:
        steps.append(transforms.Resize((image_size, image_size), antialias=True))
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(mean, std))
    return transforms.Compose(steps)


def _build_inference_transform(
    dataset_name: str, image_size: int, input_channels: int
) -> transforms.Compose:
    """Select the appropriate inference transform for the given dataset."""
    ds_lower = dataset_name.lower()

    # MNIST-family datasets
    if ds_lower in ("mnist", "triplet_mnist", "balanced_triplet_mnist"):
        return _mnist_test_transform(image_size, input_channels)

    # Chinese character / general image-folder datasets
    # (subset_631, subset_1000, triplet_subset_1000, …)
    # These use the Albumentations-based test pipeline from the project.
    if ds_lower in (
        "subset_631",
        "subset_1000",
        "triplet_subset_1000",
        "cifar10",
    ):
        return get_character_test_transform(image_size, input_channels)

    # Fallback: simplest resize-to-square + normalize
    return get_simple_test_transform(image_size, input_channels)


def _resolve_input_channels(checkpoint_data: dict, config: dict) -> int:
    """Determine the number of input channels the model expects.

    Priority: checkpoint metadata > saved config > 3 (RGB default).
    """
    model_config = checkpoint_data.get("model_config", {})
    channels = model_config.get("input_channels")
    if channels is not None:
        return int(channels)
    channels = config.get("model", {}).get("input_channels")
    if channels is not None:
        return int(channels)
    # A common default; will be overridden if the model has its own default.
    return 3


def _get_image_size(config: dict) -> Tuple[int, int]:
    """Return (height, width) from the model config, defaulting to 64×64."""
    raw = config.get("model", {}).get("input_size", [64, 64])
    if isinstance(raw, (list, tuple)):
        return int(raw[0]), int(raw[1]) if len(raw) > 1 else int(raw[0])
    return int(raw), int(raw)


def _find_experiment_dir(exp_path: str) -> str:
    """Resolve a user-supplied experiment path to the actual directory.

    Accepts::

        exp60
        runs/exp60
        /absolute/path/to/runs/exp60
        runs/exp60   (Windows-style also works)
    """
    p = Path(exp_path)

    # If it already exists, use it directly
    if p.exists():
        return str(p.resolve())

    # If the user gave a bare name like "exp60", look inside runs/
    if not p.is_absolute() and not p.parts[0].startswith("runs"):
        candidate = Path("runs") / p
        if candidate.exists():
            return str(candidate.resolve())

    raise FileNotFoundError(
        f"Experiment directory not found: {exp_path}\n"
        f"  Tried: {p.resolve()}\n"
        f"  Tried: {Path('runs') / p}"
    )


def _find_checkpoint(exp_dir: str, checkpoint_rel: str | None) -> str:
    """Locate a checkpoint file inside the experiment directory."""
    ckpt_dir = os.path.join(exp_dir, "checkpoints")

    if checkpoint_rel:
        ckpt_candidate = os.path.join(ckpt_dir, checkpoint_rel)
        if os.path.isfile(ckpt_candidate):
            return ckpt_candidate
        # Maybe the user gave a full path or relative-to-exp_dir path
        if os.path.isfile(checkpoint_rel):
            return checkpoint_rel
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_rel}\n"
            f"  Also tried: {ckpt_candidate}"
        )

    # Default: best_model.pt, fall back to latest_checkpoint.pt
    for name in ("best_model.pt", "latest_checkpoint.pt"):
        path = os.path.join(ckpt_dir, name)
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        f"No checkpoint found in {ckpt_dir}/  "
        "(looked for best_model.pt, latest_checkpoint.pt)"
    )


def _load_index_label_mapping(exp_dir: str) -> dict:
    """Load the index→label JSON mapping saved during training."""
    json_path = os.path.join(exp_dir, "index_label_mapping.json")
    if not os.path.isfile(json_path):
        print(f"WARNING: index_label_mapping.json not found at {json_path}")
        print("  Predictions will show numeric class indices only.")
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        # Keys in the JSON are strings; convert back to int keys
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Main inference routine
# ---------------------------------------------------------------------------

def infer(
    image_path: str,
    exp_dir: str,
    checkpoint_path: str,
    topk: int = 5,
    device_name: str | None = None,
) -> None:
    """Run inference on a single image and print top-K predictions."""

    # ---- 1. Load config --------------------------------------------------
    config_path = os.path.join(exp_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path=config_path, args=None)
    # load_config returns a Config object; use .to_dict() for dict access
    config_dict = config.to_dict()

    model_name: str = config_dict.get("model", {}).get("name", "")
    if not model_name:
        raise ValueError("model.name not found in config.yaml")

    dataset_name: str = config_dict.get("dataset", {}).get("name", "")
    if not dataset_name:
        print("WARNING: dataset.name not found in config.yaml; will use a generic transform.")

    # ---- 2. Device -------------------------------------------------------
    device, _using_cpu = get_device(device_name)

    # ---- 3. Load checkpoint ----------------------------------------------
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint_data["model_state_dict"]

    # ---- 4. Determine model parameters -----------------------------------
    image_size: tuple = _get_image_size(config_dict)
    input_channels: int = _resolve_input_channels(checkpoint_data, config_dict)
    num_classes: int = (
        config_dict.get("model", {}).get("num_classes", 0)
        or checkpoint_data.get("model_config", {}).get("num_classes", 0)
    )

    # ---- 5. Create model & load weights ----------------------------------
    print(f"Creating model: {model_name}  (classes={num_classes}, "
          f"input_size={image_size}, channels={input_channels})")

    model = ModelRegistry.create(
        model_name,
        num_classes=num_classes,
        input_size=list(image_size),
        input_channels=input_channels,
    )

    # Handle DataParallel / DDP wrapping saved in the checkpoint keys
    if next(iter(state_dict.keys())).startswith("module."):
        from collections import OrderedDict
        new_sd = OrderedDict()
        for k, v in state_dict.items():
            new_sd[k[7:]] = v  # strip "module." prefix
        state_dict = new_sd

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"WARNING: {len(missing)} missing key(s) in state_dict: {missing[:5]}...")
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected key(s): {unexpected[:5]}...")

    model.to(device)
    model.eval()

    # Optional: restore original state_dict for channel count display
    epoch = checkpoint_data.get("epoch", "?")
    ckpt_acc = checkpoint_data.get("accuracy", None)
    print(f"Loaded epoch={epoch}"
          + (f", validation accuracy={ckpt_acc:.4f}" if ckpt_acc is not None else ""))

    # ---- 6. Load image & apply transform ---------------------------------
    print(f"Loading image: {image_path}")
    pil_image = Image.open(image_path).convert("RGB" if input_channels == 3 else "L")

    transform = _build_inference_transform(
        dataset_name, max(image_size), input_channels
    )
    input_tensor = transform(pil_image).unsqueeze(0).to(device)  # (1, C, H, W)

    # ---- 7. Forward pass -------------------------------------------------
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)  # (num_classes,)

    # ---- 8. Top-K --------------------------------------------------------
    topk = min(topk, num_classes) if num_classes > 0 else topk
    top_probs, top_indices = torch.topk(probs, k=topk)

    # ---- 9. Load label mapping & print output ----------------------------
    label_map = _load_index_label_mapping(exp_dir)

    print(f"\n{'=' * 60}")
    print(f"Top-{topk} predictions")
    print(f"{'=' * 60}")
    print(f"{'Rank':<6} {'Index':<8} {'Label':<20} {'Confidence':<12}")
    print("-" * 60)

    for rank, (prob, idx) in enumerate(zip(top_probs.tolist(), top_indices.tolist())):
        label = label_map.get(idx, str(idx))
        print(f"{rank + 1:<6} {idx:<8} {label:<20} {prob:.4f} ({prob * 100:.2f}%)")

    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer a single image using a trained experiment model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python infer.py exp60 --image path/to/image.png\n"
            "  python infer.py runs/exp60 --image img.jpg --topk 3 --device cuda\n"
            "  python infer.py exp60 --image img.png --checkpoint epoch_50.pt\n"
        ),
    )
    parser.add_argument(
        "experiment",
        type=str,
        help="Experiment name or path  (e.g. exp60, runs/exp60)",
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to the input image file",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=None,
        help="Checkpoint filename inside checkpoints/ dir (default: best_model.pt "
             "→ latest_checkpoint.pt)",
    )
    parser.add_argument(
        "--topk", "-k",
        type=int,
        default=5,
        help="Number of top predictions to show (default: 5)",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default=None,
        help='Device to use: "auto", "cpu", "cuda", "mps" (default: auto)',
    )
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    # Resolve paths
    exp_dir = _find_experiment_dir(args.experiment)
    checkpoint_path = _find_checkpoint(exp_dir, args.checkpoint)

    if not os.path.isfile(args.image):
        parser.error(f"Image file not found: {args.image}")

    # Run
    infer(
        image_path=args.image,
        exp_dir=exp_dir,
        checkpoint_path=checkpoint_path,
        topk=args.topk,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
