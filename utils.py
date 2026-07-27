#!/usr/bin/env python3
"""Utility commands for trained experiment directories.

Subcommands::

    python utils.py onnx    exp60 --output model.onnx
    python utils.py archive exp60 --output exp60.zip
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Project imports  —  rely on the same package structure as train.py/infer.py
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.config import load_config
from src.models.registry import ModelRegistry
from src.utils.device import get_device


# ---------------------------------------------------------------------------
# Path resolution  —  shared with infer.py
# ---------------------------------------------------------------------------

def _find_experiment_dir(exp_path: str) -> str:
    """Resolve a user-supplied experiment path to the actual directory.

    Accepts::

        exp60
        runs/exp60
        /absolute/path/to/runs/exp60
    """
    p = Path(exp_path)

    if p.exists():
        return str(p.resolve())

    if not p.is_absolute() and not p.parts[0].startswith("runs"):
        candidate = Path("runs") / p
        if candidate.exists():
            return str(candidate.resolve())

    raise FileNotFoundError(
        f"Experiment directory not found: {exp_path}\n"
        f"  Tried: {p.resolve()}\n"
        f"  Tried: {Path('runs') / p}"
    )


def _resolve_input_size(raw) -> tuple[int, int]:
    """Normalise *input_size* from config (list, int, or tuple) to (H, W)."""
    if isinstance(raw, (list, tuple)):
        h, w = int(raw[0]), int(raw[1]) if len(raw) > 1 else int(raw[0])
        return h, w
    v = int(raw)
    return v, v


# ===================================================================
# 1. ONNX export
# ===================================================================

def export_onnx(
    exp_dir: str,
    output_path: str = "model.onnx",
    checkpoint_name: str = "best_model.pt",
    opset_version: int = 17,
    device_name: Optional[str] = None,
    dynamic_batch: bool = True,
) -> str:
    """Export a trained model from an experiment directory to ONNX format.

    Returns the path of the written ``.onnx`` file.
    """
    exp_dir = _find_experiment_dir(exp_dir)

    # ---- 1. Load config --------------------------------------------------
    config_path = os.path.join(exp_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path=config_path, args=None)
    cfg = config.to_dict()

    model_name: str = cfg.get("model", {}).get("name", "")
    if not model_name:
        raise ValueError("model.name not found in config.yaml")

    # ---- 2. Resolve model parameters ------------------------------------
    input_size = _resolve_input_size(cfg.get("model", {}).get("input_size", [64, 64]))
    ckpt_path = os.path.join(exp_dir, "checkpoints", checkpoint_name)

    if not os.path.isfile(ckpt_path):
        # Try default fallback filenames
        for fallback in ("best_model.pt", "latest_checkpoint.pt"):
            alt = os.path.join(exp_dir, "checkpoints", fallback)
            if os.path.isfile(alt):
                ckpt_path = alt
                break
        else:
            raise FileNotFoundError(
                f"Checkpoint '{checkpoint_name}' not found under "
                f"{exp_dir}/checkpoints/  (and no best_model.pt / "
                f"latest_checkpoint.pt found either)."
            )

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model_state_dict"]
    model_cfg = checkpoint.get("model_config", {})

    num_classes: int = (
        model_cfg.get("num_classes")
        or cfg.get("model", {}).get("num_classes")
        or 631
    )
    input_channels: int = (
        model_cfg.get("input_channels")
        or cfg.get("model", {}).get("input_channels")
        or 3
    )

    # ---- 3. Create model & load weights ---------------------------------
    device, _ = get_device(device_name or "cpu")

    model = ModelRegistry.create(
        model_name,
        num_classes=num_classes,
        input_size=list(input_size),
        input_channels=input_channels,
    )

    # Strip DataParallel / DDP "module." prefix
    if state_dict and next(iter(state_dict.keys())).startswith("module."):
        from collections import OrderedDict

        new_sd = OrderedDict()
        for k, v in state_dict.items():
            new_sd[k[7:]] = v
        state_dict = new_sd

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"WARNING: {len(missing)} missing key(s): {missing[:5]}…")
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected key(s): {unexpected[:5]}…")

    model.to(device)
    model.eval()

    # Unwrap torch.compile wrapper if present
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod

    # ---- 4. Determine model output description --------------------------
    model_type = getattr(model, "model_type", "classification")
    if model_type == "siamese":
        output_desc = "embedding"
        output_dim = getattr(model, "embedding_dim", num_classes)
    else:
        output_desc = "logits"
        output_dim = num_classes

    # ---- 5. Build dummy input & dynamic axes ----------------------------
    dummy = torch.randn(1, input_channels, input_size[0], input_size[1]).to(device)
    dynamic_axes: Optional[dict] = (
        {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        if dynamic_batch
        else None
    )

    # ---- 6. Export ------------------------------------------------------
    print(f"Exporting {model_name} ({model_type}) to ONNX …")
    print(f"  Input shape  : (batch, {input_channels}, {input_size[0]}, {input_size[1]})")
    print(f"  Output shape : (batch, {output_dim})  [{output_desc}]")
    print(f"  Opset        : {opset_version}")
    if dynamic_batch:
        print("  Batch size   : dynamic")

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    print(f"ONNX model saved to: {output_path}")
    return output_path


# ===================================================================
# 2. Archive (zip) experiment
# ===================================================================

def archive_experiment(
    exp_dir: str,
    output_path: Optional[str] = None,
    include_checkpoints: bool = False,
) -> str:
    """Archive an experiment directory into a ``.zip`` file.

    By default, *checkpoints* are **excluded** because they are usually
    large and versioned separately.  Use ``--include-checkpoints`` to
    include them.

    Returns the path of the written archive.
    """
    exp_dir = _find_experiment_dir(exp_dir)
    exp_name = os.path.basename(exp_dir)

    if output_path is None:
        output_path = f"{exp_name}.zip"

    # Ensure .zip extension
    if not output_path.endswith(".zip"):
        output_path += ".zip"

    skipped_checkpoints = 0
    total_files = 0

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(exp_dir):
            # Sort for deterministic archive order
            dirs.sort()
            files.sort()

            # Optionally prune the checkpoints/ subtree
            if not include_checkpoints and os.path.basename(root) == "checkpoints":
                skipped_checkpoints += len(files)
                dirs.clear()
                continue

            for fname in files:
                file_path = os.path.join(root, fname)
                arcname = os.path.relpath(file_path, os.path.dirname(exp_dir))
                zf.write(file_path, arcname)
                total_files += 1

    print(f"Archived {total_files} file(s) from {exp_name}/  →  {output_path}")
    if skipped_checkpoints:
        print(f"  (excluded {skipped_checkpoints} checkpoint file(s); "
              f"use --include-checkpoints to add them)")
    return output_path


# ===================================================================
# CLI
# ===================================================================

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Utility commands for trained experiment directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python utils.py onnx    exp60 -o model.onnx\n"
            "  python utils.py onnx    exp60 --checkpoint epoch_50.pt --opset 20\n"
            "  python utils.py archive exp60 -o exp60.zip\n"
            "  python utils.py archive exp60 --include-checkpoints\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- onnx -----------------------------------------------------------
    p_onnx = sub.add_parser("onnx", help="Export model to ONNX format")
    p_onnx.add_argument("experiment", help="Experiment name or path  (e.g. exp60, runs/exp60)")
    p_onnx.add_argument("-o", "--output", default="model.onnx",
                        help="Output .onnx file path  (default: model.onnx)")
    p_onnx.add_argument("--checkpoint", default="best_model.pt",
                        help="Checkpoint filename  (default: best_model.pt)")
    p_onnx.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version  (default: 17)")
    p_onnx.add_argument("--device", default=None,
                        help='Device: "auto", "cpu", "cuda", "mps"  (default: auto)')
    p_onnx.add_argument("--no-dynamic-batch", action="store_false", dest="dynamic_batch",
                        help="Use a fixed batch-size 1 instead of dynamic batching")
    p_onnx.set_defaults(func=_run_onnx)

    # ---- archive --------------------------------------------------------
    p_arc = sub.add_parser("archive", help="Package experiment into a zip archive")
    p_arc.add_argument("experiment", help="Experiment name or path  (e.g. exp60, runs/exp60)")
    p_arc.add_argument("-o", "--output", default=None,
                       help="Output archive path  (default: <exp_name>.zip)")
    p_arc.add_argument("--include-checkpoints", action="store_true",
                       help="Include checkpoints/ directory in the archive")
    p_arc.set_defaults(func=_run_archive)

    return parser


def _run_onnx(args: argparse.Namespace) -> None:
    export_onnx(
        exp_dir=args.experiment,
        output_path=args.output,
        checkpoint_name=args.checkpoint,
        opset_version=args.opset,
        device_name=args.device,
        dynamic_batch=args.dynamic_batch,
    )


def _run_archive(args: argparse.Namespace) -> None:
    archive_experiment(
        exp_dir=args.experiment,
        output_path=args.output,
        include_checkpoints=args.include_checkpoints,
    )


# ===================================================================
# Entry point
# ===================================================================

def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
