"""
Resume YOLOv8 training from last checkpoint.

Usage:
    python resume_training.py
    python resume_training.py --checkpoint runs/detect/complex_circuits_v1/weights/last.pt --data datasets/complex_circuits/data.yaml
"""

import argparse
from pathlib import Path
from typing import Dict, Optional


def find_latest_checkpoint(base_dir: str = "runs/detect") -> str:
    """Find the most recent last.pt checkpoint."""
    base_path = Path(base_dir)
    checkpoints = list(base_path.rglob("weights/last.pt"))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {base_dir}")

    # Sort by modification time, newest first
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(checkpoints[0])


def _read_args_yaml(checkpoint: Path) -> Dict[str, str]:
    """Read the sibling args.yaml written by Ultralytics, if present."""
    for parent in checkpoint.parents:
        args_path = parent / "args.yaml"
        if not args_path.exists():
            continue

        values: Dict[str, str] = {}
        for line in args_path.read_text().splitlines():
            if not line or line.lstrip().startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
        return values
    return {}


def _resolve_data_override(checkpoint: Path, data: Optional[str]) -> Optional[str]:
    if data:
        return str(Path(data).expanduser())

    args = _read_args_yaml(checkpoint)
    stored_data = args.get("data")
    if stored_data and stored_data != "coco8.yaml":
        return stored_data

    if stored_data == "coco8.yaml":
        print("Warning: checkpoint args.yaml stored data=coco8.yaml; pass --data to resume with the intended dataset")
    return None


def resume_training(checkpoint: str = None, epochs: int = 100, data: Optional[str] = None):
    """Resume training from checkpoint."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed")
        print("Install with: pip install ultralytics")
        return

    if checkpoint is None:
        checkpoint = find_latest_checkpoint()
    else:
        checkpoint = str(Path(checkpoint).expanduser())

    checkpoint_path = Path(checkpoint)
    data_override = _resolve_data_override(checkpoint_path, data)

    print(f"Resuming from: {checkpoint}")
    if data_override:
        print(f"Using data config: {data_override}")

    model = YOLO(checkpoint)
    train_kwargs = {"resume": True, "epochs": epochs}
    if data_override:
        train_kwargs["data"] = data_override
    results = model.train(**train_kwargs)

    print("\nTraining complete!")
    return model, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume YOLOv8 training")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: find latest)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Total epochs to train")
    parser.add_argument("--project", type=str, default="runs/detect",
                        help="Base directory to search for checkpoints")
    parser.add_argument("--data", type=str, default=None,
                        help="Optional dataset override for checkpoints with bad stored data config")

    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = find_latest_checkpoint(args.project)
    resume_training(checkpoint=args.checkpoint, epochs=args.epochs, data=args.data)
