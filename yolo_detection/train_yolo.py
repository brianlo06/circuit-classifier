"""
Train YOLOv8 for circuit component detection.

Requirements:
    pip install ultralytics

Usage:
    python train_yolo.py                    # Train with defaults
    python train_yolo.py --model yolov8s.pt # Use larger model
    python train_yolo.py --epochs 200       # Train longer
"""

import argparse
from pathlib import Path
from typing import Optional


def train(
    model_name: str = "yolov8n.pt",
    data_config: str = "data.yaml",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/detect",
    name: str = "circuit_components",
    patience: int = 20,
    device: str = "",  # auto-detect
    workers: int = 8,
    cache: bool = False,
    pretrained_weights: Optional[str] = None,
):
    """
    Train YOLOv8 model for circuit component detection.

    Args:
        model_name: Pretrained model to start from (yolov8n/s/m/l/x.pt)
        data_config: Path to data.yaml
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        project: Project directory for saving runs
        name: Experiment name
        patience: Early stopping patience
        device: Device to use ('' for auto, 'mps' for Apple Silicon, 'cuda:0' for GPU)
        workers: Data loader workers
        cache: Cache dataset in RAM/disk if supported by Ultralytics
        pretrained_weights: Optional checkpoint to fine-tune from
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed")
        print("Install with: pip install ultralytics")
        return

    project_dir = Path(project).expanduser()
    if not project_dir.is_absolute():
        project_dir = Path.cwd() / project_dir
    project_dir.mkdir(parents=True, exist_ok=True)

    model_path = pretrained_weights or model_name

    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Train
    print(f"\nStarting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Data config: {data_config}")
    print(f"  Run directory: {project_dir / name}")

    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(project_dir),
        name=name,
        patience=patience,
        device=device if device else None,
        workers=workers,
        cache=cache,
        # Augmentation settings good for technical drawings
        hsv_h=0.0,  # No hue augmentation (preserve colors)
        hsv_s=0.3,  # Slight saturation changes
        hsv_v=0.3,  # Slight brightness changes
        degrees=15,  # Small rotations
        translate=0.1,
        scale=0.3,
        flipud=0.0,  # Don't flip vertically (gates have orientation)
        fliplr=0.5,  # Horizontal flip OK for some gates
        mosaic=0.5,  # Reduced mosaic (can create unrealistic combinations)
    )

    print(f"\nTraining complete!")
    print(f"Results saved to: {project_dir / name}")

    # Validate
    print("\nRunning validation...")
    metrics = model.val()

    print(f"\nValidation Results:")
    print(f"  mAP50: {metrics.box.map50:.3f}")
    print(f"  mAP50-95: {metrics.box.map:.3f}")

    return model, results


def predict_sample(
    model_path: str,
    image_path: str,
    conf: float = 0.25,
    save: bool = True,
):
    """
    Run prediction on a sample image.

    Args:
        model_path: Path to trained model (best.pt)
        image_path: Path to image or directory
        conf: Confidence threshold
        save: Save annotated images
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed")
        return

    model = YOLO(model_path)
    results = model.predict(
        source=image_path,
        conf=conf,
        save=save,
        show_labels=True,
        show_conf=True,
    )

    for r in results:
        print(f"\nImage: {r.path}")
        print(f"Detected {len(r.boxes)} objects:")
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = r.names[cls_id]
            conf = float(box.conf[0])
            print(f"  - {cls_name}: {conf:.2%}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 for circuit detection")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Base model (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--data", type=str, default="data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--device", type=str, default="",
                        help="Device ('' for auto, 'mps', 'cuda:0')")
    parser.add_argument("--project", type=str, default="runs/detect",
                        help="Directory for YOLO run outputs")
    parser.add_argument("--name", type=str, default="circuit_components",
                        help="Experiment name")
    parser.add_argument("--workers", type=int, default=8,
                        help="Data loader workers")
    parser.add_argument("--cache", action="store_true",
                        help="Cache dataset for faster repeated training")
    parser.add_argument("--weights", type=str, default=None,
                        help="Checkpoint to fine-tune from instead of base model")

    args = parser.parse_args()

    train(
        model_name=args.model,
        data_config=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        cache=args.cache,
        pretrained_weights=args.weights,
    )
