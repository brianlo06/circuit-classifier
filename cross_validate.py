"""
Grouped cross-validation runner for the circuit classifier.
"""

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from model import get_model
from train_improved import LabelSmoothingCrossEntropy, evaluate, train_epoch
from training_common import create_loaders_from_indices, prepare_crossval_folds


def create_model(model_name: str, num_classes: int):
    """Create a model with transfer-specific kwargs only where needed."""
    if model_name in {"resnet", "efficientnet"}:
        return get_model(model_name, num_classes=num_classes, freeze_layers=False)
    return get_model(model_name, num_classes=num_classes)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_fold(
    data_dir: str,
    class_names,
    train_indices,
    val_indices,
    output_dir: Path,
    fold_idx: int,
    model_name: str,
    epochs: int,
    batch_size: int,
    image_size: int,
    base_lr: float,
    weight_decay: float,
    label_smoothing: float,
    mixup_alpha: float,
    patience: int,
    device: torch.device,
):
    train_loader, val_loader, _ = create_loaders_from_indices(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        train_indices=train_indices,
        val_indices=val_indices,
        drop_last_train=False,
    )

    model = create_model(model_name, num_classes=len(class_names)).to(device)
    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    val_criterion = nn.CrossEntropyLoss()

    if hasattr(model, "get_parameter_groups"):
        optimizer = optim.AdamW(model.get_parameter_groups(base_lr), weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=base_lr * 0.01)

    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    print(f"\nFold {fold_idx + 1}: {len(train_indices)} train / {len(val_indices)} val")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, mixup_alpha
        )
        val_loss, val_acc = evaluate(model, val_loader, val_criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[-1]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"Fold {fold_idx + 1} Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | "
            f"LR: {current_lr:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "class_names": list(class_names),
                    "model_name": model_name,
                },
                output_dir / f"fold_{fold_idx + 1}_best_model.pth",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    elapsed = time.time() - start_time
    with open(output_dir / f"fold_{fold_idx + 1}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {
        "fold": fold_idx + 1,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "epochs_trained": len(history["val_acc"]),
        "training_time_seconds": elapsed,
    }


def cross_validate(
    data_dir: str,
    output_dir: str = "crossval",
    model_name: str = "resnet",
    folds: int = 5,
    epochs: int = 30,
    batch_size: int = 12,
    image_size: int = 224,
    base_lr: float = 0.0003,
    weight_decay: float = 0.01,
    label_smoothing: float = 0.1,
    mixup_alpha: float = 0.2,
    seed: int = 42,
    patience: int = 10,
    duplicate_threshold: int = 10,
):
    torch.manual_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    device = get_device()
    print(f"Using {device.type.upper()}")

    class_names, fold_indices, diagnostics = prepare_crossval_folds(
        data_dir=data_dir,
        num_folds=folds,
        seed=seed,
        duplicate_threshold=duplicate_threshold,
    )

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Total images: {diagnostics['num_samples']}")
    print(f"Fold sizes: {diagnostics['fold_sizes']}")
    print(f"Duplicate groups kept within a single fold: {diagnostics['cross_fold_duplicate_groups'] == 0}")

    with open(output_path / "folds_info.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    fold_results = []
    overall_start = time.time()

    for fold_idx in range(folds):
        val_indices = fold_indices[fold_idx]
        train_indices = []
        for other_idx, indices in enumerate(fold_indices):
            if other_idx != fold_idx:
                train_indices.extend(indices)

        result = train_fold(
            data_dir=data_dir,
            class_names=class_names,
            train_indices=train_indices,
            val_indices=val_indices,
            output_dir=output_path,
            fold_idx=fold_idx,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            base_lr=base_lr,
            weight_decay=weight_decay,
            label_smoothing=label_smoothing,
            mixup_alpha=mixup_alpha,
            patience=patience,
            device=device,
        )
        fold_results.append(result)

    val_scores = [result["best_val_acc"] for result in fold_results]
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "num_folds": folds,
        "mean_best_val_acc": statistics.mean(val_scores),
        "std_best_val_acc": statistics.pstdev(val_scores) if len(val_scores) > 1 else 0.0,
        "min_best_val_acc": min(val_scores),
        "max_best_val_acc": max(val_scores),
        "total_time_seconds": time.time() - overall_start,
        "fold_results": fold_results,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "image_size": image_size,
            "base_lr": base_lr,
            "weight_decay": weight_decay,
            "label_smoothing": label_smoothing,
            "mixup_alpha": mixup_alpha,
            "patience": patience,
            "duplicate_threshold": duplicate_threshold,
            "seed": seed,
        },
    }

    with open(output_path / "cv_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nCross-validation summary")
    print("-" * 60)
    for result in fold_results:
        print(
            f"Fold {result['fold']}: best val {result['best_val_acc']:.1f}% "
            f"at epoch {result['best_epoch']}"
        )
    print(
        f"Mean best val accuracy: {summary['mean_best_val_acc']:.2f}% "
        f"+/- {summary['std_best_val_acc']:.2f}"
    )
    print(f"Results saved to: {output_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grouped cross-validation for Circuit Classifier")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="crossval")
    parser.add_argument("--model", type=str, default="resnet", choices=["small", "standard", "resnet", "efficientnet"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duplicate-threshold", type=int, default=10)
    args = parser.parse_args()

    cross_validate(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        base_lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup,
        seed=args.seed,
        patience=args.patience,
        duplicate_threshold=args.duplicate_threshold,
    )
