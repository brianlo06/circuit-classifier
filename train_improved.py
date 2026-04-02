"""
Improved training script with better techniques for small datasets.
- Differential learning rates for transfer learning
- Label smoothing
- Cosine annealing with warm restarts
- Stronger regularization
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from model import get_model
from training_common import prepare_datasets, save_split_info


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_preds = torch.log_softmax(pred, dim=1)

        # Create smoothed labels
        with torch.no_grad():
            smooth_labels = torch.zeros_like(log_preds)
            smooth_labels.fill_(self.smoothing / (n_classes - 1))
            smooth_labels.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        loss = (-smooth_labels * log_preds).sum(dim=1).mean()
        return loss


def train_epoch(model, loader, criterion, optimizer, device, mixup_alpha=0.0):
    """Train for one epoch with optional mixup."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Mixup augmentation
        if mixup_alpha > 0:
            lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
            idx = torch.randperm(images.size(0))
            images = lam * images + (1 - lam) * images[idx]
            labels_a, labels_b = labels, labels[idx]

            optimizer.zero_grad()
            outputs = model(images)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def train(
    data_dir: str,
    output_dir: str = "checkpoints",
    model_name: str = "resnet",
    epochs: int = 60,
    batch_size: int = 16,
    base_lr: float = 0.0003,
    weight_decay: float = 0.01,
    label_smoothing: float = 0.1,
    mixup_alpha: float = 0.2,
    train_split: float = 0.7,
    val_split: float = 0.15,
    image_size: int = 224,
    seed: int = 42,
    patience: int = 20,
    duplicate_threshold: int = 10,
):
    """Train with improved techniques."""

    torch.manual_seed(seed)

    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\nLoading data from {data_dir}...")
    train_loader, val_loader, test_loader, class_names, split_info = prepare_datasets(
        data_dir=data_dir,
        image_size=image_size,
        train_split=train_split,
        val_split=val_split,
        seed=seed,
        batch_size=batch_size,
        duplicate_threshold=duplicate_threshold,
        drop_last_train=False,
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Total images: {split_info['num_samples']}")
    print(
        f"Split: {split_info['split_sizes']['train']} train / "
        f"{split_info['split_sizes']['val']} val / "
        f"{split_info['split_sizes']['test']} test"
    )
    print(f"Duplicate groups kept within a single split: {split_info['cross_split_duplicate_groups'] == 0}")

    # Create model
    print(f"\nCreating model: {model_name}")
    model = get_model(model_name, num_classes=num_classes, freeze_layers=False)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss with label smoothing
    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    val_criterion = nn.CrossEntropyLoss()  # Regular CE for validation

    # Optimizer with differential learning rates
    if hasattr(model, 'get_parameter_groups'):
        param_groups = model.get_parameter_groups(base_lr)
        optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
        print("Using differential learning rates")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    # Cosine annealing scheduler with warm restarts
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=base_lr * 0.01)

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Label smoothing: {label_smoothing}, Mixup alpha: {mixup_alpha}")
    print("-" * 70)

    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, mixup_alpha
        )
        val_loss, val_acc = evaluate(model, val_loader, val_criterion, device)

        scheduler.step()

        current_lr = optimizer.param_groups[-1]['lr']  # Get classifier LR
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | "
              f"LR: {current_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_names": class_names,
                "model_name": model_name,
                "image_size": image_size,
            }
            torch.save(checkpoint, output_path / "best_model.pth")
            print(f"  -> Saved best model (val_acc: {val_acc:.1f}%)")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    total_time = time.time() - start_time
    print("-" * 70)
    print(f"Training complete in {total_time:.1f}s")
    print(f"Best validation accuracy: {best_val_acc:.1f}% (epoch {best_epoch})")

    # Save final model
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "model_name": model_name,
        "image_size": image_size,
    }, output_path / "final_model.pth")

    with open(output_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    save_split_info(output_path, split_info)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    checkpoint = torch.load(output_path / "best_model.pth", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, val_criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.1f}%")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    model.eval()
    class_correct = {c: 0 for c in class_names}
    class_total = {c: 0 for c in class_names}

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            for label, pred in zip(labels, predicted):
                class_name = class_names[label.item()]
                class_total[class_name] += 1
                if label == pred:
                    class_correct[class_name] += 1

    for class_name in class_names:
        if class_total[class_name] > 0:
            acc = 100.0 * class_correct[class_name] / class_total[class_name]
            print(f"  {class_name:6s}: {acc:5.1f}% ({class_correct[class_name]}/{class_total[class_name]})")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "best_epoch": best_epoch,
        "epochs_trained": epoch,
        "training_time_seconds": total_time,
        "config": {
            "batch_size": batch_size,
            "base_lr": base_lr,
            "weight_decay": weight_decay,
            "label_smoothing": label_smoothing,
            "mixup_alpha": mixup_alpha,
            "image_size": image_size,
        },
        "split_info_file": "split_info.json",
        "cross_split_duplicate_groups": split_info["cross_split_duplicate_groups"],
    }
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nCheckpoints saved to: {output_path}")
    return best_val_acc, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved training for Circuit Classifier")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--model", type=str, default="resnet", choices=["resnet", "efficientnet"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup", type=float, default=0.2)
    parser.add_argument("--duplicate-threshold", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)

    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        base_lr=args.lr,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup,
        patience=args.patience,
        duplicate_threshold=args.duplicate_threshold,
        image_size=args.image_size,
    )
