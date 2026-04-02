"""
Training script for Circuit Topology Classifier.
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from model import get_model
from training_common import prepare_datasets, save_split_info


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

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
    model_name: str = "small",
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    train_split: float = 0.7,
    val_split: float = 0.15,
    image_size: int = 224,
    seed: int = 42,
    patience: int = 10,
    duplicate_threshold: int = 10,
):
    """
    Train the circuit classifier.

    Args:
        data_dir: Path to data directory
        output_dir: Directory to save checkpoints
        model_name: Model architecture ("small" or "standard")
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        train_split: Fraction for training
        val_split: Fraction for validation
        image_size: Input image size
        seed: Random seed
        patience: Early stopping patience
    """
    # Set seed for reproducibility
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

    # Create output directory
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
    print(f"Class distribution: {split_info['class_distribution']}")
    print(
        f"\nSplit: {split_info['split_sizes']['train']} train / "
        f"{split_info['split_sizes']['val']} val / "
        f"{split_info['split_sizes']['test']} test"
    )
    print(f"Duplicate groups kept within a single split: {split_info['cross_split_duplicate_groups'] == 0}")

    # Create model
    print(f"\nCreating model: {model_name}")
    model = get_model(model_name, num_classes=num_classes)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 60)

    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | "
              f"LR: {current_lr:.6f}")

        # Save best model
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
            }
            torch.save(checkpoint, output_path / "best_model.pth")
            print(f"  -> Saved best model (val_acc: {val_acc:.1f}%)")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training complete in {total_time:.1f}s")
    print(f"Best validation accuracy: {best_val_acc:.1f}% (epoch {best_epoch})")

    # Save final model
    final_checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_names": class_names,
        "model_name": model_name,
    }
    torch.save(final_checkpoint, output_path / "final_model.pth")

    # Save training history
    with open(output_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    save_split_info(output_path, split_info)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    checkpoint = torch.load(output_path / "best_model.pth", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
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

    # Save results summary
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "epochs_trained": epoch,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "total_params": total_params,
        "training_time_seconds": total_time,
        "class_names": class_names,
        "split_info_file": "split_info.json",
        "cross_split_duplicate_groups": split_info["cross_split_duplicate_groups"],
    }
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nCheckpoints saved to: {output_path}")
    return best_val_acc, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Circuit Topology Classifier")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--model", type=str, default="resnet", choices=["small", "standard", "resnet", "efficientnet"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--duplicate-threshold", type=int, default=10)

    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        patience=args.patience,
        duplicate_threshold=args.duplicate_threshold,
    )
