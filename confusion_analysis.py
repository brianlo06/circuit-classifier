"""
Confusion matrix analysis for circuit classifier.
Evaluates model on all data and shows misclassification patterns.
"""

import json
from pathlib import Path
from collections import defaultdict

import torch
from torchvision import transforms

from data_loader import load_circuit_image
from model import get_model


def load_model(checkpoint_path: str, device: torch.device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = checkpoint.get("model_name", "small")
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", 224)
    num_classes = len(class_names)
    model = get_model(model_name, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, class_names, image_size


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def analyze_confusion(
    data_dir: str = "data",
    checkpoint_path: str = "checkpoints_diagnostic_latest/best_model.pth",
):
    """Generate confusion matrix and analyze misclassifications."""
    device = get_device()
    model, class_names, image_size = load_model(checkpoint_path, device)
    print(f"Using image size: {image_size}x{image_size}")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_path = Path(data_dir)
    extensions = {'.png', '.jpg', '.jpeg', '.webp', '.avif', '.gif'}

    # Initialize confusion matrix
    n_classes = len(class_names)
    confusion = [[0] * n_classes for _ in range(n_classes)]
    misclassifications = []

    # Evaluate all images
    for class_idx, class_name in enumerate(class_names):
        class_dir = data_path / class_name
        if not class_dir.exists():
            continue

        images = [f for f in class_dir.iterdir() if f.suffix.lower() in extensions]

        for img_path in images:
            try:
                image = load_circuit_image(img_path)
                tensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred_idx = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][pred_idx].item()

                confusion[class_idx][pred_idx] += 1

                if pred_idx != class_idx:
                    misclassifications.append({
                        "file": str(img_path),
                        "true": class_name,
                        "predicted": class_names[pred_idx],
                        "confidence": confidence
                    })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    # Print confusion matrix
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)
    print(f"\n{'':8s}", end="")
    for name in class_names:
        print(f"{name:>7s}", end="")
    print("  | Total  Acc%")
    print("-" * 70)

    class_correct = []
    class_total = []

    for i, true_class in enumerate(class_names):
        print(f"{true_class:8s}", end="")
        row_total = sum(confusion[i])
        correct = confusion[i][i]
        class_correct.append(correct)
        class_total.append(row_total)

        for j in range(n_classes):
            val = confusion[i][j]
            if i == j:
                print(f"{val:>7d}", end="")  # Correct predictions
            elif val > 0:
                print(f"\033[91m{val:>7d}\033[0m", end="")  # Errors in red
            else:
                print(f"{val:>7d}", end="")

        acc = (correct / row_total * 100) if row_total > 0 else 0
        print(f"  | {row_total:>5d}  {acc:>5.1f}%")

    print("-" * 70)
    total_correct = sum(class_correct)
    total_samples = sum(class_total)
    overall_acc = (total_correct / total_samples * 100) if total_samples > 0 else 0
    print(f"{'Total':8s}", end="")
    for j in range(n_classes):
        col_total = sum(confusion[i][j] for i in range(n_classes))
        print(f"{col_total:>7d}", end="")
    print(f"  | {total_samples:>5d}  {overall_acc:>5.1f}%")

    # Analyze error patterns
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    error_pairs = defaultdict(int)
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and confusion[i][j] > 0:
                error_pairs[(class_names[i], class_names[j])] = confusion[i][j]

    if error_pairs:
        sorted_errors = sorted(error_pairs.items(), key=lambda x: -x[1])
        print("\nMost common misclassification pairs:")
        for (true, pred), count in sorted_errors[:10]:
            print(f"  {true:6s} -> {pred:6s}: {count} errors")

    # Group by confusion type
    print("\n" + "=" * 70)
    print("CONFUSION BY GATE RELATIONSHIP")
    print("=" * 70)

    # Pairs that differ only by output bubble
    bubble_pairs = [("AND", "NAND"), ("OR", "NOR"), ("XOR", "XNOR")]
    print("\nBubble confusion (differ by output inverter):")
    for a, b in bubble_pairs:
        if a in class_names and b in class_names:
            ai, bi = class_names.index(a), class_names.index(b)
            errors = confusion[ai][bi] + confusion[bi][ai]
            print(f"  {a} <-> {b}: {errors} mutual errors")

    # Pairs that differ by input line (OR vs XOR type)
    input_pairs = [("OR", "XOR"), ("NOR", "XNOR")]
    print("\nInput-line confusion (differ by extra curved line):")
    for a, b in input_pairs:
        if a in class_names and b in class_names:
            ai, bi = class_names.index(a), class_names.index(b)
            errors = confusion[ai][bi] + confusion[bi][ai]
            print(f"  {a} <-> {b}: {errors} mutual errors")

    # Shape confusion (AND-type vs OR-type)
    print("\nShape confusion (AND-type vs OR-type):")
    and_type = ["AND", "NAND"]
    or_type = ["OR", "NOR", "XOR", "XNOR"]
    cross_errors = 0
    for a in and_type:
        for b in or_type:
            if a in class_names and b in class_names:
                ai, bi = class_names.index(a), class_names.index(b)
                cross_errors += confusion[ai][bi] + confusion[bi][ai]
    print(f"  AND-type <-> OR-type: {cross_errors} mutual errors")

    # NOT gate confusion
    if "NOT" in class_names:
        not_idx = class_names.index("NOT")
        not_errors = sum(confusion[not_idx]) - confusion[not_idx][not_idx]
        not_errors += sum(confusion[i][not_idx] for i in range(n_classes) if i != not_idx)
        print(f"\nNOT gate total confusion: {not_errors} errors")

    # List specific misclassified files
    if misclassifications:
        print("\n" + "=" * 70)
        print(f"MISCLASSIFIED FILES ({len(misclassifications)} total)")
        print("=" * 70)

        # Sort by confidence (most confident mistakes first)
        sorted_misc = sorted(misclassifications, key=lambda x: -x["confidence"])

        for m in sorted_misc[:20]:  # Show top 20
            fname = Path(m["file"]).name
            print(f"  {fname:25s}  {m['true']:6s} -> {m['predicted']:6s}  ({m['confidence']*100:.1f}%)")

        if len(misclassifications) > 20:
            print(f"  ... and {len(misclassifications) - 20} more")

    return confusion, misclassifications


if __name__ == "__main__":
    import sys
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_diagnostic_latest/best_model.pth"
    analyze_confusion(checkpoint_path=checkpoint)
