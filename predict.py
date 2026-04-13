"""
Inference script for Circuit Topology Classifier.
Predict the gate type of a circuit image.
"""

import argparse
from pathlib import Path

import torch
from torchvision import transforms

from data_loader import load_circuit_image
from model import get_model


def load_model(checkpoint_path: str, device: torch.device):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_name = checkpoint.get("model_name", "small")
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = get_model(model_name, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, class_names


def preprocess_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """Load and preprocess an image for inference."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = load_circuit_image(Path(image_path))
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor


def predict(
    image_path: str,
    checkpoint_path: str = "checkpoints/best_model.pth",
    image_size: int = 224,
    top_k: int = 3,
):
    """
    Predict the gate type of an image.

    Args:
        image_path: Path to the image
        checkpoint_path: Path to model checkpoint
        image_size: Input image size
        top_k: Number of top predictions to show
    """
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load model
    model, class_names = load_model(checkpoint_path, device)

    # Preprocess image
    image_tensor = preprocess_image(image_path, image_size).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, min(top_k, len(class_names)))

    # Print results
    print(f"\nPredictions for: {image_path}")
    print("-" * 40)

    for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
        class_name = class_names[idx.item()]
        confidence = prob.item() * 100
        marker = "<<<" if i == 0 else ""
        print(f"  {class_name:6s}: {confidence:5.1f}% {marker}")

    # Return top prediction
    top_class = class_names[top_indices[0][0].item()]
    top_confidence = top_probs[0][0].item()

    return top_class, top_confidence


def predict_batch(
    image_dir: str,
    checkpoint_path: str = "checkpoints/best_model.pth",
    image_size: int = 224,
):
    """Predict for all images in a directory."""
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load model
    model, class_names = load_model(checkpoint_path, device)

    # Find all images
    image_dir = Path(image_dir)
    extensions = {'.png', '.jpg', '.jpeg', '.webp', '.avif', '.gif'}
    images = [f for f in image_dir.iterdir() if f.suffix.lower() in extensions]

    if not images:
        print(f"No images found in {image_dir}")
        return

    print(f"\nPredicting {len(images)} images...")
    print("-" * 50)

    results = []
    for img_path in sorted(images):
        image_tensor = preprocess_image(str(img_path), image_size).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_prob, top_idx = torch.max(probabilities, dim=1)

        predicted_class = class_names[top_idx.item()]
        confidence = top_prob.item() * 100

        print(f"  {img_path.name:30s} -> {predicted_class:6s} ({confidence:.1f}%)")
        results.append((img_path.name, predicted_class, confidence))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict circuit gate type")
    parser.add_argument("image", type=str, help="Image path or directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()

    path = Path(args.image)
    if path.is_dir():
        predict_batch(args.image, args.checkpoint, args.image_size)
    else:
        predict(args.image, args.checkpoint, args.image_size, args.top_k)
