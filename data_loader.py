"""
Data loader for Circuit Topology Classifier.
Handles mixed image formats (png, jpg, jpeg, webp, avif, gif).
"""

from pathlib import Path
from typing import Optional, Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image, ImageOps
import pillow_avif  # Required for AVIF support


BACKGROUND_COLOR = (255, 255, 255)
ALPHA_THRESHOLD = 10
LUMA_THRESHOLD = 245
CONTENT_MARGIN = 12


def prepare_circuit_image(image: Image.Image, min_margin_ratio: float = 0.18) -> Image.Image:
    """
    Flatten transparency and center the visible symbol on a square canvas.
    This preserves small topology cues like bubbles and XOR offset lines.
    """
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")

    if alpha.getbbox():
        alpha_mask = alpha.point(lambda value: 255 if value > ALPHA_THRESHOLD else 0)
        bbox = alpha_mask.getbbox()
    else:
        bbox = None

    flattened = Image.alpha_composite(
        Image.new("RGBA", rgba.size, BACKGROUND_COLOR + (255,)),
        rgba,
    ).convert("RGB")

    if bbox is None:
        grayscale = flattened.convert("L")
        foreground_mask = grayscale.point(lambda value: 255 if value < LUMA_THRESHOLD else 0)
        bbox = foreground_mask.getbbox()

    if bbox is None:
        return flattened

    left, top, right, bottom = bbox
    left = max(0, left - CONTENT_MARGIN)
    top = max(0, top - CONTENT_MARGIN)
    right = min(flattened.width, right + CONTENT_MARGIN)
    bottom = min(flattened.height, bottom + CONTENT_MARGIN)
    cropped = flattened.crop((left, top, right, bottom))

    content_size = max(cropped.width, cropped.height)
    margin = max(int(content_size * min_margin_ratio), CONTENT_MARGIN)
    canvas_size = content_size + margin * 2
    canvas = Image.new("RGB", (canvas_size, canvas_size), BACKGROUND_COLOR)

    paste_x = (canvas_size - cropped.width) // 2
    paste_y = (canvas_size - cropped.height) // 2
    canvas.paste(cropped, (paste_x, paste_y))

    return canvas


def load_circuit_image(img_path: Path) -> Image.Image:
    """Load an image and apply circuit-specific preprocessing."""
    with Image.open(img_path) as image:
        return prepare_circuit_image(image)


class CircuitDataset(Dataset):
    """Dataset for loading circuit topology images."""

    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.webp', '.avif', '.gif'}

    def __init__(
        self,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        """
        Args:
            data_dir: Path to data directory with class subfolders
            transform: Optional torchvision transforms
            target_size: Target image size (height, width)
        """
        self.data_dir = Path(data_dir)
        self.target_size = target_size

        # Default transform if none provided
        self.transform = transform or self._default_transform()

        # Discover classes from subdirectories
        self.classes = sorted([
            d.name for d in self.data_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect all image paths and labels
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _default_transform(self) -> transforms.Compose:
        """Default preprocessing transform."""
        return transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_samples(self):
        """Load all image paths and their labels."""
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            class_idx = self.class_to_idx[class_name]

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.SUPPORTED_FORMATS:
                    self.samples.append((img_path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        image = load_circuit_image(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """Return count of images per class."""
        distribution = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            class_name = self.classes[label]
            distribution[class_name] += 1
        return distribution


def get_transforms(target_size: Tuple[int, int] = (224, 224), augment: bool = False):
    """
    Get transform pipelines for training and validation.

    Args:
        target_size: Target image size
        augment: Whether to apply data augmentation (for training)
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if augment:
        return transforms.Compose([
            transforms.Resize((int(target_size[0] * 1.1), int(target_size[1] * 1.1))),
            transforms.RandomCrop(target_size),
            transforms.RandomRotation(degrees=5, fill=255),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.04, 0.04),
                scale=(0.95, 1.05),
                shear=3,
                fill=255,
            ),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            normalize
        ])


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    target_size: Tuple[int, int] = (224, 224),
    train_split: float = 0.7,
    val_split: float = 0.15,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create train, validation, and test data loaders.

    Args:
        data_dir: Path to data directory
        batch_size: Batch size for data loaders
        target_size: Target image size
        train_split: Fraction for training set
        val_split: Fraction for validation set (test = 1 - train - val)
        num_workers: Number of worker processes
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    # Load full dataset without augmentation first for splitting
    full_dataset = CircuitDataset(data_dir, target_size=target_size)

    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size

    # Split dataset
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Apply augmentation to training set
    train_transform = get_transforms(target_size, augment=True)
    val_transform = get_transforms(target_size, augment=False)

    # Create new datasets with appropriate transforms
    train_dataset.dataset = CircuitDataset(data_dir, transform=train_transform, target_size=target_size)
    val_dataset.dataset = CircuitDataset(data_dir, transform=val_transform, target_size=target_size)
    test_dataset.dataset = CircuitDataset(data_dir, transform=val_transform, target_size=target_size)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, full_dataset.classes


if __name__ == "__main__":
    # Example usage
    data_dir = Path(__file__).parent / "data"

    # Create dataset and inspect
    dataset = CircuitDataset(data_dir)
    print(f"Total images: {len(dataset)}")
    print(f"Classes: {dataset.classes}")
    print(f"Class distribution: {dataset.get_class_distribution()}")

    # Create data loaders
    train_loader, val_loader, test_loader, classes = create_data_loaders(
        data_dir,
        batch_size=16,
        num_workers=0  # Set to 0 for debugging
    )

    print(f"\nData loader sizes:")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")

    # Test loading a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels: {labels}")
