"""
Image augmentation script for expanding the circuit dataset.
Creates variations of existing images through transformations.
"""

import random
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pillow_avif


SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.webp', '.avif', '.gif'}


class ImageAugmentor:
    """Creates augmented variations of images."""

    def __init__(self):
        self.transforms = [
            self._rotate,
            self._flip_horizontal,
            self._adjust_brightness,
            self._adjust_contrast,
            self._add_noise,
            self._scale,
            self._slight_skew,
            self._invert_colors,
            self._blur,
            self._sharpen,
        ]

    def _rotate(self, img: Image.Image) -> Image.Image:
        """Rotate by small random angle."""
        angle = random.uniform(-15, 15)
        return img.rotate(angle, fillcolor='white', expand=False)

    def _flip_horizontal(self, img: Image.Image) -> Image.Image:
        """Horizontal flip."""
        return ImageOps.mirror(img)

    def _adjust_brightness(self, img: Image.Image) -> Image.Image:
        """Adjust brightness."""
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)

    def _adjust_contrast(self, img: Image.Image) -> Image.Image:
        """Adjust contrast."""
        factor = random.uniform(0.7, 1.4)
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)

    def _add_noise(self, img: Image.Image) -> Image.Image:
        """Add slight noise/grain."""
        img_array = img.copy()
        width, height = img.size

        # Add random pixel noise
        pixels = img_array.load()
        for _ in range(int(width * height * 0.01)):  # 1% of pixels
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            if img.mode == 'RGB':
                noise = tuple(random.randint(-30, 30) for _ in range(3))
                original = pixels[x, y]
                new_val = tuple(max(0, min(255, original[i] + noise[i])) for i in range(3))
                pixels[x, y] = new_val

        return img_array

    def _scale(self, img: Image.Image) -> Image.Image:
        """Scale image slightly."""
        factor = random.uniform(0.85, 1.15)
        new_size = (int(img.width * factor), int(img.height * factor))
        scaled = img.resize(new_size, Image.Resampling.LANCZOS)

        # Pad or crop back to original size
        result = Image.new('RGB', img.size, 'white')
        paste_x = (img.width - scaled.width) // 2
        paste_y = (img.height - scaled.height) // 2
        result.paste(scaled, (paste_x, paste_y))

        return result

    def _slight_skew(self, img: Image.Image) -> Image.Image:
        """Apply slight perspective skew."""
        width, height = img.size
        skew = random.uniform(-0.1, 0.1)

        # Affine transform coefficients
        coeffs = (
            1, skew, -skew * height / 2,
            0, 1, 0
        )
        return img.transform(img.size, Image.AFFINE, coeffs, fillcolor='white')

    def _invert_colors(self, img: Image.Image) -> Image.Image:
        """Invert colors (creates white-on-black variant)."""
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return ImageOps.invert(img)

    def _blur(self, img: Image.Image) -> Image.Image:
        """Apply slight blur."""
        return img.filter(ImageFilter.GaussianBlur(radius=0.5))

    def _sharpen(self, img: Image.Image) -> Image.Image:
        """Apply sharpening."""
        return img.filter(ImageFilter.SHARPEN)

    def augment(self, img: Image.Image, num_transforms: int = 2,
                seed: int = None) -> Image.Image:
        """
        Apply random transforms to an image.

        Args:
            img: Input PIL Image
            num_transforms: Number of transforms to apply
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        # Ensure RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Select and apply random transforms
        transforms = random.sample(self.transforms, min(num_transforms, len(self.transforms)))

        result = img.copy()
        for transform in transforms:
            result = transform(result)

        return result

    def create_variations(self, img: Image.Image, count: int) -> List[Image.Image]:
        """Create multiple unique variations of an image."""
        variations = []
        for i in range(count):
            # Use different seed for each variation
            augmented = self.augment(img, num_transforms=random.randint(1, 3), seed=i * 1000 + hash(str(img.size)))
            variations.append(augmented)
        return variations


def augment_dataset(data_dir: str, target_per_class: int = 30, max_augments_per_image: int = 3):
    """
    Augment existing images to reach target count per class.

    Args:
        data_dir: Path to data directory
        target_per_class: Target number of images per class
        max_augments_per_image: Maximum augmented versions per original image
    """
    data_path = Path(data_dir)
    augmentor = ImageAugmentor()

    for class_dir in sorted(data_path.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue

        class_name = class_dir.name

        # Get existing images
        existing_images = sorted([
            f for f in class_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_FORMATS
        ])

        existing_count = len(existing_images)
        needed = target_per_class - existing_count

        if needed <= 0:
            print(f"{class_name}: Already has {existing_count} images")
            continue

        print(f"{class_name}: Need {needed} more images (have {existing_count})")

        # Calculate augmentations per image
        augments_per_image = min(max_augments_per_image, (needed // existing_count) + 1)
        created = 0
        next_idx = existing_count + 1

        for img_path in existing_images:
            if created >= needed:
                break

            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')

                    # Create variations
                    num_to_create = min(augments_per_image, needed - created)
                    variations = augmentor.create_variations(img, num_to_create)

                    for var_img in variations:
                        if created >= needed:
                            break

                        # Save with next available index
                        new_path = class_dir / f"{class_name}{next_idx}.png"
                        var_img.save(new_path)
                        next_idx += 1
                        created += 1

            except Exception as e:
                print(f"  Warning: Could not process {img_path.name}: {e}")

        print(f"  Created {created} augmented images")


if __name__ == "__main__":
    import sys

    data_dir = Path(__file__).parent / "data"

    target = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    max_aug = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    print(f"Augmenting dataset to {target} images per class...")
    print(f"Max {max_aug} augmentations per original image\n")

    augment_dataset(data_dir, target, max_aug)
    print("\nDone!")
