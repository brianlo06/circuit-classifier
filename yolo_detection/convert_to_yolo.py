"""
Convert classification dataset to YOLO detection format.

For single-object classification images, we estimate bounding boxes
by finding the non-white region (the gate symbol) in each image.
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image
import numpy as np

# Try to import pillow-avif-plugin for AVIF support
try:
    import pillow_avif
except ImportError:
    pass


def find_bounding_box(image_path: Path, padding: float = 0.05) -> tuple:
    """
    Find bounding box of the non-background content in an image.
    Returns (center_x, center_y, width, height) normalized to 0-1.

    Args:
        image_path: Path to the image
        padding: Extra padding around detected content (fraction of image size)
    """
    try:
        img = Image.open(image_path).convert('RGBA')
    except Exception as e:
        print(f"Error opening {image_path}: {e}")
        return None

    # Convert to numpy array
    arr = np.array(img)
    width, height = img.size

    # Find non-white and non-transparent pixels
    # Check if pixel is "content" (not white background, not transparent)
    if arr.shape[2] == 4:  # RGBA
        alpha = arr[:, :, 3]
        rgb = arr[:, :, :3]
        # Content = not transparent AND not white
        is_content = (alpha > 128) & (np.any(rgb < 240, axis=2))
    else:  # RGB
        rgb = arr[:, :, :3]
        is_content = np.any(rgb < 240, axis=2)

    # Find rows and columns with content
    rows_with_content = np.any(is_content, axis=1)
    cols_with_content = np.any(is_content, axis=0)

    if not np.any(rows_with_content) or not np.any(cols_with_content):
        # No content found, use full image
        return (0.5, 0.5, 1.0, 1.0)

    # Get bounding box
    row_indices = np.where(rows_with_content)[0]
    col_indices = np.where(cols_with_content)[0]

    y_min, y_max = row_indices[0], row_indices[-1]
    x_min, x_max = col_indices[0], col_indices[-1]

    # Add padding
    pad_x = int(width * padding)
    pad_y = int(height * padding)

    x_min = max(0, x_min - pad_x)
    x_max = min(width - 1, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(height - 1, y_max + pad_y)

    # Convert to YOLO format (center_x, center_y, width, height) normalized
    box_width = x_max - x_min
    box_height = y_max - y_min
    center_x = (x_min + box_width / 2) / width
    center_y = (y_min + box_height / 2) / height
    norm_width = box_width / width
    norm_height = box_height / height

    return (center_x, center_y, norm_width, norm_height)


def convert_dataset(
    source_dir: str = "../data",
    output_dir: str = "datasets/circuit_components",
    train_split: float = 0.8,
    seed: int = 42
):
    """
    Convert classification dataset to YOLO detection format.

    Args:
        source_dir: Path to classification dataset (with class subdirs)
        output_dir: Path to output YOLO dataset
        train_split: Fraction of data for training
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Class names in order (must match data.yaml)
    class_names = ['AND', 'NAND', 'NOR', 'NOT', 'OR', 'XNOR', 'XOR']
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    # Supported extensions
    extensions = {'.png', '.jpg', '.jpeg', '.webp', '.avif', '.gif'}

    # Collect all images
    all_images = []
    for class_name in class_names:
        class_dir = source_path / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist")
            continue

        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in extensions:
                all_images.append((img_file, class_name))

    print(f"Found {len(all_images)} images across {len(class_names)} classes")

    # Shuffle and split
    random.shuffle(all_images)
    split_idx = int(len(all_images) * train_split)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    print(f"Train: {len(train_images)}, Val: {len(val_images)}")

    # Process images
    stats = {'train': 0, 'val': 0, 'errors': 0}

    for split_name, image_list in [('train', train_images), ('val', val_images)]:
        images_dir = output_path / 'images' / split_name
        labels_dir = output_path / 'labels' / split_name

        for img_path, class_name in image_list:
            class_id = class_to_id[class_name]

            # Find bounding box
            bbox = find_bounding_box(img_path)
            if bbox is None:
                stats['errors'] += 1
                continue

            center_x, center_y, width, height = bbox

            # Copy image (convert to jpg for consistency)
            new_name = f"{class_name}_{img_path.stem}"

            try:
                # Open and save as RGB jpg
                img = Image.open(img_path).convert('RGB')
                output_img_path = images_dir / f"{new_name}.jpg"
                img.save(output_img_path, 'JPEG', quality=95)

                # Write label file
                output_label_path = labels_dir / f"{new_name}.txt"
                with open(output_label_path, 'w') as f:
                    f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

                stats[split_name] += 1

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                stats['errors'] += 1

    print(f"\nConversion complete:")
    print(f"  Train images: {stats['train']}")
    print(f"  Val images: {stats['val']}")
    print(f"  Errors: {stats['errors']}")
    print(f"\nOutput directory: {output_path.absolute()}")


def visualize_sample(output_dir: str = "datasets/circuit_components", num_samples: int = 5):
    """
    Visualize a few samples to verify bounding boxes are correct.
    Saves visualization images to output_dir/visualizations/
    """
    try:
        from PIL import ImageDraw
    except ImportError:
        print("PIL not available for visualization")
        return

    output_path = Path(output_dir)
    vis_dir = output_path / 'visualizations'
    vis_dir.mkdir(exist_ok=True)

    class_names = ['AND', 'NAND', 'NOR', 'NOT', 'OR', 'XNOR', 'XOR']

    images_dir = output_path / 'images' / 'train'
    labels_dir = output_path / 'labels' / 'train'

    image_files = list(images_dir.glob('*.jpg'))[:num_samples]

    for img_file in image_files:
        label_file = labels_dir / f"{img_file.stem}.txt"

        if not label_file.exists():
            continue

        # Load image
        img = Image.open(img_file)
        draw = ImageDraw.Draw(img)
        width, height = img.size

        # Load labels
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])

                # Convert to pixel coordinates
                x1 = int((cx - w/2) * width)
                y1 = int((cy - h/2) * height)
                x2 = int((cx + w/2) * width)
                y2 = int((cy + h/2) * height)

                # Draw box
                draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                draw.text((x1, y1 - 15), class_names[class_id], fill='red')

        # Save
        vis_path = vis_dir / f"vis_{img_file.name}"
        img.save(vis_path)
        print(f"Saved visualization: {vis_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert classification dataset to YOLO format")
    parser.add_argument("--source", type=str, default="../data", help="Source classification dataset")
    parser.add_argument("--output", type=str, default="datasets/circuit_components", help="Output YOLO dataset")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train/val split ratio")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization samples")
    parser.add_argument("--num-vis", type=int, default=5, help="Number of visualizations to generate")

    args = parser.parse_args()

    convert_dataset(
        source_dir=args.source,
        output_dir=args.output,
        train_split=args.train_split
    )

    if args.visualize:
        visualize_sample(output_dir=args.output, num_samples=args.num_vis)
