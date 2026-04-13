"""
Duplicate image checker for the circuit dataset.
Uses perceptual hashing to find similar images.
Can check existing dataset or validate new images before adding.
"""

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

from PIL import Image
import pillow_avif


SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.webp', '.avif', '.gif'}


def get_perceptual_hash(img: Image.Image, hash_size: int = 16) -> str:
    """
    Calculate perceptual hash (pHash) of an image.
    More robust than simple pixel hashing - handles resizing, minor edits, etc.
    """
    # Convert to grayscale and resize
    img = img.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)

    pixels = list(img.getdata())

    # Calculate difference hash (dHash)
    # Compare adjacent pixels
    diff = []
    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]
            diff.append(left > right)

    # Convert to hex string
    hash_value = 0
    for bit in diff:
        hash_value = (hash_value << 1) | int(bit)

    return format(hash_value, f'0{hash_size * hash_size // 4}x')


def hamming_distance(hash1: str, hash2: str) -> int:
    """Calculate Hamming distance between two hashes."""
    if len(hash1) != len(hash2):
        return float('inf')

    # Convert hex to binary and count differences
    b1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
    b2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)

    return sum(c1 != c2 for c1, c2 in zip(b1, b2))


def load_image_hash(img_path: Path) -> Optional[str]:
    """Load an image and return its perceptual hash."""
    try:
        with Image.open(img_path) as img:
            return get_perceptual_hash(img)
    except Exception as e:
        print(f"Warning: Could not process {img_path}: {e}")
        return None


def build_hash_database(data_dir: Path) -> Dict[str, List[Path]]:
    """Build a database of all image hashes in the dataset."""
    hash_db = defaultdict(list)

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue

        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in SUPPORTED_FORMATS:
                img_hash = load_image_hash(img_path)
                if img_hash:
                    hash_db[img_hash].append(img_path)

    return hash_db


def find_duplicates(data_dir: str, similarity_threshold: int = 10) -> List[List[Path]]:
    """
    Find duplicate/similar images in the dataset.

    Args:
        data_dir: Path to data directory
        similarity_threshold: Max Hamming distance to consider as duplicate (0=exact, higher=more lenient)

    Returns:
        List of duplicate groups (each group is a list of similar image paths)
    """
    data_path = Path(data_dir)
    hash_db = build_hash_database(data_path)

    # Find exact duplicates first
    exact_duplicates = [paths for paths in hash_db.values() if len(paths) > 1]

    # Find similar images (within threshold)
    if similarity_threshold > 0:
        hashes = list(hash_db.keys())
        similar_groups = []
        processed: Set[str] = set()

        for i, hash1 in enumerate(hashes):
            if hash1 in processed:
                continue

            group = set(hash_db[hash1])

            for j, hash2 in enumerate(hashes[i + 1:], i + 1):
                if hash2 in processed:
                    continue

                distance = hamming_distance(hash1, hash2)
                if distance <= similarity_threshold:
                    group.update(hash_db[hash2])
                    processed.add(hash2)

            if len(group) > 1:
                similar_groups.append(sorted(group))
                processed.add(hash1)

        return similar_groups

    return exact_duplicates


def check_new_image(new_image_path: str, data_dir: str,
                    similarity_threshold: int = 10) -> List[Tuple[Path, int]]:
    """
    Check if a new image is similar to existing images in the dataset.

    Args:
        new_image_path: Path to the new image to check
        data_dir: Path to data directory
        similarity_threshold: Max Hamming distance to consider as similar

    Returns:
        List of (similar_image_path, distance) tuples
    """
    new_path = Path(new_image_path)
    data_path = Path(data_dir)

    # Get hash of new image
    new_hash = load_image_hash(new_path)
    if not new_hash:
        print(f"Error: Could not process {new_image_path}")
        return []

    # Build database and check against it
    hash_db = build_hash_database(data_path)

    similar = []
    for existing_hash, paths in hash_db.items():
        distance = hamming_distance(new_hash, existing_hash)
        if distance <= similarity_threshold:
            for path in paths:
                similar.append((path, distance))

    return sorted(similar, key=lambda x: x[1])


def check_folder(folder_path: str, data_dir: str, similarity_threshold: int = 10):
    """
    Check all images in a folder against the existing dataset.
    Useful for validating new images before adding them.
    """
    folder = Path(folder_path)
    data_path = Path(data_dir)

    if not folder.exists():
        print(f"Error: Folder {folder_path} does not exist")
        return

    # Get all images in folder
    new_images = [f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS]

    if not new_images:
        print(f"No images found in {folder_path}")
        return

    print(f"Checking {len(new_images)} images against dataset...\n")

    # Build hash database
    hash_db = build_hash_database(data_path)
    print(f"Dataset contains {sum(len(v) for v in hash_db.values())} images\n")

    # Check each new image
    safe_to_add = []
    has_duplicates = []

    for img_path in sorted(new_images):
        new_hash = load_image_hash(img_path)
        if not new_hash:
            continue

        # Find similar images
        similar = []
        for existing_hash, paths in hash_db.items():
            distance = hamming_distance(new_hash, existing_hash)
            if distance <= similarity_threshold:
                similar.extend([(p, distance) for p in paths])

        if similar:
            has_duplicates.append((img_path, similar))
        else:
            safe_to_add.append(img_path)

    # Report results
    print(f"{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")

    print(f"\nSafe to add ({len(safe_to_add)} images):")
    for path in safe_to_add:
        print(f"  + {path.name}")

    if has_duplicates:
        print(f"\nPotential duplicates ({len(has_duplicates)} images):")
        for path, similar in has_duplicates:
            print(f"\n  ! {path.name} is similar to:")
            for sim_path, distance in sorted(similar, key=lambda x: x[1])[:3]:
                print(f"      - {sim_path.parent.name}/{sim_path.name} (distance: {distance})")


def print_report(duplicates: List[List[Path]]):
    """Print a formatted duplicate report."""
    if not duplicates:
        print("No duplicates found!")
        return

    print(f"\nFound {len(duplicates)} group(s) of similar images:\n")

    for i, group in enumerate(duplicates, 1):
        print(f"Group {i}:")
        for path in group:
            print(f"  - {path.parent.name}/{path.name}")
        print()


def main():
    """Main entry point."""
    data_dir = Path(__file__).parent / "data"

    if len(sys.argv) < 2:
        # Default: scan dataset for duplicates
        print("Scanning dataset for duplicates...\n")
        duplicates = find_duplicates(data_dir, similarity_threshold=10)
        print_report(duplicates)

    elif sys.argv[1] == '--check':
        # Check a specific image or folder
        if len(sys.argv) < 3:
            print("Usage: python check_duplicates.py --check <image_or_folder>")
            sys.exit(1)

        target = Path(sys.argv[2])

        if target.is_dir():
            check_folder(target, data_dir)
        elif target.is_file():
            similar = check_new_image(target, data_dir)
            if similar:
                print(f"\n{target.name} is similar to:")
                for path, distance in similar:
                    print(f"  - {path.parent.name}/{path.name} (distance: {distance})")
            else:
                print(f"\n{target.name} is unique - safe to add!")
        else:
            print(f"Error: {target} not found")

    elif sys.argv[1] == '--threshold':
        # Custom threshold
        threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        print(f"Scanning with similarity threshold: {threshold}\n")
        duplicates = find_duplicates(data_dir, similarity_threshold=threshold)
        print_report(duplicates)

    else:
        print("Usage:")
        print("  python check_duplicates.py              # Scan for duplicates")
        print("  python check_duplicates.py --check <path>  # Check image/folder")
        print("  python check_duplicates.py --threshold N   # Custom similarity threshold")


if __name__ == "__main__":
    main()
