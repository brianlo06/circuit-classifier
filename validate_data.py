"""
Validation script for Circuit Topology Classifier dataset.
Checks image quality, dimensions, and reports issues.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from PIL import Image
import pillow_avif  # Required for AVIF support


@dataclass
class ImageInfo:
    """Information about a single image."""
    path: Path
    width: int = 0
    height: int = 0
    format: str = ""
    mode: str = ""
    file_size_kb: float = 0
    is_valid: bool = True
    error: Optional[str] = None


@dataclass
class ValidationReport:
    """Aggregated validation report."""
    total_images: int = 0
    valid_images: int = 0
    invalid_images: int = 0
    issues: List[Tuple[str, str]] = field(default_factory=list)  # (path, issue)
    class_distribution: Dict[str, int] = field(default_factory=dict)
    format_distribution: Dict[str, int] = field(default_factory=dict)
    dimension_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    warnings: List[Tuple[str, str]] = field(default_factory=list)


SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.webp', '.avif', '.gif'}

# Validation thresholds
MIN_DIMENSION = 32       # Minimum width or height in pixels
MAX_DIMENSION = 4096     # Maximum width or height in pixels
MIN_FILE_SIZE_KB = 1     # Minimum file size in KB
MAX_FILE_SIZE_KB = 10240 # Maximum file size in KB (10 MB)
RECOMMENDED_MIN_DIM = 100  # Recommended minimum for good quality


def validate_image(img_path: Path) -> ImageInfo:
    """Validate a single image and return its info."""
    info = ImageInfo(path=img_path)

    # Check file size
    try:
        info.file_size_kb = img_path.stat().st_size / 1024
    except OSError as e:
        info.is_valid = False
        info.error = f"Cannot read file: {e}"
        return info

    # Try to open and inspect the image
    try:
        with Image.open(img_path) as img:
            info.width, info.height = img.size
            info.format = img.format or "Unknown"
            info.mode = img.mode

            # Verify image can be fully loaded (catches truncated files)
            img.load()

    except Exception as e:
        info.is_valid = False
        info.error = f"Cannot open image: {e}"
        return info

    return info


def validate_dataset(data_dir: str, verbose: bool = True) -> ValidationReport:
    """
    Validate all images in the dataset.

    Args:
        data_dir: Path to data directory with class subfolders
        verbose: Whether to print progress

    Returns:
        ValidationReport with all findings
    """
    data_path = Path(data_dir)
    report = ValidationReport()

    # Get all class directories
    class_dirs = sorted([
        d for d in data_path.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])

    if not class_dirs:
        print(f"No class directories found in {data_dir}")
        return report

    # Collect dimension stats per class
    width_stats = defaultdict(list)
    height_stats = defaultdict(list)

    for class_dir in class_dirs:
        class_name = class_dir.name
        report.class_distribution[class_name] = 0

        if verbose:
            print(f"\nValidating class: {class_name}")

        # Get all images in class directory
        images = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_FORMATS
        ]

        for img_path in sorted(images):
            report.total_images += 1
            info = validate_image(img_path)

            # Track format distribution
            ext = img_path.suffix.lower()
            report.format_distribution[ext] = report.format_distribution.get(ext, 0) + 1

            if not info.is_valid:
                report.invalid_images += 1
                report.issues.append((str(img_path), info.error))
                if verbose:
                    print(f"  ERROR: {img_path.name} - {info.error}")
                continue

            report.valid_images += 1
            report.class_distribution[class_name] += 1

            # Track dimensions
            width_stats[class_name].append(info.width)
            height_stats[class_name].append(info.height)

            # Check for warnings (valid but potentially problematic)
            warnings = []

            if info.width < MIN_DIMENSION or info.height < MIN_DIMENSION:
                warnings.append(f"Very small: {info.width}x{info.height}")
            elif info.width < RECOMMENDED_MIN_DIM or info.height < RECOMMENDED_MIN_DIM:
                warnings.append(f"Small dimensions: {info.width}x{info.height}")

            if info.width > MAX_DIMENSION or info.height > MAX_DIMENSION:
                warnings.append(f"Very large: {info.width}x{info.height}")

            if info.file_size_kb < MIN_FILE_SIZE_KB:
                warnings.append(f"Tiny file: {info.file_size_kb:.1f} KB")

            if info.file_size_kb > MAX_FILE_SIZE_KB:
                warnings.append(f"Large file: {info.file_size_kb:.1f} KB")

            # Check aspect ratio (very extreme ratios may indicate issues)
            aspect_ratio = max(info.width, info.height) / max(min(info.width, info.height), 1)
            if aspect_ratio > 5:
                warnings.append(f"Extreme aspect ratio: {aspect_ratio:.1f}:1")

            for warning in warnings:
                report.warnings.append((str(img_path), warning))
                if verbose:
                    print(f"  WARNING: {img_path.name} - {warning}")

    # Calculate dimension statistics
    for class_name in report.class_distribution.keys():
        if width_stats[class_name]:
            widths = width_stats[class_name]
            heights = height_stats[class_name]
            report.dimension_stats[class_name] = {
                'min_width': min(widths),
                'max_width': max(widths),
                'avg_width': sum(widths) / len(widths),
                'min_height': min(heights),
                'max_height': max(heights),
                'avg_height': sum(heights) / len(heights),
            }

    return report


def print_report(report: ValidationReport):
    """Print a formatted validation report."""
    print("\n" + "=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)

    # Summary
    print(f"\n{'SUMMARY':-^60}")
    print(f"  Total images scanned: {report.total_images}")
    print(f"  Valid images: {report.valid_images}")
    print(f"  Invalid images: {report.invalid_images}")
    print(f"  Warnings: {len(report.warnings)}")

    # Class distribution
    print(f"\n{'CLASS DISTRIBUTION':-^60}")
    for class_name, count in sorted(report.class_distribution.items()):
        bar = "#" * min(count, 40)
        print(f"  {class_name:<10} {count:>4} {bar}")

    # Format distribution
    print(f"\n{'FORMAT DISTRIBUTION':-^60}")
    for fmt, count in sorted(report.format_distribution.items(), key=lambda x: -x[1]):
        print(f"  {fmt:<10} {count:>4}")

    # Dimension statistics
    print(f"\n{'DIMENSION STATISTICS (per class)':-^60}")
    for class_name, stats in sorted(report.dimension_stats.items()):
        print(f"\n  {class_name}:")
        print(f"    Width:  min={stats['min_width']:.0f}, max={stats['max_width']:.0f}, avg={stats['avg_width']:.0f}")
        print(f"    Height: min={stats['min_height']:.0f}, max={stats['max_height']:.0f}, avg={stats['avg_height']:.0f}")

    # Issues (errors)
    if report.issues:
        print(f"\n{'ERRORS (invalid images)':-^60}")
        for path, issue in report.issues:
            print(f"  {path}")
            print(f"    -> {issue}")

    # Warnings
    if report.warnings:
        print(f"\n{'WARNINGS':-^60}")
        for path, warning in report.warnings[:20]:  # Limit to first 20
            rel_path = Path(path).name
            print(f"  {rel_path}: {warning}")
        if len(report.warnings) > 20:
            print(f"  ... and {len(report.warnings) - 20} more warnings")

    # Recommendations
    print(f"\n{'RECOMMENDATIONS':-^60}")

    if report.invalid_images > 0:
        print(f"  - Fix or remove {report.invalid_images} invalid image(s)")

    class_counts = list(report.class_distribution.values())
    if class_counts and max(class_counts) > min(class_counts) * 2:
        print("  - Consider balancing class sizes (current imbalance detected)")

    small_warnings = [w for w in report.warnings if "Small" in w[1] or "small" in w[1]]
    if len(small_warnings) > 5:
        print(f"  - {len(small_warnings)} images have small dimensions; consider upscaling or replacing")

    if all(c >= 30 for c in class_counts if c > 0):
        print("  - Dataset size looks good for initial training")
    elif class_counts:
        print("  - Consider adding more images (aim for 30+ per class)")

    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    # Determine data directory
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = Path(__file__).parent / "data"

    print(f"Validating dataset at: {data_dir}")

    # Run validation
    report = validate_dataset(data_dir, verbose=True)

    # Print report
    print_report(report)

    # Exit with error code if there are issues
    if report.invalid_images > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
