"""
Run the isolated-gate classifier on labeled gate crops from symbol-style schematics.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image

from .gate_reclassifier import DEFAULT_GATE_CLASSIFIER_CHECKPOINT, GateCropClassifier
from .types import BoundingBox


DEFAULT_BENCHMARK_PATH = (
    Path(__file__).resolve().parent.parent
    / "benchmarks"
    / "symbol_gate_crop_benchmark.json"
)


@dataclass(frozen=True)
class CropBenchmarkItem:
    image: str
    label: str
    bbox: BoundingBox
    sample_id: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the isolated-gate classifier on a crop benchmark from symbol-style schematics"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=str(DEFAULT_BENCHMARK_PATH),
        help="Path to the crop benchmark manifest JSON",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_GATE_CLASSIFIER_CHECKPOINT),
        help="Path to the isolated-gate classifier checkpoint",
    )
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.12,
        help="Extra padding ratio to apply around each labeled crop",
    )
    parser.add_argument(
        "--suppress-edge-wires",
        action="store_true",
        help="Remove dark connected components touching the crop border before classification",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of predictions to keep per sample")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of plain text")
    return parser


def load_crop_benchmark(path: Path) -> List[CropBenchmarkItem]:
    raw_items = json.loads(path.read_text())
    items: List[CropBenchmarkItem] = []

    for index, item in enumerate(raw_items):
        bbox_values = item.get("bbox")
        if not isinstance(bbox_values, list) or len(bbox_values) != 4:
            raise ValueError(
                f"Benchmark item {index} in {path} must include a four-value bbox list"
            )

        image = item.get("image")
        label = item.get("label")
        if not image or not label:
            raise ValueError(
                f"Benchmark item {index} in {path} must include image and label fields"
            )

        sample_id = item.get("id", f"sample_{index}")
        items.append(
            CropBenchmarkItem(
                image=image,
                label=str(label).upper(),
                bbox=BoundingBox(*[float(value) for value in bbox_values]),
                sample_id=str(sample_id),
            )
        )

    return items


def evaluate_crop_benchmark(
    benchmark_path: Path,
    checkpoint_path: Path,
    top_k: int = 3,
    crop_padding: float = 0.12,
    suppress_edge_wires: bool = False,
) -> dict:
    root = Path(__file__).resolve().parent.parent
    items = load_crop_benchmark(benchmark_path)

    classifier = GateCropClassifier(
        checkpoint_path=Path(checkpoint_path),
        crop_padding_ratio=crop_padding,
    )
    results: List[dict] = []

    for item in items:
        image_path = root / item.image
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            expanded_bbox = classifier._expanded_crop_box(item.bbox, rgb.width, rgb.height)
            crop = rgb.crop(expanded_bbox.to_int_tuple())
            if suppress_edge_wires:
                predicted, confidence, ranked = classifier.classify_image_with_edge_suppression(
                    crop,
                    top_k=top_k,
                )
            else:
                predicted, confidence, ranked = classifier.classify_image(crop, top_k=top_k)

        results.append(
            {
                "id": item.sample_id,
                "image": item.image,
                "expected": item.label,
                "predicted": predicted,
                "confidence": confidence,
                "correct": predicted == item.label,
                "bbox": [item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2],
                "crop_padding": crop_padding,
                "suppress_edge_wires": suppress_edge_wires,
                "top_k": [{"label": candidate, "confidence": score} for candidate, score in ranked],
            }
        )

    accuracy = sum(1 for item in results if item["correct"]) / max(len(results), 1)
    return {
        "benchmark": str(benchmark_path),
        "checkpoint": str(Path(checkpoint_path)),
        "sample_count": len(results),
        "accuracy": accuracy,
        "results": results,
    }


def main() -> None:
    args = build_parser().parse_args()
    benchmark_path = Path(args.benchmark)
    payload = evaluate_crop_benchmark(
        benchmark_path=benchmark_path,
        checkpoint_path=Path(args.checkpoint),
        top_k=args.top_k,
        crop_padding=args.crop_padding,
        suppress_edge_wires=args.suppress_edge_wires,
    )

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    results = payload["results"]
    accuracy = payload["accuracy"]
    print(f"Benchmark: {benchmark_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples: {len(results)}")
    print(f"Accuracy: {accuracy:.3f}")
    for item in results:
        marker = "OK" if item["correct"] else "FAIL"
        print(
            f"- [{marker}] {item['id']}: expected {item['expected']}, "
            f"predicted {item['predicted']} ({item['confidence']:.2f})"
        )


if __name__ == "__main__":
    main()
