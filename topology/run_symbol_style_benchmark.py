"""
Run the isolated-gate classifier on a curated symbol-style full-image benchmark.
"""

import argparse
import json
from pathlib import Path
from typing import List

from PIL import Image

from .gate_reclassifier import DEFAULT_GATE_CLASSIFIER_CHECKPOINT, GateCropClassifier


DEFAULT_BENCHMARK_PATH = (
    Path(__file__).resolve().parent.parent
    / "benchmarks"
    / "symbol_gate_full_image_benchmark.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the isolated-gate classifier on the symbol-style benchmark")
    parser.add_argument(
        "--benchmark",
        type=str,
        default=str(DEFAULT_BENCHMARK_PATH),
        help="Path to the benchmark manifest JSON",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_GATE_CLASSIFIER_CHECKPOINT),
        help="Path to the isolated-gate classifier checkpoint",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of predictions to keep per sample")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of plain text")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parent.parent
    benchmark_path = Path(args.benchmark)
    items = json.loads(benchmark_path.read_text())

    classifier = GateCropClassifier(checkpoint_path=Path(args.checkpoint))
    results: List[dict] = []

    for item in items:
        image_path = root / item["image"]
        expected = item["label"].upper()
        with Image.open(image_path) as image:
            label, confidence, ranked = classifier.classify_image(image.convert("RGB"), top_k=args.top_k)
        results.append(
            {
                "image": item["image"],
                "expected": expected,
                "predicted": label,
                "confidence": confidence,
                "correct": label == expected,
                "top_k": [{"label": candidate, "confidence": score} for candidate, score in ranked],
            }
        )

    accuracy = sum(1 for item in results if item["correct"]) / max(len(results), 1)
    payload = {
        "benchmark": str(benchmark_path),
        "checkpoint": str(Path(args.checkpoint)),
        "sample_count": len(results),
        "accuracy": accuracy,
        "results": results,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"Benchmark: {benchmark_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples: {len(results)}")
    print(f"Accuracy: {accuracy:.3f}")
    for item in results:
        marker = "OK" if item["correct"] else "FAIL"
        print(
            f"- [{marker}] {item['image']}: expected {item['expected']}, "
            f"predicted {item['predicted']} ({item['confidence']:.2f})"
        )


if __name__ == "__main__":
    main()
