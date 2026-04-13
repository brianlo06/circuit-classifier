"""
Evaluate heuristic symbol-style gate proposals against the real crop benchmark.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image

from .analyze_symbol_with_proposals import _build_augmented_proposals
from .symbol_gate_proposer import SymbolGateProposer


DEFAULT_BENCHMARK_PATH = (
    Path(__file__).resolve().parent.parent
    / "benchmarks"
    / "symbol_gate_real_crop_benchmark.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the heuristic symbol gate proposer on the real benchmark")
    parser.add_argument("--benchmark", type=str, default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--iou-threshold", type=float, default=0.2)
    parser.add_argument("--json", action="store_true")
    return parser


def load_grouped_benchmark(path: Path) -> Dict[str, List[dict]]:
    items = json.loads(path.read_text())
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for item in items:
        grouped[item["image"]].append(item)
    return dict(grouped)


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    return intersection / max(area_a + area_b - intersection, 1e-6)


def evaluate(benchmark_path: Path, iou_threshold: float) -> dict:
    root = Path(__file__).resolve().parent.parent
    grouped = load_grouped_benchmark(benchmark_path)
    proposer = SymbolGateProposer()

    image_results = []
    total_matches = 0
    total_expected = 0
    total_proposals = 0

    for image, entries in sorted(grouped.items()):
        proposals = proposer.propose(root / image)
        with Image.open(root / image) as source_image:
            image_width, image_height = source_image.size
        proposals += _build_augmented_proposals(
            proposals=proposals,
            proposer=proposer,
            image_width=image_width,
            image_height=image_height,
            aggressive=True,
        )
        proposal_boxes = [
            [item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2]
            for item in proposals
        ]
        matches = []
        for entry in entries:
            best_iou = 0.0
            best_index = -1
            for index, proposal_box in enumerate(proposal_boxes):
                score = bbox_iou(entry["bbox"], proposal_box)
                if score > best_iou:
                    best_iou = score
                    best_index = index
            matched = best_iou >= iou_threshold
            if matched:
                total_matches += 1
            total_expected += 1
            matches.append(
                {
                    "id": entry["id"],
                    "label": entry["label"],
                    "matched": matched,
                    "best_iou": best_iou,
                    "best_proposal_index": best_index,
                }
            )
        total_proposals += len(proposals)
        image_results.append(
            {
                "image": image,
                "proposal_count": len(proposals),
                "expected_count": len(entries),
                "matched_count": sum(1 for item in matches if item["matched"]),
                "matches": matches,
            }
        )

    recall = total_matches / max(total_expected, 1)
    avg_proposals = total_proposals / max(len(image_results), 1)
    return {
        "benchmark": str(benchmark_path),
        "iou_threshold": iou_threshold,
        "image_count": len(image_results),
        "total_expected": total_expected,
        "total_matches": total_matches,
        "recall": recall,
        "avg_proposals_per_image": avg_proposals,
        "results": image_results,
    }


def main() -> None:
    args = build_parser().parse_args()
    payload = evaluate(Path(args.benchmark), args.iou_threshold)
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    print(f"Benchmark: {payload['benchmark']}")
    print(f"Images: {payload['image_count']}")
    print(f"Expected gates: {payload['total_expected']}")
    print(f"Matched gates: {payload['total_matches']}")
    print(f"Recall: {payload['recall']:.3f}")
    print(f"Avg proposals/image: {payload['avg_proposals_per_image']:.2f}")
    for result in payload["results"]:
        print(
            f"- {result['image']}: matched {result['matched_count']}/{result['expected_count']} "
            f"with {result['proposal_count']} proposal(s)"
        )


if __name__ == "__main__":
    main()
