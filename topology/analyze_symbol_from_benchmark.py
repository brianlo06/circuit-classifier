"""
Analyze a real symbol-style schematic using benchmark-provided gate boxes.
"""

import argparse
import json
from pathlib import Path
from typing import List

from .pipeline import CircuitAnalysisPipeline
from .types import BoundingBox, GateDetection
from .visualization import render_analysis, render_debug_analysis


DEFAULT_BENCHMARK_PATH = (
    Path(__file__).resolve().parent.parent
    / "benchmarks"
    / "symbol_gate_real_crop_benchmark.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a symbol-style schematic using manually verified benchmark boxes"
    )
    parser.add_argument("image", type=str, help="Image path exactly as stored in the benchmark manifest")
    parser.add_argument(
        "--benchmark",
        type=str,
        default=str(DEFAULT_BENCHMARK_PATH),
        help="Path to the real symbol-style benchmark manifest",
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=str,
        default=None,
        help="Optional isolated-gate classifier checkpoint override",
    )
    parser.add_argument("--classifier-top-k", type=int, default=3)
    parser.add_argument(
        "--suppress-edge-wires",
        action="store_true",
        help="Suppress dark border-connected strokes before classifying gate crops",
    )
    parser.add_argument("--save-vis", type=str, default=None)
    parser.add_argument("--save-debug-vis", type=str, default=None)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output")
    return parser


def load_benchmark_detections(benchmark_path: Path, image_path: str) -> List[GateDetection]:
    entries = json.loads(benchmark_path.read_text())
    matches = [item for item in entries if item.get("image") == image_path]
    if not matches:
        raise ValueError(f"No benchmark entries found for image {image_path}")

    return [
        GateDetection(
            gate_id=item.get("id", f"gate_{index}"),
            gate_type=str(item["label"]).upper(),
            bbox=BoundingBox(*[float(value) for value in item["bbox"]]),
            confidence=1.0,
        )
        for index, item in enumerate(matches)
    ]


def build_payload(result) -> dict:
    return {
        "image_path": str(result.image_path),
        "image_size": list(result.image_size),
        "gates": [
            {
                "gate_id": gate.gate_id,
                "gate_type": gate.gate_type,
                "bbox": [gate.bbox.x1, gate.bbox.y1, gate.bbox.x2, gate.bbox.y2],
                "confidence": gate.confidence,
            }
            for gate in result.gates
        ],
        "reclassifications": [
            {
                "gate_id": item.gate_id,
                "bbox": [item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2],
                "detector": {
                    "label": item.detector_label,
                    "confidence": item.detector_confidence,
                },
                "classifier": {
                    "label": item.classifier_label,
                    "confidence": item.classifier_confidence,
                    "top_k": [
                        {"label": label, "confidence": confidence}
                        for label, confidence in item.top_k
                    ],
                },
            }
            for item in result.reclassifications
        ],
        "wire_count": len(result.wires),
        "graph": result.graph.to_dict(),
        "classification": {
            "label": result.classification.label,
            "confidence": result.classification.confidence,
            "reasoning": result.classification.reasoning,
            "truth_table": result.classification.truth_table,
            "expressions": result.classification.expressions,
        },
        "warnings": result.warnings,
    }


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parent.parent
    image_path = project_root / args.image
    detections = load_benchmark_detections(Path(args.benchmark), args.image)

    pipeline = CircuitAnalysisPipeline()
    result = pipeline.analyze_symbol_style(
        image_path=image_path,
        detections=detections,
        classifier_checkpoint=Path(args.classifier_checkpoint) if args.classifier_checkpoint else None,
        classifier_top_k=args.classifier_top_k,
        suppress_edge_wires=args.suppress_edge_wires,
    )

    if args.save_vis:
        render_analysis(result, Path(args.save_vis))
    if args.save_debug_vis:
        render_debug_analysis(result, Path(args.save_debug_vis))

    payload = build_payload(result)
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"Image: {args.image}")
    print(f"Benchmark detections: {len(detections)}")
    print(f"Circuit function: {result.classification.label} ({result.classification.confidence:.2f})")
    if result.reclassifications:
        print("Gate relabeling:")
        for item in result.reclassifications:
            print(
                f"  {item.gate_id}: {item.detector_label} ({item.detector_confidence:.2f})"
                f" -> {item.classifier_label} ({item.classifier_confidence:.2f})"
            )
    if result.classification.expressions:
        print("Outputs:")
        for name, expression in result.classification.expressions.items():
            print(f"  {name} = {expression}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
