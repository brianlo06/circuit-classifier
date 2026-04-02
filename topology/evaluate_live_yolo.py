"""
Quick evaluation helper for the live YOLO -> topology handoff.
"""

import argparse
import json
from pathlib import Path
from typing import List

from .pipeline import CircuitAnalysisPipeline


DEFAULT_IMAGES = [
    "examples/schematics/half_adder.png",
    "examples/schematics/half_subtractor.png",
    "examples/schematics/full_adder_crossed.png",
    "data/XOR/XOR42.png",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run live YOLO detections through the topology pipeline")
    parser.add_argument("images", nargs="*", default=DEFAULT_IMAGES, help="Images to evaluate")
    parser.add_argument("--model", type=str, default=None, help="Optional YOLO weights path override")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of plain text")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = CircuitAnalysisPipeline(
        model_path=Path(args.model) if args.model else None,
        confidence_threshold=args.conf,
    )

    summaries: List[dict] = []
    for image in args.images:
        image_path = Path(image)
        result = pipeline.analyze(image_path, detections=None)
        summaries.append(
            {
                "image": str(image_path),
                "gate_count": len(result.gates),
                "gates": [
                    {
                        "gate_id": gate.gate_id,
                        "gate_type": gate.gate_type,
                        "confidence": gate.confidence,
                        "bbox": [gate.bbox.x1, gate.bbox.y1, gate.bbox.x2, gate.bbox.y2],
                    }
                    for gate in result.gates
                ],
                "classification": {
                    "label": result.classification.label,
                    "confidence": result.classification.confidence,
                    "reasoning": result.classification.reasoning,
                },
                "warning_count": len(result.warnings),
                "warnings": list(result.warnings),
            }
        )

    if args.json:
        print(json.dumps(summaries, indent=2))
        return

    for item in summaries:
        print(f"Image: {item['image']}")
        print(f"  Gates detected: {item['gate_count']}")
        print(f"  Classification: {item['classification']['label']} ({item['classification']['confidence']:.2f})")
        if item["gates"]:
            for gate in item["gates"]:
                bbox = [round(value, 1) for value in gate["bbox"]]
                print(f"  - {gate['gate_type']} {gate['confidence']:.2f} {bbox}")
        if item["warnings"]:
            print("  Warnings:")
            for warning in item["warnings"]:
                print(f"    - {warning}")


if __name__ == "__main__":
    main()
