"""
CLI entry point for circuit topology analysis.
"""

import argparse
import json
from pathlib import Path

from .pipeline import CircuitAnalysisPipeline
from .visualization import render_analysis, render_debug_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze circuit topology from a schematic image")
    parser.add_argument("image", type=str, help="Path to a schematic image")
    parser.add_argument("--model", type=str, default=None, help="Path to YOLO gate detector weights")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--detections-json", type=str, default=None, help="Optional precomputed gate detections JSON")
    parser.add_argument("--save-vis", type=str, default=None, help="Optional output image path for visualization")
    parser.add_argument("--save-debug-vis", type=str, default=None, help="Optional output image path for debug visualization")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pipeline = CircuitAnalysisPipeline(model_path=Path(args.model) if args.model else None, confidence_threshold=args.conf)
    detections = None
    if args.detections_json:
        detections = pipeline.load_detections_json(Path(args.detections_json))

    result = pipeline.analyze(Path(args.image), detections=detections)

    if args.save_vis:
        render_analysis(result, Path(args.save_vis))
    if args.save_debug_vis:
        render_debug_analysis(result, Path(args.save_debug_vis))

    payload = {
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

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Gates detected: {len(result.gates)}")
        print(f"Wire segments detected: {len(result.wires)}")
        print(f"Circuit function: {result.classification.label} ({result.classification.confidence:.2f})")
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
