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
    parser.add_argument(
        "--use-reclassifier",
        action="store_true",
        help="Run the isolated-gate classifier on supplied detections before topology analysis",
    )
    parser.add_argument(
        "--classifier-checkpoint",
        type=str,
        default=None,
        help="Optional isolated-gate classifier checkpoint override",
    )
    parser.add_argument("--classifier-top-k", type=int, default=3, help="Number of classifier predictions to keep")
    parser.add_argument(
        "--suppress-edge-wires",
        action="store_true",
        help="Suppress dark border-connected strokes before classifying gate crops",
    )
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

    if args.use_reclassifier:
        if detections is None:
            parser.error("--use-reclassifier requires --detections-json for the symbol-style beta path")
        result = pipeline.analyze_symbol_style(
            Path(args.image),
            detections=detections,
            classifier_checkpoint=Path(args.classifier_checkpoint) if args.classifier_checkpoint else None,
            classifier_top_k=args.classifier_top_k,
            suppress_edge_wires=args.suppress_edge_wires,
        )
    else:
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

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Gates detected: {len(result.gates)}")
        print(f"Wire segments detected: {len(result.wires)}")
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
