"""
Evaluate the isolated-gate classifier on crops from detected schematic gates.
"""

import argparse
import json
from pathlib import Path
from typing import List

from .gate_reclassifier import DEFAULT_GATE_CLASSIFIER_CHECKPOINT, GateCropClassifier
from .pipeline import CircuitAnalysisPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare YOLO gate labels with the isolated-gate classifier on detected crops"
    )
    parser.add_argument("image", type=str, help="Path to a schematic image")
    parser.add_argument("--model", type=str, default=None, help="Optional YOLO detector weights override")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional isolated-gate classifier checkpoint override",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--top-k", type=int, default=3, help="Number of classifier predictions to show")
    parser.add_argument(
        "--full-image",
        action="store_true",
        help="Classify the whole image with the isolated-gate model instead of using detector crops",
    )
    parser.add_argument(
        "--detections-json",
        type=str,
        default=None,
        help="Optional precomputed gate detections JSON instead of live YOLO",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of plain text")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    image_path = Path(args.image)

    pipeline = CircuitAnalysisPipeline(
        model_path=Path(args.model) if args.model else None,
        confidence_threshold=args.conf,
    )
    if args.detections_json:
        detections = pipeline.load_detections_json(Path(args.detections_json))
    else:
        detections = pipeline.detect_gates(image_path)

    classifier = GateCropClassifier(
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else DEFAULT_GATE_CLASSIFIER_CHECKPOINT
    )

    if args.full_image:
        from PIL import Image

        with Image.open(image_path) as image:
            label, confidence, ranked = classifier.classify_image(image.convert("RGB"), top_k=args.top_k)

        payload = {
            "image": str(image_path),
            "mode": "full_image",
            "classifier": {
                "label": label,
                "confidence": confidence,
                "top_k": [{"label": item_label, "confidence": item_conf} for item_label, item_conf in ranked],
            },
        }
        if args.json:
            print(json.dumps(payload, indent=2))
            return

        print(f"Image: {image_path}")
        print(f"Classifier: {label} ({confidence:.2f})")
        for rank in payload["classifier"]["top_k"]:
            print(f"- {rank['label']} {rank['confidence']:.2f}")
        return

    results = classifier.classify_detections(image_path, detections, top_k=args.top_k)

    payload: List[dict] = []
    for item in results:
        payload.append(
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
        )

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"Image: {image_path}")
    print(f"Detected gates: {len(payload)}")
    for item in payload:
        detector = item["detector"]
        classifier_payload = item["classifier"]
        print(
            f"- {item['gate_id']} {detector['label']} ({detector['confidence']:.2f})"
            f" -> {classifier_payload['label']} ({classifier_payload['confidence']:.2f})"
        )
        print(f"  bbox: {[round(value, 1) for value in item['bbox']]}")
        for rank in classifier_payload["top_k"]:
            print(f"  classifier: {rank['label']} {rank['confidence']:.2f}")


if __name__ == "__main__":
    main()
