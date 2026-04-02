"""
End-to-end pipeline for topology analysis.
"""

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from PIL import Image

from .circuit_classifier import CircuitClassifier
from .graph_builder import GraphBuilder
from .types import BoundingBox, GateDetection, PipelineResult
from .wire_detection import WireDetector


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "fixture_demo_best.pt"
)


class CircuitAnalysisPipeline:
    """Runs gate detection, wire detection, graph building, and classification."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.3,
        agnostic_nms: bool = True,
        refine_fixture_boxes: bool = True,
        wire_detector: Optional[WireDetector] = None,
        graph_builder: Optional[GraphBuilder] = None,
        classifier: Optional[CircuitClassifier] = None,
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.agnostic_nms = agnostic_nms
        self.refine_fixture_boxes = refine_fixture_boxes
        self.wire_detector = wire_detector or WireDetector()
        self.graph_builder = graph_builder or GraphBuilder()
        self.classifier = classifier or CircuitClassifier()

    def analyze(
        self,
        image_path: Path,
        detections: Optional[Sequence[GateDetection]] = None,
    ) -> PipelineResult:
        image_path = Path(image_path)
        with Image.open(image_path) as image:
            image_size = image.size

        warnings: List[str] = []
        gates = list(detections) if detections is not None else self.detect_gates(image_path)
        if not gates:
            warnings.append("No gates detected")

        wire_result = self.wire_detector.detect(image_path, gates)
        graph_result = self.graph_builder.build_graph(gates, wire_result.components)
        warnings.extend(graph_result.warnings)

        classification = self.classifier.classify(graph_result.graph)

        return PipelineResult(
            image_path=image_path,
            image_size=image_size,
            gates=gates,
            wires=wire_result.segments,
            wire_components=wire_result.components,
            terminals=graph_result.terminals,
            component_matches=graph_result.component_matches,
            wire_mask=wire_result.mask,
            graph=graph_result.graph,
            classification=classification,
            warnings=warnings,
        )

    def analyze_batch(self, image_paths: Iterable[Path]) -> List[PipelineResult]:
        return [self.analyze(Path(image_path)) for image_path in image_paths]

    def detect_gates(self, image_path: Path) -> List[GateDetection]:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is required for YOLO gate detection") from exc

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found at {self.model_path}")

        model = YOLO(str(self.model_path))
        results = model.predict(
            source=str(image_path),
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            agnostic_nms=self.agnostic_nms,
            verbose=False,
        )

        detections: List[GateDetection] = []
        for result in results:
            names = result.names
            for index, box in enumerate(result.boxes):
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                detections.append(
                    GateDetection(
                        gate_id=f"gate_{index}",
                        gate_type=str(names[cls_id]).upper(),
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        confidence=confidence,
                    )
                )

        detections.sort(key=lambda item: (item.center[0], item.center[1]))
        detections = [
            GateDetection(
                gate_id=f"gate_{index}",
                gate_type=item.gate_type,
                bbox=item.bbox,
                confidence=item.confidence,
            )
            for index, item in enumerate(detections)
        ]
        if self.refine_fixture_boxes:
            detections = self._refine_detected_gate_boxes(image_path, detections)
        return detections

    def _refine_detected_gate_boxes(self, image_path: Path, detections: Sequence[GateDetection]) -> List[GateDetection]:
        with Image.open(image_path) as image:
            grayscale = image.convert("L")
            image_width, image_height = grayscale.size
            refined: List[GateDetection] = []
            for detection in detections:
                bbox = self._refine_fixture_box(grayscale, detection.bbox, image_width, image_height)
                refined.append(
                    GateDetection(
                        gate_id=detection.gate_id,
                        gate_type=detection.gate_type,
                        bbox=bbox,
                        confidence=detection.confidence,
                    )
                )
            return refined

    @staticmethod
    def _refine_fixture_box(
        grayscale: Image.Image,
        bbox: BoundingBox,
        image_width: int,
        image_height: int,
    ) -> BoundingBox:
        x1, y1, x2, y2 = bbox.to_int_tuple()
        x1 = max(0, min(image_width - 1, x1))
        y1 = max(0, min(image_height - 1, y1))
        x2 = max(x1 + 1, min(image_width, x2))
        y2 = max(y1 + 1, min(image_height, y2))
        crop = grayscale.crop((x1, y1, x2, y2))

        fill_mask = [
            [1 if 230 <= crop.getpixel((x, y)) <= 252 else 0 for x in range(crop.width)]
            for y in range(crop.height)
        ]
        col_counts = [sum(fill_mask[y][x] for y in range(crop.height)) for x in range(crop.width)]
        row_counts = [sum(fill_mask[y][x] for x in range(crop.width)) for y in range(crop.height)]

        col_threshold = max(8, int(crop.height * 0.35))
        row_threshold = max(8, int(crop.width * 0.35))

        left = 0
        while left < len(col_counts) and col_counts[left] < col_threshold:
            left += 1
        right = len(col_counts) - 1
        while right >= 0 and col_counts[right] < col_threshold:
            right -= 1
        top = 0
        while top < len(row_counts) and row_counts[top] < row_threshold:
            top += 1
        bottom = len(row_counts) - 1
        while bottom >= 0 and row_counts[bottom] < row_threshold:
            bottom -= 1

        if left >= right or top >= bottom:
            return bbox

        refined_width = right - left + 1
        refined_height = bottom - top + 1
        if refined_width < crop.width * 0.45 or refined_height < crop.height * 0.45:
            return bbox

        pad = 2
        return BoundingBox(
            x1=max(0, x1 + left - pad),
            y1=max(0, y1 + top - pad),
            x2=min(image_width, x1 + right + 1 + pad),
            y2=min(image_height, y1 + bottom + 1 + pad),
        )

    @staticmethod
    def load_detections_json(path: Path) -> List[GateDetection]:
        payload = json.loads(Path(path).read_text())
        detections = []
        for index, item in enumerate(payload):
            detections.append(
                GateDetection(
                    gate_id=item.get("gate_id", f"gate_{index}"),
                    gate_type=item["gate_type"].upper(),
                    bbox=BoundingBox(*item["bbox"]),
                    confidence=float(item.get("confidence", 1.0)),
                )
            )
        return detections
