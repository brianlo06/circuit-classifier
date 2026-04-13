"""
Shared data types for circuit topology analysis.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


Point = Tuple[float, float]


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in image coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> Point:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def expanded(self, pad: float) -> "BoundingBox":
        return BoundingBox(self.x1 - pad, self.y1 - pad, self.x2 + pad, self.y2 + pad)

    def to_int_tuple(self) -> Tuple[int, int, int, int]:
        return (int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2)))


@dataclass(frozen=True)
class GateDetection:
    """Detected logic gate from object detection."""

    gate_id: str
    gate_type: str
    bbox: BoundingBox
    confidence: float = 1.0

    @property
    def center(self) -> Point:
        return self.bbox.center


@dataclass(frozen=True)
class GateReclassification:
    """Classifier output for a detected gate crop."""

    gate_id: str
    detector_label: str
    detector_confidence: float
    classifier_label: str
    classifier_confidence: float
    bbox: BoundingBox
    top_k: List[Tuple[str, float]]


@dataclass(frozen=True)
class Terminal:
    """Input or output pin on a gate."""

    gate_id: str
    gate_type: str
    kind: str
    index: int
    point: Point


@dataclass(frozen=True)
class WireSegment:
    """Detected wire segment represented as a straight line."""

    start: Point
    end: Point
    length: float


@dataclass(frozen=True)
class WireComponent:
    """Connected collection of wire segments."""

    component_id: str
    segments: List[WireSegment]
    points: List[Point]


@dataclass
class GateNode:
    """Graph node for a logic gate."""

    gate_id: str
    gate_type: str
    bbox: BoundingBox
    confidence: float = 1.0


@dataclass
class Connection:
    """Directed edge from one gate output to another gate input."""

    source_gate: str
    target_gate: str
    target_input_index: int
    source_output_index: int = 0
    component_id: Optional[str] = None


@dataclass
class PrimaryInput:
    """External input feeding one or more gate terminals."""

    input_id: str
    targets: List[Tuple[str, int]]
    anchor: Optional[Point] = None


@dataclass
class PrimaryOutput:
    """External output driven by a gate terminal."""

    output_id: str
    source_gate: str
    source_output_index: int = 0
    anchor: Optional[Point] = None


@dataclass
class ClassificationResult:
    """Result of circuit-level classification."""

    label: str
    confidence: float
    reasoning: str
    truth_table: List[Dict[str, int]] = field(default_factory=list)
    expressions: Dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Output bundle from the end-to-end topology pipeline."""

    image_path: Path
    image_size: Tuple[int, int]
    gates: List[GateDetection]
    wires: List[WireSegment]
    graph: Any
    classification: ClassificationResult
    wire_components: List[WireComponent]
    terminals: List[Terminal]
    component_matches: Dict[str, List[Terminal]]
    wire_mask: Optional[Any] = None
    reclassifications: List[GateReclassification] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
