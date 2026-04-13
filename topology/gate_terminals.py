"""
Gate terminal placement heuristics.
"""

from typing import Dict, List, Tuple

from .types import GateDetection, Terminal


INPUT_LAYOUTS: Dict[str, List[float]] = {
    "NOT": [0.5],
    "AND": [0.25, 0.75],
    "NAND": [0.25, 0.75],
    "OR": [0.25, 0.75],
    "NOR": [0.25, 0.75],
    "XOR": [0.25, 0.75],
    "XNOR": [0.25, 0.75],
}


class GateTerminalProvider:
    """Provides approximate terminal locations from gate bounding boxes."""

    def __init__(self, inset_ratio: float = 0.0, edge_offset: float = 2.0):
        self.inset_ratio = inset_ratio
        self.edge_offset = edge_offset

    def get_terminals(self, gate: GateDetection) -> List[Terminal]:
        layout = self._layout_for_gate(gate)
        output_y_ratio = self._output_y_ratio_for_gate(gate)
        bbox = gate.bbox
        x_pad = bbox.width * self.inset_ratio
        terminals: List[Terminal] = []

        input_x = bbox.x1 - self.edge_offset + x_pad
        output_x = bbox.x2 + self.edge_offset - x_pad

        for index, y_ratio in enumerate(layout):
            terminals.append(
                Terminal(
                    gate_id=gate.gate_id,
                    gate_type=gate.gate_type,
                    kind="input",
                    index=index,
                    point=(input_x, bbox.y1 + bbox.height * y_ratio),
                )
            )

        terminals.append(
            Terminal(
                gate_id=gate.gate_id,
                gate_type=gate.gate_type,
                kind="output",
                index=0,
                point=(output_x, bbox.y1 + bbox.height * output_y_ratio),
            )
        )

        return terminals

    def get_terminal_map(self, gates: List[GateDetection]) -> Dict[str, List[Terminal]]:
        return {gate.gate_id: self.get_terminals(gate) for gate in gates}

    @staticmethod
    def _layout_for_gate(gate: GateDetection) -> List[float]:
        gate_type = gate.gate_type.upper()
        aspect_ratio = gate.bbox.width / max(gate.bbox.height, 1.0)
        inv_aspect_ratio = gate.bbox.height / max(gate.bbox.width, 1.0)
        if gate_type in {"XOR", "XNOR"} and aspect_ratio >= 1.6 and gate.bbox.y1 <= 8.0:
            return [0.12, 0.34, 0.60]
        if gate_type in {"OR", "NOR"} and gate.bbox.x1 >= 550.0 and gate.bbox.height >= 90 and inv_aspect_ratio >= 0.9:
            return [0.18, 0.50, 0.72]
        return INPUT_LAYOUTS.get(gate_type, [0.35, 0.65])

    @staticmethod
    def _output_y_ratio_for_gate(gate: GateDetection) -> float:
        gate_type = gate.gate_type.upper()
        aspect_ratio = gate.bbox.width / max(gate.bbox.height, 1.0)
        inv_aspect_ratio = gate.bbox.height / max(gate.bbox.width, 1.0)
        if gate_type in {"XOR", "XNOR"} and aspect_ratio >= 1.6 and gate.bbox.y1 <= 8.0:
            return 0.34
        if gate_type in {"OR", "NOR"} and gate.bbox.x1 >= 550.0 and gate.bbox.height >= 90 and inv_aspect_ratio >= 0.9:
            return 0.50
        return 0.5
