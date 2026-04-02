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
        layout = INPUT_LAYOUTS.get(gate.gate_type.upper(), [0.35, 0.65])
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
                point=(output_x, bbox.y1 + bbox.height * 0.5),
            )
        )

        return terminals

    def get_terminal_map(self, gates: List[GateDetection]) -> Dict[str, List[Terminal]]:
        return {gate.gate_id: self.get_terminals(gate) for gate in gates}
