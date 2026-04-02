"""
Generate clean schematic fixtures and matching detection JSON for topology tests.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from .gate_terminals import GateTerminalProvider
from .types import BoundingBox, GateDetection


ROOT = Path(__file__).resolve().parent.parent
SCHEMATICS_DIR = ROOT / "examples" / "schematics"
DETECTIONS_DIR = ROOT / "examples" / "detections"


def _font():
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        return ImageFont.load_default()


def _draw_gate(draw: ImageDraw.ImageDraw, gate: GateDetection, label: str) -> None:
    x1, y1, x2, y2 = gate.bbox.to_int_tuple()
    draw.rectangle((x1, y1, x2, y2), outline="black", fill="#f6f6f6", width=2)
    text_x = x1 + 8
    text_y = y1 + max(6, (y2 - y1) // 2 - 8)
    draw.text((text_x, text_y), label, font=_font(), fill="black")


def _draw_wire(draw: ImageDraw.ImageDraw, points: Sequence[Tuple[int, int]]) -> None:
    for start, end in zip(points, points[1:]):
        draw.line([start, end], fill="black", width=2)


def _save_fixture(name: str, size: Tuple[int, int], gates: List[GateDetection], wires: List[List[Tuple[int, int]]]) -> None:
    SCHEMATICS_DIR.mkdir(parents=True, exist_ok=True)
    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)

    # Draw gates first (with fill), then wires on top so wire pixels are visible
    for gate in gates:
        _draw_gate(draw, gate, gate.gate_type)

    for wire in wires:
        _draw_wire(draw, wire)

    image.save(SCHEMATICS_DIR / f"{name}.png")

    payload = [
        {
            "gate_id": gate.gate_id,
            "gate_type": gate.gate_type,
            "bbox": [gate.bbox.x1, gate.bbox.y1, gate.bbox.x2, gate.bbox.y2],
            "confidence": gate.confidence,
        }
        for gate in gates
    ]
    (DETECTIONS_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2))


def _terminal_map(gates: Sequence[GateDetection]) -> Dict[str, Dict[str, Tuple[int, int]]]:
    provider = GateTerminalProvider()
    mapping: Dict[str, Dict[str, Tuple[int, int]]] = {}
    for gate in gates:
        terminals = provider.get_terminals(gate)
        gate_map: Dict[str, Tuple[int, int]] = {}
        for terminal in terminals:
            key = f"{terminal.kind}{terminal.index}"
            gate_map[key] = (int(round(terminal.point[0])), int(round(terminal.point[1])))
        mapping[gate.gate_id] = gate_map
    return mapping


def make_half_adder() -> None:
    gates = [
        GateDetection("xor1", "XOR", BoundingBox(280, 50, 420, 150), 0.99),
        GateDetection("and1", "AND", BoundingBox(280, 230, 420, 330), 0.99),
    ]
    terminals = _terminal_map(gates)
    wires = [
        [(20, terminals["xor1"]["input0"][1]), (130, terminals["xor1"]["input0"][1]), terminals["xor1"]["input0"]],
        [(130, terminals["xor1"]["input0"][1]), (130, terminals["and1"]["input0"][1]), terminals["and1"]["input0"]],
        [(20, terminals["xor1"]["input1"][1]), (200, terminals["xor1"]["input1"][1]), terminals["xor1"]["input1"]],
        [(200, terminals["xor1"]["input1"][1]), (200, terminals["and1"]["input1"][1]), terminals["and1"]["input1"]],
        [terminals["xor1"]["output0"], (540, terminals["xor1"]["output0"][1]), (760, terminals["xor1"]["output0"][1])],
        [terminals["and1"]["output0"], (540, terminals["and1"]["output0"][1]), (760, terminals["and1"]["output0"][1])],
    ]
    _save_fixture("half_adder", (820, 380), gates, wires)


def make_half_adder_dense() -> None:
    gates = [
        GateDetection("xor1", "XOR", BoundingBox(250, 70, 390, 170), 0.99),
        GateDetection("and1", "AND", BoundingBox(250, 200, 390, 300), 0.99),
    ]
    terminals = _terminal_map(gates)
    wires = [
        [(24, terminals["xor1"]["input0"][1]), (118, terminals["xor1"]["input0"][1]), terminals["xor1"]["input0"]],
        [(118, terminals["xor1"]["input0"][1]), (118, terminals["and1"]["input0"][1]), terminals["and1"]["input0"]],
        [(24, terminals["xor1"]["input1"][1]), (176, terminals["xor1"]["input1"][1]), terminals["xor1"]["input1"]],
        [(176, terminals["xor1"]["input1"][1]), (176, terminals["and1"]["input1"][1]), terminals["and1"]["input1"]],
        [terminals["xor1"]["output0"], (500, terminals["xor1"]["output0"][1]), (500, 120), (700, 120)],
        [terminals["and1"]["output0"], (470, terminals["and1"]["output0"][1]), (470, 260), (700, 260)],
    ]
    _save_fixture("half_adder_dense", (760, 360), gates, wires)


def make_half_subtractor() -> None:
    gates = [
        GateDetection("xor1", "XOR", BoundingBox(300, 50, 440, 150), 0.99),
        GateDetection("not1", "NOT", BoundingBox(80, 220, 220, 340), 0.99),
        GateDetection("and1", "AND", BoundingBox(500, 220, 640, 320), 0.99),
    ]
    terminals = _terminal_map(gates)
    b_bus_y = 360
    b_branch_x = 260
    wires = [
        [(20, terminals["xor1"]["input0"][1]), (60, terminals["xor1"]["input0"][1]), terminals["xor1"]["input0"]],
        [(60, terminals["xor1"]["input0"][1]), (60, terminals["not1"]["input0"][1]), terminals["not1"]["input0"]],
        [terminals["not1"]["output0"], (360, terminals["not1"]["output0"][1]), (360, terminals["and1"]["input0"][1]), terminals["and1"]["input0"]],
        [(20, b_bus_y), (b_branch_x, b_bus_y), (b_branch_x, terminals["xor1"]["input1"][1]), terminals["xor1"]["input1"]],
        [(b_branch_x, b_bus_y), (b_branch_x, terminals["and1"]["input1"][1]), terminals["and1"]["input1"]],
        [terminals["xor1"]["output0"], (560, terminals["xor1"]["output0"][1]), (800, terminals["xor1"]["output0"][1])],
        [terminals["and1"]["output0"], (720, terminals["and1"]["output0"][1]), (800, terminals["and1"]["output0"][1])],
    ]
    _save_fixture("half_subtractor", (840, 380), gates, wires)


def make_full_adder() -> None:
    gates = [
        GateDetection("xor1", "XOR", BoundingBox(220, 40, 360, 140), 0.99),
        GateDetection("and1", "AND", BoundingBox(220, 240, 360, 340), 0.99),
        GateDetection("xor2", "XOR", BoundingBox(620, 40, 760, 140), 0.99),
        GateDetection("and2", "AND", BoundingBox(620, 240, 760, 340), 0.99),
        GateDetection("or1", "OR", BoundingBox(900, 140, 1040, 240), 0.99),
    ]
    terminals = _terminal_map(gates)
    or1_input0_elbow_x = 860
    or1_input1_elbow_x = 880
    wires = [
        [(20, terminals["xor1"]["input0"][1]), (120, terminals["xor1"]["input0"][1]), terminals["xor1"]["input0"]],
        [(120, terminals["xor1"]["input0"][1]), (120, terminals["and1"]["input0"][1]), terminals["and1"]["input0"]],
        [(20, terminals["xor1"]["input1"][1]), (190, terminals["xor1"]["input1"][1]), terminals["xor1"]["input1"]],
        [(190, terminals["xor1"]["input1"][1]), (190, terminals["and1"]["input1"][1]), terminals["and1"]["input1"]],
        [terminals["xor1"]["output0"], (500, terminals["xor1"]["output0"][1]), (500, terminals["xor2"]["input0"][1]), terminals["xor2"]["input0"]],
        [(500, terminals["xor1"]["output0"][1]), (500, terminals["and2"]["input0"][1]), terminals["and2"]["input0"]],
        [(20, 390), (580, 390), (580, terminals["xor2"]["input1"][1]), terminals["xor2"]["input1"]],
        [(580, terminals["xor2"]["input1"][1]), (580, terminals["and2"]["input1"][1]), terminals["and2"]["input1"]],
        [
            terminals["and1"]["output0"],
            (360, 380),
            (or1_input0_elbow_x, 380),
            (or1_input0_elbow_x, terminals["or1"]["input0"][1]),
            terminals["or1"]["input0"],
        ],
        [
            terminals["and2"]["output0"],
            (or1_input1_elbow_x, terminals["and2"]["output0"][1]),
            (or1_input1_elbow_x, terminals["or1"]["input1"][1]),
            terminals["or1"]["input1"],
        ],
        [terminals["xor2"]["output0"], (1120, terminals["xor2"]["output0"][1])],
        [terminals["or1"]["output0"], (1140, terminals["or1"]["output0"][1])],
    ]
    _save_fixture("full_adder", (1160, 430), gates, wires)


def make_full_adder_crossed() -> None:
    gates = [
        GateDetection("xor1", "XOR", BoundingBox(180, 60, 320, 160), 0.99),
        GateDetection("and1", "AND", BoundingBox(180, 220, 320, 320), 0.99),
        GateDetection("xor2", "XOR", BoundingBox(520, 60, 660, 160), 0.99),
        GateDetection("and2", "AND", BoundingBox(520, 220, 660, 320), 0.99),
        GateDetection("or1", "OR", BoundingBox(860, 140, 1000, 240), 0.99),
    ]
    terminals = _terminal_map(gates)
    cin_y = 360
    cin_branch_x = 480
    carry_a_elbow_x = 720
    carry_b_elbow_x = 800
    wires = [
        [(18, terminals["xor1"]["input0"][1]), (96, terminals["xor1"]["input0"][1]), terminals["xor1"]["input0"]],
        [(96, terminals["xor1"]["input0"][1]), (96, terminals["and1"]["input0"][1]), terminals["and1"]["input0"]],
        [(18, terminals["xor1"]["input1"][1]), (144, terminals["xor1"]["input1"][1]), terminals["xor1"]["input1"]],
        [(144, terminals["xor1"]["input1"][1]), (144, terminals["and1"]["input1"][1]), terminals["and1"]["input1"]],
        [terminals["xor1"]["output0"], (420, terminals["xor1"]["output0"][1]), (420, terminals["xor2"]["input0"][1]), terminals["xor2"]["input0"]],
        [(420, terminals["xor1"]["output0"][1]), (420, terminals["and2"]["input0"][1]), terminals["and2"]["input0"]],
        [(18, cin_y), (cin_branch_x, cin_y), (cin_branch_x, terminals["xor2"]["input1"][1]), terminals["xor2"]["input1"]],
        [(cin_branch_x, cin_y), (cin_branch_x, terminals["and2"]["input1"][1]), terminals["and2"]["input1"]],
        [
            terminals["and1"]["output0"],
            (340, terminals["and1"]["output0"][1]),
            (340, 340),
            (carry_a_elbow_x, 340),
            (carry_a_elbow_x, terminals["or1"]["input0"][1]),
            terminals["or1"]["input0"],
        ],
        [
            terminals["and2"]["output0"],
            (carry_b_elbow_x, terminals["and2"]["output0"][1]),
            (carry_b_elbow_x, terminals["or1"]["input1"][1]),
            terminals["or1"]["input1"],
        ],
        [terminals["xor2"]["output0"], (760, terminals["xor2"]["output0"][1]), (1080, terminals["xor2"]["output0"][1])],
        [terminals["or1"]["output0"], (1080, terminals["or1"]["output0"][1])],
    ]
    _save_fixture("full_adder_crossed", (1100, 400), gates, wires)


def main() -> None:
    make_half_adder()
    make_half_adder_dense()
    make_half_subtractor()
    make_full_adder()
    make_full_adder_crossed()
    print(f"Wrote fixtures to {SCHEMATICS_DIR} and {DETECTIONS_DIR}")


if __name__ == "__main__":
    main()
