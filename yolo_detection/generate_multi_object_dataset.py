"""
Generate a synthetic multi-object schematic dataset for YOLO training.

This creates larger schematic-like canvases containing multiple gate symbols,
wire routing, and one YOLO box per placed gate. It is intended to bridge the
gap between the existing single-gate detection dataset and real multi-gate
schematic detection.
"""

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from convert_to_yolo import find_bounding_box


CLASS_NAMES = ["AND", "NAND", "NOR", "NOT", "OR", "XNOR", "XOR"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".gif"}
INPUT_COUNTS = {
    "NOT": 1,
    "AND": 2,
    "NAND": 2,
    "NOR": 2,
    "OR": 2,
    "XNOR": 2,
    "XOR": 2,
}


@dataclass(frozen=True)
class GateAsset:
    gate_type: str
    image_path: Path


@dataclass
class PlacedGate:
    gate_id: str
    gate_type: str
    image: Image.Image
    bbox: Tuple[int, int, int, int]
    input_points: List[Tuple[int, int]]
    output_point: Tuple[int, int]


def _font(size: int = 18) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


def _fit_fixture_font(draw: ImageDraw.ImageDraw, label: str, max_width: int, max_height: int) -> ImageFont.ImageFont:
    for size in range(28, 13, -1):
        font = _font(size)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        if (right - left) <= max_width and (bottom - top) <= max_height:
            return font
    return _font(14)


def load_gate_assets(source_dir: Path) -> Dict[str, List[GateAsset]]:
    assets: Dict[str, List[GateAsset]] = {name: [] for name in CLASS_NAMES}
    for class_name in CLASS_NAMES:
        class_dir = source_dir / class_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                assets[class_name].append(GateAsset(gate_type=class_name, image_path=image_path))
    return assets


def prepare_gate_symbol(image_path: Path, target_height: int, rng: random.Random) -> Image.Image:
    bbox = find_bounding_box(image_path, padding=0.02)
    source = Image.open(image_path).convert("RGBA")
    arr = np.array(source)

    if bbox is None:
        cropped = source
    else:
        center_x, center_y, width, height = bbox
        x1 = int(max(0, round((center_x - width / 2.0) * source.width)))
        y1 = int(max(0, round((center_y - height / 2.0) * source.height)))
        x2 = int(min(source.width, round((center_x + width / 2.0) * source.width)))
        y2 = int(min(source.height, round((center_y + height / 2.0) * source.height)))
        cropped = source.crop((x1, y1, x2, y2))

    rgb = np.array(cropped.convert("RGB"))
    mask = np.any(rgb < 240, axis=2).astype(np.uint8) * 255
    alpha = Image.fromarray(mask)
    symbol = Image.new("RGBA", cropped.size, (255, 255, 255, 0))
    symbol.paste(cropped.convert("RGBA"), (0, 0), alpha)

    scale = target_height / max(symbol.height, 1)
    target_width = max(24, int(round(symbol.width * scale)))
    symbol = symbol.resize((target_width, target_height), Image.Resampling.LANCZOS)

    if rng.random() < 0.35:
        symbol = symbol.filter(ImageFilter.GaussianBlur(radius=0.25))
    return symbol


def gate_terminal_points(bbox: Tuple[int, int, int, int], gate_type: str) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
    x1, y1, x2, y2 = bbox
    height = y2 - y1
    if INPUT_COUNTS[gate_type] == 1:
        input_points = [(x1 - 2, y1 + height // 2)]
    else:
        input_points = [
            (x1 - 2, y1 + int(round(height * 0.25))),
            (x1 - 2, y1 + int(round(height * 0.75))),
        ]
    output_point = (x2 + 2, y1 + height // 2)
    return input_points, output_point


def prepare_fixture_gate(gate_type: str, rng: random.Random) -> Image.Image:
    width = rng.randint(132, 148)
    height = rng.randint(96, 108)
    image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    fill_shade = rng.randint(244, 248)
    outline_width = rng.choice((2, 2, 3))
    draw.rectangle(
        (1, 1, width - 2, height - 2),
        outline="black",
        fill=(fill_shade, fill_shade, fill_shade, 255),
        width=outline_width,
    )

    label = gate_type
    font = _fit_fixture_font(draw, label, max_width=width - 18, max_height=height - 14)
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = 8
    text_y = max(6, (height - text_height) // 2 - 2)
    # Slightly embolden the label so gate classes survive JPEG artifacts and downsampling better.
    for dx, dy in ((0, 0), (1, 0)):
        draw.text((text_x + dx, text_y + dy), label, font=font, fill="black")

    if rng.random() < 0.2:
        stub_len = rng.randint(4, 8)
        mid_y = height // 2
        draw.line([(0, mid_y), (stub_len, mid_y)], fill="black", width=2)
        draw.line([(width - 1 - stub_len, mid_y), (width - 1, mid_y)], fill="black", width=2)
    return image


def place_gates(
    assets: Dict[str, List[GateAsset]],
    image_size: Tuple[int, int],
    gate_count: int,
    rng: random.Random,
    gate_style: str,
) -> List[PlacedGate]:
    width, height = image_size
    columns = min(3, max(2, gate_count // 2))
    left_margin = 140
    right_margin = 140
    top_margin = 60
    bottom_margin = 60
    col_step = max(170, (width - left_margin - right_margin) // max(columns - 1, 1))

    gates_per_column: List[int] = []
    remaining = gate_count
    for column_index in range(columns):
        columns_left = columns - column_index
        min_here = 1
        max_here = remaining - (columns_left - 1)
        count_here = rng.randint(min_here, max_here)
        gates_per_column.append(count_here)
        remaining -= count_here

    placed: List[PlacedGate] = []
    gate_index = 0

    for column_index, count_here in enumerate(gates_per_column):
        x_center = left_margin + column_index * col_step + rng.randint(-18, 18)
        usable_height = height - top_margin - bottom_margin
        row_step = usable_height / (count_here + 1)
        for row_index in range(count_here):
            gate_type = rng.choice(CLASS_NAMES)
            if gate_style == "symbol":
                asset = rng.choice(assets[gate_type])
                symbol_height = rng.randint(70, 110)
                symbol = prepare_gate_symbol(asset.image_path, symbol_height, rng)
            else:
                symbol = prepare_fixture_gate(gate_type, rng)
            x1 = int(x_center - symbol.width // 2)
            y1 = int(top_margin + row_step * (row_index + 1) - symbol.height // 2 + rng.randint(-12, 12))
            x1 = max(40, min(width - symbol.width - 40, x1))
            y1 = max(20, min(height - symbol.height - 20, y1))
            bbox = (x1, y1, x1 + symbol.width, y1 + symbol.height)
            input_points, output_point = gate_terminal_points(bbox, gate_type)
            placed.append(
                PlacedGate(
                    gate_id=f"gate_{gate_index}",
                    gate_type=gate_type,
                    image=symbol,
                    bbox=bbox,
                    input_points=input_points,
                    output_point=output_point,
                )
            )
            gate_index += 1

    placed.sort(key=lambda gate: (gate.bbox[0], gate.bbox[1]))
    return placed


def assign_connections(gates: Sequence[PlacedGate], rng: random.Random) -> Tuple[List[Tuple[Tuple[int, int], Tuple[int, int]]], List[dict]]:
    columns: Dict[int, List[PlacedGate]] = {}
    ordered_x = sorted({gate.bbox[0] for gate in gates})
    x_to_column = {x: index for index, x in enumerate(ordered_x)}
    for gate in gates:
        columns.setdefault(x_to_column[gate.bbox[0]], []).append(gate)

    segments: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    metadata: List[dict] = []
    junction_dots: List[Tuple[int, int]] = []

    for column_index, column_gates in columns.items():
        previous_gates = [gate for index, items in columns.items() if index < column_index for gate in items]
        for gate in column_gates:
            used_inputs = 0
            for input_index, point in enumerate(gate.input_points):
                if previous_gates and rng.random() < 0.65:
                    source = rng.choice(previous_gates)
                    routed = route_wire(source.output_point, point, rng, jog_x=None)
                    segments.extend(pairwise_segments(routed))
                    metadata.append(
                        {
                            "source_gate": source.gate_id,
                            "target_gate": gate.gate_id,
                            "target_input": input_index,
                        }
                    )
                else:
                    external_start = (rng.randint(16, 44), point[1] + rng.randint(-8, 8))
                    routed = route_wire(external_start, point, rng, jog_x=max(70, point[0] - rng.randint(45, 85)))
                    segments.extend(pairwise_segments(routed))
                    if len(routed) >= 3:
                        junction_dots.append(routed[1])
                    metadata.append(
                        {
                            "source_gate": None,
                            "target_gate": gate.gate_id,
                            "target_input": input_index,
                        }
                    )
                used_inputs += 1

            if used_inputs == 0:
                continue

            output_end = (min(gate.output_point[0] + rng.randint(50, 140), max(g.output_point[0] for g in gates) + 150), gate.output_point[1] + rng.randint(-8, 8))
            routed_out = route_wire(gate.output_point, output_end, rng, jog_x=gate.output_point[0] + rng.randint(30, 90))
            segments.extend(pairwise_segments(routed_out))
            if len(routed_out) >= 3 and rng.random() < 0.35:
                junction_dots.append(routed_out[1])

    return segments, metadata, junction_dots


def route_wire(
    start: Tuple[int, int],
    end: Tuple[int, int],
    rng: random.Random,
    jog_x: int = None,
) -> List[Tuple[int, int]]:
    sx, sy = start
    ex, ey = end
    if jog_x is None:
        low = min(sx, ex) + 24
        high = max(sx, ex) - 24
        if low < high:
            jog_x = rng.randint(low, high)
        else:
            jog_x = (sx + ex) // 2
    return [(sx, sy), (jog_x, sy), (jog_x, ey), (ex, ey)]


def pairwise_segments(points: Sequence[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    return list(zip(points, points[1:]))


def draw_schematic(
    image_size: Tuple[int, int],
    gates: Sequence[PlacedGate],
    wire_segments: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]],
    rng: random.Random,
    junction_dots: Sequence[Tuple[int, int]],
    gate_style: str,
) -> Image.Image:
    canvas = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(canvas)

    if gate_style == "symbol" and rng.random() < 0.5:
        for _ in range(rng.randint(4, 10)):
            y = rng.randint(0, image_size[1] - 1)
            shade = rng.randint(246, 252)
            draw.line([(0, y), (image_size[0], y)], fill=(shade, shade, shade), width=1)

    for start, end in wire_segments:
        width = 2 if gate_style == "fixture" else (2 if rng.random() < 0.8 else 3)
        draw.line([start, end], fill="black", width=width)

    for x, y in junction_dots:
        radius = 3 if gate_style == "fixture" else 2
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="black")

    for gate in gates:
        canvas.paste(gate.image, (gate.bbox[0], gate.bbox[1]), gate.image)

    if gate_style == "symbol" and rng.random() < 0.35:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.15))

    return canvas


def yolo_label_from_bbox(bbox: Tuple[int, int, int, int], image_size: Tuple[int, int]) -> Tuple[float, float, float, float]:
    width, height = image_size
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    center_x = x1 + box_width / 2.0
    center_y = y1 + box_height / 2.0
    return (
        center_x / width,
        center_y / height,
        box_width / width,
        box_height / height,
    )


def reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_data_yaml(output_dir: Path) -> None:
    lines = [
        "# Synthetic multi-object schematic dataset for YOLOv8",
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for index, name in enumerate(CLASS_NAMES):
        lines.append(f"  {index}: {name}")
    lines.append(f"nc: {len(CLASS_NAMES)}")
    (output_dir / "data.yaml").write_text("\n".join(lines) + "\n")


def generate_dataset(
    source_dir: Path,
    output_dir: Path,
    train_count: int,
    val_count: int,
    seed: int,
    gate_style: str,
) -> dict:
    rng = random.Random(seed)
    if gate_style == "symbol":
        assets = load_gate_assets(source_dir)
        missing = [name for name, items in assets.items() if not items]
        if missing:
            raise FileNotFoundError(f"Missing source images for classes: {', '.join(missing)}")
    else:
        assets = {name: [] for name in CLASS_NAMES}

    reset_output_dir(output_dir)
    write_data_yaml(output_dir)

    manifest = []
    image_index = 0
    split_counts = {"train": train_count, "val": val_count}

    for split_name, count in split_counts.items():
        for split_index in range(count):
            image_size = (rng.randint(900, 1400), rng.randint(520, 800))
            gate_count = rng.randint(3, 7)
            gates = place_gates(assets, image_size, gate_count, rng, gate_style=gate_style)
            wire_segments, graph_metadata, junction_dots = assign_connections(gates, rng)
            image = draw_schematic(image_size, gates, wire_segments, rng, junction_dots, gate_style)

            image_name = f"schematic_{image_index:05d}.jpg"
            label_name = f"schematic_{image_index:05d}.txt"
            image.save(output_dir / "images" / split_name / image_name, "JPEG", quality=95)

            label_lines = []
            gate_records = []
            for gate in gates:
                cx, cy, w, h = yolo_label_from_bbox(gate.bbox, image_size)
                label_lines.append(f"{CLASS_TO_ID[gate.gate_type]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                gate_records.append(
                    {
                        "gate_id": gate.gate_id,
                        "gate_type": gate.gate_type,
                        "bbox": list(gate.bbox),
                    }
                )
            (output_dir / "labels" / split_name / label_name).write_text("\n".join(label_lines) + "\n")

            manifest.append(
                {
                    "split": split_name,
                    "image": f"images/{split_name}/{image_name}",
                    "label": f"labels/{split_name}/{label_name}",
                    "image_size": list(image_size),
                    "gate_style": gate_style,
                    "gates": gate_records,
                    "connections": graph_metadata,
                }
            )
            image_index += 1

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {
        "train": train_count,
        "val": val_count,
        "gate_style": gate_style,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic multi-object schematic dataset")
    parser.add_argument("--source", type=str, default="../data", help="Source single-gate dataset")
    parser.add_argument("--output", type=str, default="datasets/complex_circuits", help="Output YOLO dataset dir")
    parser.add_argument("--train-count", type=int, default=240, help="Number of train images to generate")
    parser.add_argument("--val-count", type=int, default=60, help="Number of val images to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--gate-style",
        choices=("fixture", "symbol"),
        default="fixture",
        help="Render fixture-style labeled gate boxes or cropped symbol images",
    )
    args = parser.parse_args()

    summary = generate_dataset(
        source_dir=Path(args.source),
        output_dir=Path(args.output),
        train_count=args.train_count,
        val_count=args.val_count,
        seed=args.seed,
        gate_style=args.gate_style,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
