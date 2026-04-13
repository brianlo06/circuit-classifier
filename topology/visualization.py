"""
Visualization helpers for topology analysis.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .types import PipelineResult


def render_analysis(result: PipelineResult, output_path: Optional[Path] = None) -> Image.Image:
    """Render gates and wires onto a copy of the source image."""
    image = Image.open(result.image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for gate in result.gates:
        draw.rectangle(gate.bbox.to_int_tuple(), outline="red", width=2)
        draw.text((gate.bbox.x1, max(0, gate.bbox.y1 - 12)), f"{gate.gate_type} {gate.confidence:.2f}", fill="red")

    for segment in result.wires:
        draw.line([segment.start, segment.end], fill="blue", width=2)

    footer_y = max(0, image.height - 40)
    draw.text((10, footer_y), f"{result.classification.label} ({result.classification.confidence:.2f})", fill="green")

    if output_path:
        image.save(output_path)

    return image


def render_debug_analysis(result: PipelineResult, output_path: Optional[Path] = None) -> Image.Image:
    """Render a richer debugging view with wire components and terminal matches."""
    base = Image.open(result.image_path).convert("RGB")
    overlay = base.copy()
    draw = ImageDraw.Draw(overlay)
    font = _load_font()

    palette = [
        "#d73027", "#4575b4", "#1a9850", "#984ea3", "#ff7f00",
        "#a65628", "#f781bf", "#4daf4a", "#377eb8", "#e41a1c",
    ]

    if result.wire_mask is not None:
        mask_rgb = _mask_to_rgb(result.wire_mask)
        mask_panel = Image.fromarray(mask_rgb)
    else:
        mask_panel = Image.new("RGB", base.size, "white")

    for gate in result.gates:
        draw.rectangle(gate.bbox.to_int_tuple(), outline="red", width=2)
        draw.text((gate.bbox.x1, max(0, gate.bbox.y1 - 14)), f"{gate.gate_id}:{gate.gate_type}", fill="red", font=font)

    for index, component in enumerate(result.wire_components):
        color = palette[index % len(palette)]
        for segment in component.segments:
            draw.line([segment.start, segment.end], fill=color, width=3)
        for point in component.points:
            draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=color, outline=color)
        if component.points:
            anchor = min(component.points, key=lambda item: (item[0], item[1]))
            draw.text((anchor[0] + 4, anchor[1] + 2), component.component_id, fill=color, font=font)

    for terminal in result.terminals:
        color = "#111111" if terminal.kind == "input" else "#006d2c"
        x, y = terminal.point
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, fill="white", width=2)
        draw.text((x + 6, y - 8), f"{terminal.gate_id}.{terminal.kind[0]}{terminal.index}", fill=color, font=font)

    for component_id, terminals in result.component_matches.items():
        if not terminals:
            continue
        component = next((item for item in result.wire_components if item.component_id == component_id), None)
        if component is None or not component.points:
            continue
        anchor = min(component.points, key=lambda item: (item[0], item[1]))
        lines = [component_id] + [f"{terminal.gate_id}.{terminal.kind[0]}{terminal.index}" for terminal in terminals]
        _draw_match_box(draw, (anchor[0] + 10, anchor[1] + 10), lines, font)

    footer = f"{result.classification.label} ({result.classification.confidence:.2f})"
    draw.text((10, max(0, overlay.height - 24)), footer, fill="green", font=font)

    canvas = Image.new("RGB", (base.width * 2, base.height), "white")
    canvas.paste(overlay, (0, 0))
    canvas.paste(mask_panel, (base.width, 0))

    panel_draw = ImageDraw.Draw(canvas)
    panel_draw.text((10, 10), "Overlay", fill="black", font=font)
    panel_draw.text((base.width + 10, 10), "Wire Mask", fill="black", font=font)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)

    return canvas


def _load_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        return ImageFont.load_default()


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        return np.stack([mask, mask, mask], axis=-1).astype(np.uint8)
    return mask.astype(np.uint8)


def _draw_match_box(
    draw: ImageDraw.ImageDraw,
    origin: Tuple[float, float],
    lines: List[str],
    font: ImageFont.ImageFont,
) -> None:
    x, y = origin
    widths = []
    heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    width = max(widths) + 8
    height = sum(heights) + 6 + max(0, len(lines) - 1) * 2
    draw.rectangle((x, y, x + width, y + height), fill="white", outline="black")
    cursor_y = y + 3
    for line, text_height in zip(lines, heights):
        draw.text((x + 4, cursor_y), line, fill="black", font=font)
        cursor_y += text_height + 2
