"""
Synthetic logic gate image generator.
Generates unique gate symbols with variations for training data.
"""

import math
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, List, Optional


class GateGenerator:
    """Generates synthetic logic gate images with variations."""

    def __init__(self, size: Tuple[int, int] = (300, 200)):
        self.size = size
        self.variations = {
            'line_width': [2, 3, 4, 5],
            'scale': [0.7, 0.8, 0.9, 1.0, 1.1],
            'rotation': [-5, -3, 0, 3, 5],
            'bg_color': ['white', '#f5f5f5', '#fafafa', '#f0f0f0', '#fff8f0'],
            'line_color': ['black', '#1a1a1a', '#333333', '#0a0a0a', '#2c2c2c'],
            'show_labels': [True, False],
            'label_style': ['AB', 'XY', '12', 'ab', 'none'],
            'input_style': ['lines', 'short', 'none'],
        }

    def _draw_and_shape(self, draw: ImageDraw, cx: float, cy: float,
                        scale: float, line_width: int, color: str):
        """Draw AND gate shape (flat left, curved right)."""
        w, h = 80 * scale, 60 * scale

        # Left flat edge
        left = cx - w/2
        right = cx + w/2
        top = cy - h/2
        bottom = cy + h/2

        # Draw the flat left side and top/bottom
        draw.line([(left, top), (left, bottom)], fill=color, width=line_width)
        draw.line([(left, top), (cx, top)], fill=color, width=line_width)
        draw.line([(left, bottom), (cx, bottom)], fill=color, width=line_width)

        # Draw curved right side (arc)
        draw.arc([(cx - h/2, top), (cx + h/2, bottom)], -90, 90, fill=color, width=line_width)

        return left, right + h/2 - w/2, top, bottom

    def _draw_or_shape(self, draw: ImageDraw, cx: float, cy: float,
                       scale: float, line_width: int, color: str):
        """Draw OR gate shape (curved input, pointed output)."""
        w, h = 80 * scale, 60 * scale

        left = cx - w/2
        top = cy - h/2
        bottom = cy + h/2

        # Curved back
        draw.arc([(left - w/3, top), (left + w/3, bottom)], -90, 90, fill=color, width=line_width)

        # Top curve to point
        for i in range(20):
            t = i / 19
            x1 = left + w/6 + t * w * 0.7
            y1 = top + (1 - t) * h * 0.1 + t * h/2 - math.sin(t * math.pi) * h * 0.15
            x2 = left + w/6 + (t + 0.05) * w * 0.7
            y2 = top + (1 - (t + 0.05)) * h * 0.1 + (t + 0.05) * h/2 - math.sin((t + 0.05) * math.pi) * h * 0.15
            if i < 19:
                draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)

        # Bottom curve to point
        for i in range(20):
            t = i / 19
            x1 = left + w/6 + t * w * 0.7
            y1 = bottom - (1 - t) * h * 0.1 - t * h/2 + math.sin(t * math.pi) * h * 0.15
            x2 = left + w/6 + (t + 0.05) * w * 0.7
            y2 = bottom - (1 - (t + 0.05)) * h * 0.1 - (t + 0.05) * h/2 + math.sin((t + 0.05) * math.pi) * h * 0.15
            if i < 19:
                draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)

        return left, left + w * 0.85, top, bottom

    def _draw_not_shape(self, draw: ImageDraw, cx: float, cy: float,
                        scale: float, line_width: int, color: str):
        """Draw NOT gate (triangle with bubble)."""
        w, h = 60 * scale, 50 * scale
        bubble_r = 6 * scale

        left = cx - w/2
        right = cx + w/2
        top = cy - h/2
        bottom = cy + h/2

        # Triangle
        draw.polygon([(left, top), (left, bottom), (right - bubble_r * 2, cy)],
                     outline=color, width=line_width)

        # Bubble
        draw.ellipse([(right - bubble_r * 2, cy - bubble_r),
                      (right, cy + bubble_r)], outline=color, width=line_width)

        return left, right, top, bottom

    def _draw_xor_shape(self, draw: ImageDraw, cx: float, cy: float,
                        scale: float, line_width: int, color: str):
        """Draw XOR gate (OR with extra curved line)."""
        w, h = 80 * scale, 60 * scale
        left = cx - w/2
        top = cy - h/2
        bottom = cy + h/2

        # Extra curved line at input
        draw.arc([(left - w/3 - w/8, top), (left + w/3 - w/8, bottom)], -90, 90, fill=color, width=line_width)

        # Regular OR shape
        self._draw_or_shape(draw, cx, cy, scale, line_width, color)

        return left - w/8, left + w * 0.85, top, bottom

    def _draw_bubble(self, draw: ImageDraw, x: float, y: float,
                     radius: float, line_width: int, color: str):
        """Draw inversion bubble."""
        draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)],
                     outline=color, width=line_width)

    def _draw_inputs(self, draw: ImageDraw, left: float, cy: float, h: float,
                     line_width: int, color: str, style: str, num_inputs: int = 2):
        """Draw input lines."""
        if style == 'none':
            return

        length = 30 if style == 'lines' else 15

        if num_inputs == 1:
            positions = [cy]
        else:
            spacing = h * 0.5
            positions = [cy - spacing/2, cy + spacing/2]

        for y in positions:
            draw.line([(left - length, y), (left, y)], fill=color, width=line_width)

    def _draw_output(self, draw: ImageDraw, right: float, cy: float,
                     line_width: int, color: str, length: int = 30):
        """Draw output line."""
        draw.line([(right, cy), (right + length, cy)], fill=color, width=line_width)

    def _draw_labels(self, draw: ImageDraw, left: float, right: float, cy: float, h: float,
                     style: str, color: str, num_inputs: int = 2):
        """Draw input/output labels."""
        if style == 'none':
            return

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except:
            font = ImageFont.load_default()

        labels = {
            'AB': (['A', 'B'], 'Y'),
            'XY': (['X', 'Y'], 'Z'),
            '12': (['1', '2'], 'Q'),
            'ab': (['a', 'b'], 'q'),
        }

        if style in labels:
            inputs, output = labels[style]
            spacing = h * 0.5

            if num_inputs == 1:
                draw.text((left - 45, cy - 8), inputs[0], fill=color, font=font)
            else:
                draw.text((left - 45, cy - spacing/2 - 8), inputs[0], fill=color, font=font)
                draw.text((left - 45, cy + spacing/2 - 8), inputs[1], fill=color, font=font)

            draw.text((right + 35, cy - 8), output, fill=color, font=font)

    def generate(self, gate_type: str, variation_seed: Optional[int] = None) -> Image.Image:
        """
        Generate a gate image with random variations.

        Args:
            gate_type: One of 'AND', 'NAND', 'OR', 'NOR', 'XOR', 'XNOR', 'NOT'
            variation_seed: Optional seed for reproducible variations
        """
        if variation_seed is not None:
            random.seed(variation_seed)

        # Pick random variations
        line_width = random.choice(self.variations['line_width'])
        scale = random.choice(self.variations['scale'])
        rotation = random.choice(self.variations['rotation'])
        bg_color = random.choice(self.variations['bg_color'])
        line_color = random.choice(self.variations['line_color'])
        show_labels = random.choice(self.variations['show_labels'])
        label_style = random.choice(self.variations['label_style']) if show_labels else 'none'
        input_style = random.choice(self.variations['input_style'])

        # Create image
        img = Image.new('RGB', self.size, bg_color)
        draw = ImageDraw.Draw(img)

        cx, cy = self.size[0] / 2, self.size[1] / 2

        # Determine number of inputs
        num_inputs = 1 if gate_type == 'NOT' else 2

        # Draw the gate shape
        if gate_type == 'AND':
            left, right, top, bottom = self._draw_and_shape(draw, cx, cy, scale, line_width, line_color)
            self._draw_inputs(draw, left, cy, bottom - top, line_width, line_color, input_style, num_inputs)
            self._draw_output(draw, right, cy, line_width, line_color)

        elif gate_type == 'NAND':
            left, right, top, bottom = self._draw_and_shape(draw, cx - 5, cy, scale, line_width, line_color)
            bubble_r = 6 * scale
            self._draw_bubble(draw, right + bubble_r, cy, bubble_r, line_width, line_color)
            self._draw_inputs(draw, left, cy, bottom - top, line_width, line_color, input_style, num_inputs)
            self._draw_output(draw, right + bubble_r * 2, cy, line_width, line_color)
            right += bubble_r * 2

        elif gate_type == 'OR':
            left, right, top, bottom = self._draw_or_shape(draw, cx, cy, scale, line_width, line_color)
            self._draw_inputs(draw, left + 15 * scale, cy, bottom - top, line_width, line_color, input_style, num_inputs)
            self._draw_output(draw, right, cy, line_width, line_color)

        elif gate_type == 'NOR':
            left, right, top, bottom = self._draw_or_shape(draw, cx - 5, cy, scale, line_width, line_color)
            bubble_r = 6 * scale
            self._draw_bubble(draw, right + bubble_r, cy, bubble_r, line_width, line_color)
            self._draw_inputs(draw, left + 15 * scale, cy, bottom - top, line_width, line_color, input_style, num_inputs)
            self._draw_output(draw, right + bubble_r * 2, cy, line_width, line_color)
            right += bubble_r * 2

        elif gate_type == 'XOR':
            left, right, top, bottom = self._draw_xor_shape(draw, cx, cy, scale, line_width, line_color)
            self._draw_inputs(draw, left + 15 * scale, cy, bottom - top, line_width, line_color, input_style, num_inputs)
            self._draw_output(draw, right, cy, line_width, line_color)

        elif gate_type == 'XNOR':
            left, right, top, bottom = self._draw_xor_shape(draw, cx - 5, cy, scale, line_width, line_color)
            bubble_r = 6 * scale
            self._draw_bubble(draw, right + bubble_r, cy, bubble_r, line_width, line_color)
            self._draw_inputs(draw, left + 15 * scale, cy, bottom - top, line_width, line_color, input_style, num_inputs)
            self._draw_output(draw, right + bubble_r * 2, cy, line_width, line_color)
            right += bubble_r * 2

        elif gate_type == 'NOT':
            left, right, top, bottom = self._draw_not_shape(draw, cx, cy, scale, line_width, line_color)
            self._draw_inputs(draw, left, cy, bottom - top, line_width, line_color, input_style, num_inputs=1)
            self._draw_output(draw, right, cy, line_width, line_color)

        # Draw labels
        if show_labels and label_style != 'none':
            self._draw_labels(draw, left, right, cy, bottom - top, label_style, line_color, num_inputs)

        # Apply rotation
        if rotation != 0:
            img = img.rotate(rotation, fillcolor=bg_color, expand=False)

        return img

    def generate_batch(self, gate_type: str, count: int, start_idx: int = 1) -> List[Tuple[Image.Image, str]]:
        """Generate multiple unique gate images."""
        images = []
        for i in range(count):
            img = self.generate(gate_type, variation_seed=start_idx + i + hash(gate_type))
            filename = f"{gate_type}{start_idx + i}.png"
            images.append((img, filename))
        return images


def fill_dataset(data_dir: str, target_per_class: int = 30):
    """
    Fill dataset to target count per class using synthetic images.

    Args:
        data_dir: Path to data directory
        target_per_class: Target number of images per class
    """
    data_path = Path(data_dir)
    generator = GateGenerator()

    gate_types = ['AND', 'NAND', 'OR', 'NOR', 'XOR', 'XNOR', 'NOT']

    for gate_type in gate_types:
        class_dir = data_path / gate_type
        class_dir.mkdir(exist_ok=True)

        # Count existing images
        existing = list(class_dir.glob('*.[pjgwa]*'))  # png, jpg, gif, webp, avif
        existing_count = len(existing)

        needed = target_per_class - existing_count

        if needed <= 0:
            print(f"{gate_type}: Already has {existing_count} images (target: {target_per_class})")
            continue

        print(f"{gate_type}: Generating {needed} synthetic images...")

        # Generate images starting from next index
        images = generator.generate_batch(gate_type, needed, start_idx=existing_count + 1)

        for img, filename in images:
            filepath = class_dir / filename
            img.save(filepath)

        print(f"  Created {needed} images ({gate_type}1.png to {gate_type}{existing_count + needed}.png)")


if __name__ == "__main__":
    import sys

    data_dir = Path(__file__).parent / "data"

    if len(sys.argv) > 1:
        target = int(sys.argv[1])
    else:
        target = 30

    print(f"Filling dataset to {target} images per class...\n")
    fill_dataset(data_dir, target)
    print("\nDone!")
