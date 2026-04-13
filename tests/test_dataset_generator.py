import random
import tempfile
import unittest
from pathlib import Path

from yolo_detection.generate_multi_object_dataset import (
    PlacedGate,
    draw_schematic,
    generate_dataset,
    prepare_fixture_gate,
)


class FixtureDatasetGeneratorTests(unittest.TestCase):
    def test_prepare_fixture_gate_uses_fixture_fill_palette(self) -> None:
        gate = prepare_fixture_gate("AND", random.Random(42))

        self.assertEqual(gate.mode, "RGBA")
        self.assertEqual(gate.getpixel((10, 10)), (246, 246, 246, 255))

    def test_fixture_draw_schematic_keeps_wire_visible_over_gate(self) -> None:
        gate_image = prepare_fixture_gate("XOR", random.Random(7))
        placed_gate = PlacedGate(
            gate_id="gate_0",
            gate_type="XOR",
            image=gate_image,
            bbox=(40, 20, 40 + gate_image.width, 20 + gate_image.height),
            input_points=[],
            output_point=(0, 0),
        )

        image = draw_schematic(
            image_size=(220, 160),
            gates=[placed_gate],
            wire_segments=[((20, 70), (190, 70))],
            rng=random.Random(9),
            junction_dots=[],
            gate_style="fixture",
        )

        self.assertEqual(image.getpixel((80, 70)), (0, 0, 0))

    def test_fixture_dataset_uses_png_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "dataset"
            generate_dataset(
                source_dir=Path("/Users/brianlo/circuit-classifier/data"),
                output_dir=output_dir,
                train_count=1,
                val_count=1,
                seed=42,
                gate_style="fixture",
            )

            train_images = sorted((output_dir / "images" / "train").iterdir())
            val_images = sorted((output_dir / "images" / "val").iterdir())

            self.assertEqual([path.suffix for path in train_images], [".png"])
            self.assertEqual([path.suffix for path in val_images], [".png"])


if __name__ == "__main__":
    unittest.main()
