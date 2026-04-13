import json
import tempfile
import unittest
from pathlib import Path

from topology.run_symbol_crop_benchmark import load_crop_benchmark


class SymbolCropBenchmarkTests(unittest.TestCase):
    def test_load_crop_benchmark_normalizes_label_and_uses_default_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "benchmark.json"
            benchmark_path.write_text(
                json.dumps(
                    [
                        {
                            "image": "examples/symbol.png",
                            "label": "xor",
                            "bbox": [10, 20, 30, 40],
                        }
                    ]
                )
            )

            items = load_crop_benchmark(benchmark_path)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].sample_id, "sample_0")
        self.assertEqual(items[0].label, "XOR")
        self.assertEqual(items[0].bbox.to_int_tuple(), (10, 20, 30, 40))

    def test_load_crop_benchmark_rejects_invalid_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_path = Path(tmpdir) / "benchmark.json"
            benchmark_path.write_text(
                json.dumps(
                    [
                        {
                            "image": "examples/symbol.png",
                            "label": "xor",
                            "bbox": [10, 20, 30],
                        }
                    ]
                )
            )

            with self.assertRaises(ValueError):
                load_crop_benchmark(benchmark_path)


if __name__ == "__main__":
    unittest.main()
