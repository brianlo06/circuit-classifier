import unittest
from pathlib import Path

from PIL import Image

from topology.generate_example_schematics import main as generate_example_schematics
from topology.pipeline import CircuitAnalysisPipeline
from topology.types import BoundingBox


ROOT = Path(__file__).resolve().parent.parent
HALF_ADDER_IMAGE = ROOT / "examples" / "schematics" / "half_adder.png"
HALF_ADDER_DETECTIONS = ROOT / "examples" / "detections" / "half_adder.json"
HALF_ADDER_DENSE_IMAGE = ROOT / "examples" / "schematics" / "half_adder_dense.png"
HALF_ADDER_DENSE_DETECTIONS = ROOT / "examples" / "detections" / "half_adder_dense.json"
HALF_SUBTRACTOR_IMAGE = ROOT / "examples" / "schematics" / "half_subtractor.png"
HALF_SUBTRACTOR_DETECTIONS = ROOT / "examples" / "detections" / "half_subtractor.json"
FULL_ADDER_IMAGE = ROOT / "examples" / "schematics" / "full_adder.png"
FULL_ADDER_DETECTIONS = ROOT / "examples" / "detections" / "full_adder.json"
FULL_ADDER_CROSSED_IMAGE = ROOT / "examples" / "schematics" / "full_adder_crossed.png"
FULL_ADDER_CROSSED_DETECTIONS = ROOT / "examples" / "detections" / "full_adder_crossed.json"


class TopologyPipelineFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        generate_example_schematics()
        cls.pipeline = CircuitAnalysisPipeline()

    def _analyze(self, image_path: Path, detections_path: Path):
        detections = self.pipeline.load_detections_json(detections_path)
        return self.pipeline.analyze(image_path, detections=detections)

    def test_half_adder_fixture_classifies_and_connects_correctly(self) -> None:
        result = self._analyze(HALF_ADDER_IMAGE, HALF_ADDER_DETECTIONS)

        self.assertEqual(result.classification.label, "half_adder")
        self.assertEqual(len(result.graph.primary_inputs), 2)
        self.assertEqual(len(result.graph.primary_outputs), 2)
        self.assertEqual(len(result.graph.connections), 0)
        self.assertEqual(result.classification.expressions["OUT0"], "(IN0 AND IN1)")
        self.assertEqual(result.classification.expressions["OUT1"], "(IN0 XOR IN1)")

    def test_full_adder_fixture_classifies_and_connects_correctly(self) -> None:
        result = self._analyze(FULL_ADDER_IMAGE, FULL_ADDER_DETECTIONS)

        self._assert_full_adder_result(result)

    def test_half_adder_dense_fixture_classifies_and_connects_correctly(self) -> None:
        result = self._analyze(HALF_ADDER_DENSE_IMAGE, HALF_ADDER_DENSE_DETECTIONS)

        self.assertEqual(result.classification.label, "half_adder")
        self.assertEqual(len(result.graph.primary_inputs), 2)
        self.assertEqual(len(result.graph.primary_outputs), 2)
        self.assertEqual(len(result.graph.connections), 0)

    def test_half_subtractor_fixture_classifies_and_connects_correctly(self) -> None:
        result = self._analyze(HALF_SUBTRACTOR_IMAGE, HALF_SUBTRACTOR_DETECTIONS)

        self.assertEqual(result.classification.label, "half_subtractor")
        self.assertEqual(len(result.graph.primary_inputs), 2)
        self.assertEqual(len(result.graph.primary_outputs), 2)
        expressions = set(result.classification.expressions.values())
        self.assertTrue(any("XOR" in expression for expression in expressions))
        self.assertTrue(any("NOT(" in expression and "AND" in expression for expression in expressions))

        actual_connections = {
            (connection.source_gate, connection.target_gate, connection.target_input_index)
            for connection in result.graph.connections
        }
        expected_connections = {
            ("not1", "and1", 0),
        }
        self.assertEqual(actual_connections, expected_connections)

    def test_full_adder_crossed_fixture_classifies_and_connects_correctly(self) -> None:
        result = self._analyze(FULL_ADDER_CROSSED_IMAGE, FULL_ADDER_CROSSED_DETECTIONS)

        self._assert_full_adder_result(result)

    def _assert_full_adder_result(self, result) -> None:
        self.assertEqual(result.classification.label, "full_adder")
        self.assertEqual(len(result.graph.primary_inputs), 3)
        self.assertEqual(len(result.graph.primary_outputs), 2)
        self.assertEqual(result.classification.expressions["OUT0"], "((IN0 XOR IN1) XOR IN2)")
        self.assertEqual(
            result.classification.expressions["OUT1"],
            "((IN0 AND IN1) OR ((IN0 XOR IN1) AND IN2))",
        )

        actual_connections = {
            (connection.source_gate, connection.target_gate, connection.target_input_index)
            for connection in result.graph.connections
        }
        expected_connections = {
            ("xor1", "xor2", 0),
            ("xor1", "and2", 0),
            ("and1", "or1", 0),
            ("and2", "or1", 1),
        }
        self.assertEqual(actual_connections, expected_connections)


class TopologyConfigurationTests(unittest.TestCase):
    def test_default_yolo_model_path_points_to_current_weights(self) -> None:
        pipeline = CircuitAnalysisPipeline()

        self.assertTrue(pipeline.model_path.exists(), f"Missing default model path: {pipeline.model_path}")
        self.assertIn("fixture_demo_best.pt", str(pipeline.model_path))

    def test_fixture_box_refinement_shrinks_oversized_live_box_to_gate_body(self) -> None:
        pipeline = CircuitAnalysisPipeline(refine_fixture_boxes=False)
        image_path = ROOT / "examples" / "schematics" / "full_adder_crossed.png"
        with Image.open(image_path) as image:
            refined = pipeline._refine_fixture_box(
                image.convert("L"),
                BoundingBox(477, 57, 662, 162),
                image.width,
                image.height,
            )

        self.assertGreaterEqual(refined.x1, 518)
        self.assertLessEqual(refined.x1, 526)
        self.assertGreaterEqual(refined.y1, 58)
        self.assertLessEqual(refined.y1, 64)
        self.assertGreaterEqual(refined.x2, 658)
        self.assertLessEqual(refined.x2, 662)
        self.assertGreaterEqual(refined.y2, 158)
        self.assertLessEqual(refined.y2, 162)


class TopologyLiveYoloFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        generate_example_schematics()
        cls.pipeline = CircuitAnalysisPipeline()

    def test_live_yolo_half_adder_fixture_classifies_correctly(self) -> None:
        result = self.pipeline.analyze(HALF_ADDER_IMAGE, detections=None)

        self.assertEqual(result.classification.label, "half_adder")
        self.assertEqual(len(result.gates), 2)

    def test_live_yolo_half_subtractor_fixture_classifies_correctly(self) -> None:
        result = self.pipeline.analyze(HALF_SUBTRACTOR_IMAGE, detections=None)

        self.assertEqual(result.classification.label, "half_subtractor")
        self.assertEqual(len(result.gates), 3)

    def test_live_yolo_full_adder_crossed_fixture_classifies_correctly(self) -> None:
        result = self.pipeline.analyze(FULL_ADDER_CROSSED_IMAGE, detections=None)

        self.assertEqual(result.classification.label, "full_adder")
        self.assertEqual(len(result.gates), 5)


if __name__ == "__main__":
    unittest.main()
