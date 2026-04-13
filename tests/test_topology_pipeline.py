import json
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image

from topology.generate_example_schematics import main as generate_example_schematics
from topology.graph_builder import GraphBuilder
from topology.pipeline import CircuitAnalysisPipeline
from topology.types import BoundingBox, GateDetection, GateReclassification, Terminal


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
REAL_BENCHMARK = ROOT / "benchmarks" / "symbol_gate_real_crop_benchmark.json"
HALF_ADDER_VLABS_IMAGE = ROOT / "data" / "real_schematics" / "half_adder_vlabs.png"
FULL_ADDER_TP_IMAGE = ROOT / "data" / "real_schematics" / "full_adder_tp.jpg"
DECODER_2X4_ASIC_IMAGE = ROOT / "data" / "real_schematics" / "decoder_2x4_asic.png"


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
    def test_primary_input_split_uses_spatial_bands_for_large_vertical_symbol_bus(self) -> None:
        inputs = [
            Terminal("xor_big", "XOR", "input", 1, (388.0, 23.5)),
            Terminal("xor_big", "XOR", "input", 2, (400.0, 41.4)),
            Terminal("and_top", "AND", "input", 1, (438.2, 126.5)),
            Terminal("and_mid", "AND", "input", 1, (438.4, 186.7)),
            Terminal("and_bot", "AND", "input", 0, (436.0, 226.0)),
            Terminal("and_bot", "AND", "input", 1, (436.0, 250.0)),
        ]

        groups = GraphBuilder._split_primary_input_terminals(inputs)

        self.assertEqual(
            [[(item.gate_id, item.index) for item in group] for group in groups],
            [
                [("xor_big", 1), ("xor_big", 2)],
                [("and_top", 1)],
                [("and_mid", 1)],
                [("and_bot", 0)],
                [("and_bot", 1)],
            ],
        )

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

    def test_symbol_style_analysis_replaces_detector_labels_before_topology(self) -> None:
        pipeline = CircuitAnalysisPipeline()
        fixture_detections = pipeline.load_detections_json(HALF_ADDER_DETECTIONS)
        wrong_detections = [
            GateDetection(item.gate_id, "OR", item.bbox, 0.25)
            for item in fixture_detections
        ]
        stub_classifier = Mock()
        stub_classifier.classify_detections.return_value = [
            GateReclassification(
                gate_id="xor1",
                detector_label="OR",
                detector_confidence=0.25,
                classifier_label="XOR",
                classifier_confidence=0.92,
                bbox=wrong_detections[0].bbox,
                top_k=[("XOR", 0.92), ("OR", 0.06)],
            ),
            GateReclassification(
                gate_id="and1",
                detector_label="OR",
                detector_confidence=0.25,
                classifier_label="AND",
                classifier_confidence=0.89,
                bbox=wrong_detections[1].bbox,
                top_k=[("AND", 0.89), ("OR", 0.08)],
            ),
        ]

        plain_result = pipeline.analyze(HALF_ADDER_IMAGE, detections=wrong_detections)
        reclassified_result = pipeline.analyze_symbol_style(
            HALF_ADDER_IMAGE,
            detections=wrong_detections,
            gate_classifier=stub_classifier,
        )

        self.assertEqual(plain_result.classification.label, "unknown")
        self.assertEqual(reclassified_result.classification.label, "half_adder")
        self.assertEqual([gate.gate_type for gate in reclassified_result.gates], ["XOR", "AND"])
        self.assertEqual(len(reclassified_result.reclassifications), 2)
        self.assertTrue(any("beta path" in warning for warning in reclassified_result.warnings))
        stub_classifier.classify_detections.assert_called_once()

    def test_symbol_style_analysis_requires_detections(self) -> None:
        pipeline = CircuitAnalysisPipeline()

        with self.assertRaises(ValueError):
            pipeline.analyze_symbol_style(HALF_ADDER_IMAGE, detections=[])

    def test_symbol_style_half_adder_vlabs_benchmark_boxes_classify_as_half_adder(self) -> None:
        pipeline = CircuitAnalysisPipeline()
        benchmark_items = json.loads(REAL_BENCHMARK.read_text())
        detections = [
            GateDetection(
                gate_id=item["id"],
                gate_type=item["label"],
                bbox=BoundingBox(*item["bbox"]),
                confidence=1.0,
            )
            for item in benchmark_items
            if item["image"] == "data/real_schematics/half_adder_vlabs.png"
        ]

        result = pipeline.analyze_symbol_style(HALF_ADDER_VLABS_IMAGE, detections=detections)

        self.assertEqual(result.classification.label, "half_adder")
        self.assertGreaterEqual(result.classification.confidence, 0.9)

    def test_symbol_style_full_adder_tp_benchmark_boxes_classify_as_full_adder(self) -> None:
        pipeline = CircuitAnalysisPipeline()
        benchmark_items = json.loads(REAL_BENCHMARK.read_text())
        detections = [
            GateDetection(
                gate_id=item["id"],
                gate_type=item["label"],
                bbox=BoundingBox(*item["bbox"]),
                confidence=1.0,
            )
            for item in benchmark_items
            if item["image"] == "data/real_schematics/full_adder_tp.jpg"
        ]

        result = pipeline.analyze_symbol_style(FULL_ADDER_TP_IMAGE, detections=detections)

        self.assertEqual(result.classification.label, "full_adder")
        self.assertEqual(len(result.graph.primary_inputs), 3)
        self.assertEqual(len(result.graph.primary_outputs), 2)
        self.assertEqual(result.classification.expressions["OUT0"], "(IN0 XOR IN1 XOR IN2)")
        self.assertEqual(
            result.classification.expressions["OUT1"],
            "((IN0 AND IN1) OR (IN0 AND IN2) OR (IN1 AND IN2))",
        )
        self.assertTrue(any("Repaired malformed 3-input full-adder fanout" in warning for warning in result.warnings))

    def test_symbol_style_decoder_2x4_benchmark_boxes_classify_as_decoder(self) -> None:
        pipeline = CircuitAnalysisPipeline()
        benchmark_items = json.loads(REAL_BENCHMARK.read_text())
        detections = [
            GateDetection(
                gate_id=item["id"],
                gate_type=item["label"],
                bbox=BoundingBox(*item["bbox"]),
                confidence=1.0,
            )
            for item in benchmark_items
            if item["image"] == "data/real_schematics/decoder_2x4_asic.png"
        ]

        result = pipeline.analyze_symbol_style(DECODER_2X4_ASIC_IMAGE, detections=detections)

        self.assertEqual(result.classification.label, "decoder_2to4")
        self.assertEqual(len(result.graph.primary_inputs), 2)
        self.assertEqual(len(result.graph.primary_outputs), 4)
        self.assertEqual(
            result.classification.expressions,
            {
                "OUT0": "(NOT(IN0) AND NOT(IN1))",
                "OUT1": "(NOT(IN0) AND IN1)",
                "OUT2": "(IN0 AND NOT(IN1))",
                "OUT3": "(IN0 AND IN1)",
            },
        )
        self.assertTrue(any("Repaired malformed 2-input decoder fanout" in warning for warning in result.warnings))

    def test_symbol_style_topology_normalization_trims_wide_curvy_real_gate_boxes(self) -> None:
        pipeline = CircuitAnalysisPipeline()
        normalized = pipeline._normalize_symbol_topology_gates(
            [
                GateDetection("xor", "XOR", BoundingBox(378.0, 0.0, 562.0, 75.0), 1.0),
                GateDetection("or", "OR", BoundingBox(563.85, 144.44, 696.47, 244.26), 1.0),
            ]
        )

        self.assertGreater(normalized[0].bbox.x1, 398.0)
        self.assertLess(normalized[0].bbox.x2, 522.0)
        self.assertGreater(normalized[1].bbox.x1, 586.0)
        self.assertLess(normalized[1].bbox.x2, 658.0)


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
