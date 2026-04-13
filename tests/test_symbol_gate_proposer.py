import unittest

from pathlib import Path
import json
from PIL import Image

from topology.analyze_symbol_with_proposals import _build_augmented_proposals
from topology.pipeline import CircuitAnalysisPipeline
from topology.run_symbol_proposal_benchmark import bbox_iou
from topology.symbol_gate_proposer import SymbolGateProposer


class SymbolGateProposerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        benchmark_path = Path("benchmarks/symbol_gate_real_crop_benchmark.json")
        cls.benchmark_entries = json.loads(benchmark_path.read_text())

    def test_bbox_iou_returns_zero_for_disjoint_boxes(self) -> None:
        self.assertEqual(bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_bbox_iou_returns_expected_overlap(self) -> None:
        score = bbox_iou((0, 0, 10, 10), (5, 5, 15, 15))
        self.assertAlmostEqual(score, 25 / 175)

    def test_full_adder_tp_proposals_include_recoverable_xor_and_and_candidates(self) -> None:
        image_path = Path("data/real_schematics/full_adder_tp.jpg")
        proposer = SymbolGateProposer()
        proposals = proposer.propose(image_path)
        proposals += _build_augmented_proposals(proposals, proposer, image_width=700, image_height=300)
        pipeline = CircuitAnalysisPipeline()
        reclassifications = pipeline.reclassify_detections(image_path, proposals, top_k=3)

        top_k_labels = {
            proposal.gate_id: {label for label, _ in reclassification.top_k[:3]}
            for proposal, reclassification in zip(proposals, reclassifications)
        }

        self.assertTrue(
            any(
                bbox_iou((378.0, 0.0, 562.0, 75.0), proposal.bbox.to_int_tuple()) >= 0.35
                and "XOR" in top_k_labels[proposal.gate_id]
                for proposal in proposals
            )
        )
        self.assertTrue(
            any(
                bbox_iou((440.0, 148.0, 493.0, 202.0), proposal.bbox.to_int_tuple()) >= 0.45
                and "AND" in top_k_labels[proposal.gate_id]
                for proposal in proposals
            )
        )

    def test_proposer_recovers_selected_hard_benchmark_cases(self) -> None:
        expected_ids = {
            "half_adder_vlabs_manual_and_gate_0",
            "xnor_ansi_gate_0",
            "xor_nand_gates_gate_1",
            "xnor_nand_gates_gfg_gate_2",
            "xnor_nand_gates_gfg_gate_4",
            "xnor_nand_gates_gfg_gate_5",
        }
        entries = [item for item in self.benchmark_entries if item["id"] in expected_ids]
        proposer = SymbolGateProposer()

        for entry in entries:
            image_path = Path(entry["image"])
            proposals = proposer.propose(image_path)
            with Image.open(image_path) as image:
                image_width, image_height = image.size
            proposals += _build_augmented_proposals(
                proposals,
                proposer,
                image_width=image_width,
                image_height=image_height,
                aggressive=True,
            )
            best_iou = max(
                (bbox_iou(entry["bbox"], proposal.bbox.to_int_tuple()) for proposal in proposals),
                default=0.0,
            )
            self.assertGreaterEqual(best_iou, 0.2, entry["id"])

    def test_decoder_2x4_asic_augmented_proposals_recover_two_not_and_four_and_candidates(self) -> None:
        image_path = Path("data/real_schematics/decoder_2x4_asic.png")
        proposer = SymbolGateProposer()
        proposals = proposer.propose(image_path)
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        proposals += _build_augmented_proposals(
            proposals,
            proposer,
            image_width=image_width,
            image_height=image_height,
            aggressive=True,
        )
        pipeline = CircuitAnalysisPipeline()
        reclassifications = pipeline.reclassify_detections(image_path, proposals, top_k=3)

        benchmark_entries = [
            entry
            for entry in self.benchmark_entries
            if entry["image"] == "data/real_schematics/decoder_2x4_asic.png"
        ]
        recovered = 0
        for entry in benchmark_entries:
            if any(
                bbox_iou(entry["bbox"], proposal.bbox.to_int_tuple()) >= 0.2
                and entry["label"] in {label for label, _ in reclassification.top_k[:3]}
                for proposal, reclassification in zip(proposals, reclassifications)
            ):
                recovered += 1

        self.assertGreaterEqual(recovered, 6)


if __name__ == "__main__":
    unittest.main()
