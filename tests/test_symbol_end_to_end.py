import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from topology.analyze_symbol_with_proposals import (
    ProposalSearchResult,
    _SearchOutcome,
    _trim_primary_proposals_for_search,
    _trim_augmented_proposals_for_search,
    _is_geometry_plausible_candidate,
    _search_ranked_proposals,
    _generate_count_constrained_assignments,
    candidate_gate_counts,
    _looks_like_larger_decoder_family,
    _should_keep_signature_candidate,
    analyze_with_proposals,
)
from topology.pipeline import CircuitAnalysisPipeline
from topology.run_symbol_end_to_end_benchmark import load_cases
from topology.types import BoundingBox, ClassificationResult, GateDetection


class SymbolEndToEndTests(unittest.TestCase):
    def test_count_constrained_assignments_only_emit_supported_multiset(self) -> None:
        assignments = list(
            _generate_count_constrained_assignments(
                [["XOR", "AND"], ["XOR", "AND"], ["OR", "AND"]],
                [{"XOR": 2, "AND": 0, "OR": 1}],
            )
        )

        self.assertEqual(assignments, [("XOR", "XOR", "OR")])

    def test_candidate_gate_counts_deduplicates_and_sorts(self) -> None:
        self.assertEqual(candidate_gate_counts([5, 2, 2, 3]), [2, 3, 5])

    def test_two_gate_geometry_rejects_vertical_fragment_overlap(self) -> None:
        detections = [
            GateDetection("xor_fragment", "XOR", BoundingBox(279, 3, 314, 30), 1.0),
            GateDetection("and_fragment", "AND", BoundingBox(297, 17, 326, 83), 1.0),
        ]

        self.assertFalse(_is_geometry_plausible_candidate(detections))

    def test_signature_candidate_filter_rejects_low_value_two_gate_layout(self) -> None:
        candidate = (
            (
                GateDetection("and_gate", "UNKNOWN", BoundingBox(194, 61, 221, 91), 1.0),
                SimpleNamespace(top_k=[("AND", 0.41), ("XOR", 0.08)], classifier_confidence=0.41),
                "AND",
            ),
            (
                GateDetection("xor_gate", "UNKNOWN", BoundingBox(279, 3, 314, 30), 1.0),
                SimpleNamespace(top_k=[("XOR", 0.14), ("OR", 0.18)], classifier_confidence=0.18),
                "XOR",
            ),
        )

        self.assertFalse(_should_keep_signature_candidate(candidate, layout_score=0.05))

    def test_larger_decoder_family_detector_rejects_tall_multi_output_column(self) -> None:
        proposals = [
            GateDetection("not_0", "UNKNOWN", BoundingBox(70, 0, 110, 35), 1.0),
            GateDetection("not_1", "UNKNOWN", BoundingBox(70, 40, 110, 75), 1.0),
        ]
        proposals.extend(
            GateDetection(f"and_{index}", "UNKNOWN", BoundingBox(220, 20 + index * 55, 254, 62 + index * 55), 1.0)
            for index in range(6)
        )
        reclassifications = [
            SimpleNamespace(top_k=[("NOT", 0.5)], classifier_confidence=0.5),
            SimpleNamespace(top_k=[("NOT", 0.49)], classifier_confidence=0.49),
        ]
        reclassifications.extend(
            SimpleNamespace(top_k=[("AND", 0.8)], classifier_confidence=0.8)
            for _ in range(6)
        )

        self.assertTrue(_looks_like_larger_decoder_family(proposals, reclassifications, [6]))

    def test_larger_decoder_family_detector_does_not_block_supported_decoder_shape(self) -> None:
        proposals = [
            GateDetection("not_0", "UNKNOWN", BoundingBox(70, 0, 110, 35), 1.0),
        ]
        proposals.extend(
            GateDetection(f"and_{index}", "UNKNOWN", BoundingBox(220, 20 + index * 40, 250, 67 + index * 40), 1.0)
            for index in range(5)
        )
        reclassifications = [SimpleNamespace(top_k=[("NOT", 0.5)], classifier_confidence=0.5)]
        reclassifications.extend(
            SimpleNamespace(top_k=[("AND", 0.8)], classifier_confidence=0.8)
            for _ in range(5)
        )

        self.assertFalse(_looks_like_larger_decoder_family(proposals, reclassifications, [6]))

    def test_large_gate_search_trims_augmented_pool_toward_full_adder_recovery_shapes(self) -> None:
        proposals = [
            GateDetection("left_low", "UNKNOWN", BoundingBox(281.4, 267.5, 371.1, 300.0), 1.0),
            GateDetection("top_xor", "UNKNOWN", BoundingBox(377.6, 0.0, 512.1, 77.3), 1.0),
            GateDetection("and_top", "UNKNOWN", BoundingBox(441.8, 83.9, 504.2, 126.1), 1.0),
            GateDetection("and_mid", "UNKNOWN", BoundingBox(441.8, 143.9, 504.2, 186.1), 1.0),
            GateDetection("and_low", "UNKNOWN", BoundingBox(440.2, 224.9, 504.6, 267.1), 1.0),
            GateDetection("wide_lower", "UNKNOWN", BoundingBox(425.4, 248.0, 583.4, 289.0), 1.0),
            GateDetection("top_left_small", "UNKNOWN", BoundingBox(327.2, 13.5, 381.8, 49.5), 1.0),
            GateDetection("top_mid", "UNKNOWN", BoundingBox(387.6, 0.0, 453.9, 77.3), 1.0),
            GateDetection("lower_right", "UNKNOWN", BoundingBox(441.8, 203.9, 504.2, 246.1), 1.0),
        ]

        trimmed = _trim_augmented_proposals_for_search(
            proposals,
            search_counts=[5],
            image_width=725,
            image_height=300,
        )
        kept_ids = {proposal.gate_id for proposal in trimmed}

        self.assertLessEqual(len(trimmed), 8)
        self.assertIn("top_xor", kept_ids)
        self.assertIn("and_top", kept_ids)
        self.assertIn("and_mid", kept_ids)
        self.assertIn("and_low", kept_ids)
        self.assertNotIn("wide_lower", kept_ids)

    def test_two_gate_primary_trim_discards_left_fragment_bias_for_vlabs_case(self) -> None:
        proposals = [
            GateDetection("left_0", "UNKNOWN", BoundingBox(0, 3, 32, 39), 1.0),
            GateDetection("left_1", "UNKNOWN", BoundingBox(0, 36, 30, 73), 1.0),
            GateDetection("left_2", "UNKNOWN", BoundingBox(20, 6, 48, 35), 1.0),
            GateDetection("mid_top", "UNKNOWN", BoundingBox(165, 1, 227, 43), 1.0),
            GateDetection("mid_mid", "UNKNOWN", BoundingBox(169, 31, 228, 73), 1.0),
            GateDetection("mid_low", "UNKNOWN", BoundingBox(176, 127, 218, 166), 1.0),
            GateDetection("right_top", "UNKNOWN", BoundingBox(248, 23, 276, 52), 1.0),
            GateDetection("right_low", "UNKNOWN", BoundingBox(268, 111, 300, 148), 1.0),
            GateDetection("right_top2", "UNKNOWN", BoundingBox(268, 19, 299, 56), 1.0),
            GateDetection("extra_0", "UNKNOWN", BoundingBox(120, 0, 159, 26), 1.0),
            GateDetection("extra_1", "UNKNOWN", BoundingBox(120, 48, 149, 79), 1.0),
        ]

        trimmed = _trim_primary_proposals_for_search(
            proposals,
            search_gate_counts=[2],
            image_path=Path("data/real_schematics/half_adder_vlabs.png"),
        )
        kept_ids = {proposal.gate_id for proposal in trimmed}

        self.assertEqual(len(trimmed), 10)
        self.assertTrue({"left_0", "left_1", "left_2"} - kept_ids)
        self.assertIn("mid_top", kept_ids)
        self.assertIn("right_low", kept_ids)

    def test_end_to_end_case_manifest_defaults_to_supported_cases(self) -> None:
        cases = load_cases(Path("benchmarks/symbol_end_to_end_cases.json"), ["supported"])

        self.assertEqual(len(cases), 6)
        self.assertTrue(all(case["status"] == "supported" for case in cases))

    def test_end_to_end_case_manifest_can_include_candidates(self) -> None:
        cases = load_cases(Path("benchmarks/symbol_end_to_end_cases.json"), ["supported", "candidate"])

        self.assertGreater(len(cases), 6)
        self.assertTrue(any(case["status"] == "candidate" for case in cases))

    def test_full_adder_using_half_adders_classifies_via_symbol_beta_path(self) -> None:
        search = analyze_with_proposals(
            image_path=Path("data/real_schematics/full_adder_using_half_adders.jpg"),
            proposal_limit=6,
            label_top_k=3,
            gate_counts=[5],
            topology_bbox_expand_ratio=0.0,
        )

        self.assertEqual(search.result.classification.label, "full_adder")
        self.assertIn("aliasing split inputs", search.result.classification.reasoning)

    def test_search_prefers_full_circuit_match_over_equal_confidence_subcircuit(self) -> None:
        proposals = [
            GateDetection("proposal_0", "UNKNOWN", BoundingBox(0, 0, 10, 10), 1.0),
            GateDetection("proposal_1", "UNKNOWN", BoundingBox(20, 0, 30, 10), 1.0),
            GateDetection("proposal_2", "UNKNOWN", BoundingBox(40, 0, 50, 10), 1.0),
            GateDetection("proposal_3", "UNKNOWN", BoundingBox(60, 0, 70, 10), 1.0),
            GateDetection("proposal_4", "UNKNOWN", BoundingBox(80, 0, 90, 10), 1.0),
        ]
        reclassifications = [
            SimpleNamespace(gate_id="proposal_0", top_k=[("XOR", 0.95)], classifier_confidence=0.95),
            SimpleNamespace(gate_id="proposal_1", top_k=[("AND", 0.94)], classifier_confidence=0.94),
            SimpleNamespace(gate_id="proposal_2", top_k=[("XOR", 0.93)], classifier_confidence=0.93),
            SimpleNamespace(gate_id="proposal_3", top_k=[("AND", 0.92)], classifier_confidence=0.92),
            SimpleNamespace(gate_id="proposal_4", top_k=[("OR", 0.91)], classifier_confidence=0.91),
        ]

        class FakePipeline:
            def reclassify_detections(self, **kwargs):
                return reclassifications

            def analyze(self, image_path, detections, image_input=None):
                if len(detections) == 2:
                    return SimpleNamespace(
                        classification=ClassificationResult(
                            label="half_adder",
                            confidence=0.9,
                            reasoning="two-gate subcircuit",
                        )
                    )
                if len(detections) == 5:
                    return SimpleNamespace(
                        classification=ClassificationResult(
                            label="full_adder",
                            confidence=0.9,
                            reasoning="five-gate full circuit",
                        )
                    )
                return SimpleNamespace(
                    classification=ClassificationResult(
                        label="unknown",
                        confidence=0.0,
                        reasoning="unknown",
                    )
                )

        outcome = _search_ranked_proposals(
            image_path=Path("data/real_schematics/full_adder_using_half_adders.jpg"),
            pipeline=FakePipeline(),
            proposals=proposals,
            image_width=100,
            image_height=100,
            gate_counts=[2, 5],
            proposal_limit=5,
            label_top_k=1,
            label_pool_per_class=5,
            topology_bbox_expand_ratio=0.0,
        )

        self.assertIsNotNone(outcome.best_match)
        _, result, selected_gate_ids = outcome.best_match
        self.assertEqual(result.classification.label, "full_adder")
        self.assertEqual(selected_gate_ids, [f"proposal_{index}" for index in range(5)])

    def test_fallback_prefers_augmented_ranked_pool_when_secondary_search_runs(self) -> None:
        image_path = Path("data/real_schematics/half_adder_vlabs.png")
        primary_proposals = [GateDetection("proposal_0", "UNKNOWN", BoundingBox(100, 0, 170, 40), 1.0)]
        augmented_proposals = [GateDetection("proposal_1", "UNKNOWN", BoundingBox(110, 80, 160, 130), 1.0)]

        class DummyResult:
            classification = ClassificationResult("unknown", 0.0, "unknown")
            warnings = []

        def fake_search(*_, **kwargs):
            proposals = kwargs["proposals"]
            ranked = [
                (
                    proposal,
                    SimpleNamespace(
                        gate_id=proposal.gate_id,
                        top_k=[("XOR", 0.5), ("AND", 0.4)] if proposal.gate_id == "proposal_0" else [("AND", 0.5), ("XOR", 0.4)],
                        classifier_confidence=0.5,
                        classifier_label="XOR" if proposal.gate_id == "proposal_0" else "AND",
                    ),
                )
                for proposal in proposals
            ]
            return _SearchOutcome(ranked=ranked, explored=1, best_match=None, debug_stats={})

        with patch("topology.analyze_symbol_with_proposals.SymbolGateProposer.propose", return_value=primary_proposals), \
             patch("topology.analyze_symbol_with_proposals._build_augmented_proposals", return_value=augmented_proposals), \
             patch("topology.analyze_symbol_with_proposals._search_ranked_proposals", side_effect=fake_search), \
             patch("topology.analyze_symbol_with_proposals.CircuitAnalysisPipeline.analyze_symbol_style", return_value=DummyResult()) as fallback_mock:
            analyze_with_proposals(image_path=image_path, proposal_limit=2, gate_counts=[2])

        fallback_mock.assert_called_once()
        self.assertEqual(
            fallback_mock.call_args.kwargs["detections"],
            primary_proposals + augmented_proposals,
        )

    def test_aggressive_recovery_only_runs_after_primary_search_miss(self) -> None:
        image_path = Path("data/real_schematics/half_adder_vlabs.png")
        proposals = [GateDetection("proposal_0", "UNKNOWN", BoundingBox(0, 0, 10, 10), 1.0)]
        primary_result = SimpleNamespace(
            classification=ClassificationResult("half_adder", 0.95, "primary match"),
            warnings=[],
        )
        primary_outcome = _SearchOutcome(
            ranked=[(proposals[0], object())],
            explored=2,
            best_match=((0.95, 2, 1.9), primary_result, ["proposal_0", "proposal_1"]),
            debug_stats={},
        )

        with patch("topology.analyze_symbol_with_proposals.SymbolGateProposer.propose", return_value=proposals), \
             patch("topology.analyze_symbol_with_proposals._search_ranked_proposals", return_value=primary_outcome), \
             patch("topology.analyze_symbol_with_proposals._build_augmented_proposals") as augment_mock:
            search = analyze_with_proposals(image_path=image_path, proposal_limit=2, gate_counts=[2])

        augment_mock.assert_not_called()
        self.assertEqual(search.result.classification.label, "half_adder")

    def test_aggressive_recovery_uses_post_miss_augmented_tier(self) -> None:
        image_path = Path("data/real_schematics/half_adder_vlabs.png")
        primary_proposals = [GateDetection("proposal_0", "UNKNOWN", BoundingBox(0, 0, 10, 10), 1.0)]
        secondary_augmented_proposals = [GateDetection("proposal_1", "UNKNOWN", BoundingBox(20, 0, 30, 10), 1.0)]
        aggressive_augmented_proposals = [GateDetection("proposal_2", "UNKNOWN", BoundingBox(40, 0, 50, 10), 1.0)]
        recovered_result = SimpleNamespace(
            classification=ClassificationResult("half_adder", 0.9, "recovered"),
            warnings=[],
        )
        primary_outcome = _SearchOutcome(
            ranked=[(primary_proposals[0], object())],
            explored=2,
            best_match=None,
            debug_stats={},
        )
        secondary_outcome = _SearchOutcome(
            ranked=[
                (
                    proposal,
                    SimpleNamespace(
                        gate_id=proposal.gate_id,
                        classifier_label="AND",
                        classifier_confidence=0.4,
                        top_k=[("AND", 0.4), ("XOR", 0.3)],
                    ),
                )
                for proposal in primary_proposals + secondary_augmented_proposals
            ],
            explored=3,
            best_match=None,
            debug_stats={},
        )
        aggressive_outcome = _SearchOutcome(
            ranked=[
                (
                    proposal,
                    SimpleNamespace(
                        gate_id=proposal.gate_id,
                        classifier_label="AND",
                        classifier_confidence=0.4,
                        top_k=[("AND", 0.4), ("XOR", 0.3)],
                    ),
                )
                for proposal in primary_proposals + aggressive_augmented_proposals
            ],
            explored=3,
            best_match=((0.9, 2, 1.7), recovered_result, ["proposal_0", "proposal_2"]),
            debug_stats={},
        )

        with patch("topology.analyze_symbol_with_proposals.SymbolGateProposer.propose", return_value=primary_proposals), \
             patch("topology.analyze_symbol_with_proposals._search_ranked_proposals", side_effect=[primary_outcome, secondary_outcome, aggressive_outcome]), \
             patch("topology.analyze_symbol_with_proposals._fallback_subset_cannot_match_supported_pattern", return_value=False), \
             patch(
                 "topology.analyze_symbol_with_proposals._build_augmented_proposals",
                 side_effect=[secondary_augmented_proposals, aggressive_augmented_proposals],
             ) as augment_mock:
            search = analyze_with_proposals(image_path=image_path, proposal_limit=2, gate_counts=[2])

        self.assertEqual(augment_mock.call_count, 2)
        self.assertEqual(
            augment_mock.call_args_list[1].kwargs,
            {
                "proposals": primary_proposals,
                "proposer": unittest.mock.ANY,
                "image_width": unittest.mock.ANY,
                "image_height": unittest.mock.ANY,
                "aggressive": True,
            },
        )
        self.assertEqual(search.result.classification.label, "half_adder")
        self.assertIn(
            "Recovered additional proposal candidates from fragmented symbol shapes after the initial search missed",
            search.result.warnings,
        )

    def test_secondary_recovery_match_skips_aggressive_tier(self) -> None:
        image_path = Path("data/real_schematics/half_adder_vlabs.png")
        primary_proposals = [GateDetection("proposal_0", "UNKNOWN", BoundingBox(0, 0, 10, 10), 1.0)]
        secondary_augmented_proposals = [GateDetection("proposal_1", "UNKNOWN", BoundingBox(20, 0, 30, 10), 1.0)]
        recovered_result = SimpleNamespace(
            classification=ClassificationResult("half_adder", 0.9, "secondary recovered"),
            warnings=[],
        )
        primary_outcome = _SearchOutcome(
            ranked=[(primary_proposals[0], object())],
            explored=2,
            best_match=None,
            debug_stats={},
        )
        secondary_outcome = _SearchOutcome(
            ranked=[(proposal, object()) for proposal in primary_proposals + secondary_augmented_proposals],
            explored=3,
            best_match=((0.9, 2, 1.7), recovered_result, ["proposal_0", "proposal_1"]),
            debug_stats={},
        )

        with patch("topology.analyze_symbol_with_proposals.SymbolGateProposer.propose", return_value=primary_proposals), \
             patch("topology.analyze_symbol_with_proposals._search_ranked_proposals", side_effect=[primary_outcome, secondary_outcome]), \
             patch("topology.analyze_symbol_with_proposals._build_augmented_proposals", return_value=secondary_augmented_proposals) as augment_mock:
            search = analyze_with_proposals(image_path=image_path, proposal_limit=2, gate_counts=[2])

        augment_mock.assert_called_once_with(
            proposals=primary_proposals,
            proposer=unittest.mock.ANY,
            image_width=unittest.mock.ANY,
            image_height=unittest.mock.ANY,
        )
        self.assertEqual(search.result.classification.label, "half_adder")
        self.assertEqual(search.selected_gate_ids, ["proposal_0", "proposal_1"])
        self.assertIn(
            "Recovered additional proposal candidates from fragmented symbol shapes after the initial search missed",
            search.result.warnings,
        )

    def test_secondary_impossible_subset_skips_aggressive_tier(self) -> None:
        image_path = Path("data/real_schematics/xor_basic_gates.png")
        primary_proposals = [GateDetection("proposal_0", "UNKNOWN", BoundingBox(0, 0, 20, 20), 1.0)]
        secondary_augmented_proposals = [GateDetection("proposal_1", "UNKNOWN", BoundingBox(30, 30, 60, 60), 1.0)]
        primary_outcome = _SearchOutcome(
            ranked=[],
            explored=0,
            best_match=None,
            debug_stats={},
        )
        secondary_outcome = _SearchOutcome(
            ranked=[
                (
                    proposal,
                    SimpleNamespace(
                        gate_id=proposal.gate_id,
                        classifier_label="AND",
                        classifier_confidence=0.4,
                        top_k=[("AND", 0.4), ("OR", 0.2)],
                    ),
                )
                for proposal in primary_proposals + secondary_augmented_proposals
            ],
            explored=0,
            best_match=None,
            debug_stats={},
        )

        with patch("topology.analyze_symbol_with_proposals.SymbolGateProposer.propose", return_value=primary_proposals), \
             patch("topology.analyze_symbol_with_proposals._search_ranked_proposals", side_effect=[primary_outcome, secondary_outcome]), \
             patch("topology.analyze_symbol_with_proposals._build_augmented_proposals", return_value=secondary_augmented_proposals) as augment_mock:
            search = analyze_with_proposals(image_path=image_path, proposal_limit=2, gate_counts=[5])

        augment_mock.assert_called_once()
        self.assertEqual(search.result.classification.label, "unknown")
        self.assertIn(
            "Skipped aggressive proposal recovery because the secondary proposal subset could not form a geometry-plausible supported pattern",
            search.result.warnings,
        )

    def test_generic_search_retries_larger_gate_counts_before_returning_subcircuit(self) -> None:
        image_path = Path("data/real_schematics/full_adder_using_half_adders.jpg")
        proposals = [GateDetection("proposal_0", "UNKNOWN", BoundingBox(0, 0, 10, 10), 1.0)]

        two_gate_result = SimpleNamespace(
            classification=ClassificationResult("half_adder", 0.9, "two-gate subcircuit"),
            warnings=[],
        )
        five_gate_result = SimpleNamespace(
            classification=ClassificationResult("full_adder", 0.9, "five-gate circuit"),
            warnings=[],
        )
        primary_outcome = _SearchOutcome(
            ranked=[(proposals[0], object())],
            explored=4,
            best_match=((0.9, 2, 1.8), two_gate_result, ["proposal_0", "proposal_1"]),
            debug_stats={},
        )
        richer_outcome = _SearchOutcome(
            ranked=[(proposals[0], object())],
            explored=8,
            best_match=((0.9, 5, 1.5), five_gate_result, [f"proposal_{index}" for index in range(5)]),
            debug_stats={},
        )

        with patch("topology.analyze_symbol_with_proposals.SymbolGateProposer.propose", return_value=proposals), \
             patch("topology.analyze_symbol_with_proposals._search_ranked_proposals", side_effect=[primary_outcome, richer_outcome]):
            search = analyze_with_proposals(image_path=image_path, proposal_limit=2)

        self.assertEqual(search.result.classification.label, "full_adder")
        self.assertEqual(search.selected_gate_ids, [f"proposal_{index}" for index in range(5)])
        self.assertEqual(search.explored_candidates, 12)

    def test_pipeline_reuses_cached_gate_classifier_between_reclassification_calls(self) -> None:
        pipeline = CircuitAnalysisPipeline()
        detections = [GateDetection("proposal_0", "UNKNOWN", BoundingBox(0, 0, 10, 10), 1.0)]
        fake_classifier = SimpleNamespace(
            classify_detections=lambda **kwargs: [
                SimpleNamespace(
                    gate_id="proposal_0",
                    detector_label="UNKNOWN",
                    detector_confidence=1.0,
                    classifier_label="AND",
                    classifier_confidence=0.9,
                    bbox=detections[0].bbox,
                    top_k=[("AND", 0.9)],
                )
            ]
        )

        with patch("topology.pipeline.GateCropClassifier", return_value=fake_classifier) as classifier_cls:
            first = pipeline.reclassify_detections(Path("image_a.png"), detections)
            second = pipeline.reclassify_detections(Path("image_b.png"), detections)

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        classifier_cls.assert_called_once()

    def test_geometry_filter_rejects_full_adder_candidate_with_non_rightmost_or(self) -> None:
        detections = [
            GateDetection("xor_0", "XOR", BoundingBox(40, 0, 80, 40), 0.9),
            GateDetection("and_0", "AND", BoundingBox(90, 60, 130, 100), 0.9),
            GateDetection("and_1", "AND", BoundingBox(90, 120, 130, 160), 0.9),
            GateDetection("and_2", "AND", BoundingBox(90, 180, 130, 220), 0.9),
            GateDetection("or_0", "OR", BoundingBox(70, 110, 120, 170), 0.9),
        ]

        self.assertFalse(_is_geometry_plausible_candidate(detections))


if __name__ == "__main__":
    unittest.main()
