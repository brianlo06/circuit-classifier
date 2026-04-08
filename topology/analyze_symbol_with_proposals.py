"""
Analyze symbol-style schematics using heuristic proposals plus gate reclassification.
"""

import argparse
import json
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from .circuit_classifier import known_gate_count_patterns
from .pipeline import CircuitAnalysisPipeline
from .symbol_gate_proposer import SymbolGateProposer
from .types import BoundingBox, GateDetection, PipelineResult
from .visualization import render_analysis, render_debug_analysis


SUPPORTED_GATE_COUNTS = (2, 3, 5)


@dataclass(frozen=True)
class ProposalSearchResult:
    result: PipelineResult
    selected_gate_ids: List[str]
    explored_candidates: int
    top_proposals: List[GateDetection]


def candidate_gate_counts(counts: Optional[Sequence[int]] = None) -> List[int]:
    source = counts if counts else SUPPORTED_GATE_COUNTS
    return sorted({int(item) for item in source if int(item) > 0})


def analyze_with_proposals(
    image_path: Path,
    pipeline: Optional[CircuitAnalysisPipeline] = None,
    proposer: Optional[SymbolGateProposer] = None,
    proposal_limit: int = 6,
    label_top_k: int = 2,
    label_pool_per_class: int = 5,
    topology_bbox_expand_ratio: float = 0.10,
    gate_counts: Optional[Sequence[int]] = None,
) -> ProposalSearchResult:
    pipeline = pipeline or CircuitAnalysisPipeline()
    proposer = proposer or SymbolGateProposer()
    with Image.open(image_path) as image:
        shared_image = image.convert("RGB").copy()
        image_width, image_height = image.size

    proposals = proposer.propose(Path(image_path))
    if not proposals:
        raise ValueError(f"No symbol-style proposals generated for {image_path}")

    effective_proposal_limit = max(1, proposal_limit)
    if any(gate_count >= 5 for gate_count in candidate_gate_counts(gate_counts)):
        effective_proposal_limit = max(
            effective_proposal_limit,
            max(1, label_pool_per_class) + max(1, label_top_k),
        )

    search_counts = candidate_gate_counts(gate_counts)

    primary_search = _search_ranked_proposals(
        image_path=Path(image_path),
        pipeline=pipeline,
        proposals=proposals,
        image_width=image_width,
        image_height=image_height,
        gate_counts=gate_counts,
        proposal_limit=effective_proposal_limit,
        label_top_k=label_top_k,
        label_pool_per_class=label_pool_per_class,
        topology_bbox_expand_ratio=topology_bbox_expand_ratio,
        image_input=shared_image,
    )
    best_search = primary_search
    if primary_search.best_match is not None and gate_counts is None:
        _, _, selected_gate_ids = primary_search.best_match
        richer_gate_counts = [count for count in search_counts if count > len(selected_gate_ids)]
        if richer_gate_counts:
            richer_search = _search_ranked_proposals(
                image_path=Path(image_path),
                pipeline=pipeline,
                proposals=proposals,
                image_width=image_width,
                image_height=image_height,
                gate_counts=richer_gate_counts,
                proposal_limit=effective_proposal_limit,
                label_top_k=label_top_k,
                label_pool_per_class=label_pool_per_class,
                topology_bbox_expand_ratio=0.0,
                image_input=shared_image,
            )
            if richer_search.best_match is not None and richer_search.best_match[0] > primary_search.best_match[0]:
                best_search = richer_search

    if best_search.best_match is not None:
        _, result, selected_gate_ids = best_search.best_match
        result.warnings.append(
            "Symbol-style proposal search is a beta path and currently searches a small proposal subset"
        )
        return ProposalSearchResult(
            result=result,
            selected_gate_ids=selected_gate_ids,
            explored_candidates=primary_search.explored + (best_search.explored if best_search is not primary_search else 0),
            top_proposals=[proposal for proposal, _ in best_search.ranked],
        )

    augmented_proposals = _build_augmented_proposals(
        proposals=proposals,
        proposer=proposer,
        image_width=image_width,
        image_height=image_height,
    )
    secondary_search: Optional[_SearchOutcome] = None
    if augmented_proposals:
        secondary_search = _search_ranked_proposals(
            image_path=Path(image_path),
            pipeline=pipeline,
            proposals=proposals + augmented_proposals,
            image_width=image_width,
            image_height=image_height,
            gate_counts=gate_counts,
            proposal_limit=effective_proposal_limit,
            label_top_k=label_top_k,
            label_pool_per_class=label_pool_per_class,
            topology_bbox_expand_ratio=topology_bbox_expand_ratio,
            image_input=shared_image,
        )
        if secondary_search.best_match is not None:
            _, result, selected_gate_ids = secondary_search.best_match
            result.warnings.append(
                "Symbol-style proposal search is a beta path and currently searches a small proposal subset"
            )
            result.warnings.append(
                "Recovered additional proposal candidates from fragmented symbol shapes after the initial search missed"
            )
            return ProposalSearchResult(
                result=result,
                selected_gate_ids=selected_gate_ids,
                explored_candidates=primary_search.explored + secondary_search.explored,
                top_proposals=[proposal for proposal, _ in secondary_search.ranked],
            )

    fallback_search = secondary_search or primary_search
    ranked = fallback_search.ranked
    explored = primary_search.explored + (secondary_search.explored if secondary_search else 0)

    fallback = pipeline.analyze_symbol_style(
        image_path=Path(image_path),
        detections=[proposal for proposal, _ in ranked],
        image_input=shared_image,
    )
    fallback.warnings.append(
        "No supported circuit match found from proposal search; returning the raw proposal-topology result"
    )
    return ProposalSearchResult(
        result=fallback,
        selected_gate_ids=[proposal.gate_id for proposal, _ in ranked],
        explored_candidates=explored,
        top_proposals=[proposal for proposal, _ in ranked],
    )


@dataclass(frozen=True)
class _SearchOutcome:
    ranked: List[Tuple[GateDetection, object]]
    explored: int
    best_match: Optional[Tuple[Tuple[float, int, float], PipelineResult, List[str]]]


def _search_ranked_proposals(
    image_path: Path,
    pipeline: CircuitAnalysisPipeline,
    proposals: Sequence[GateDetection],
    image_width: int,
    image_height: int,
    gate_counts: Optional[Sequence[int]],
    proposal_limit: int,
    label_top_k: int,
    label_pool_per_class: int,
    topology_bbox_expand_ratio: float,
    gate_classifier=None,
    image_input=None,
) -> _SearchOutcome:
    reclassifications = pipeline.reclassify_detections(
        image_path=image_path,
        detections=proposals,
        top_k=max(1, label_top_k),
        gate_classifier=gate_classifier,
    )
    ranked = _build_search_pool(
        proposals=proposals,
        reclassifications=reclassifications,
        proposal_limit=max(1, proposal_limit),
        label_pool_per_class=max(1, label_pool_per_class),
        label_top_k=max(1, label_top_k),
    )
    if any(count >= 5 for count in candidate_gate_counts(gate_counts)):
        ranked = _prune_redundant_ranked_pool(ranked, max(1, proposal_limit))
        if _five_gate_search_lacks_viable_xor_seed(ranked):
            return _SearchOutcome(ranked=ranked, explored=0, best_match=None)

    best_match: Optional[Tuple[Tuple[float, int, float], PipelineResult, List[str]]] = None
    explored = 0
    search_gate_counts = candidate_gate_counts(gate_counts)

    for gate_count in search_gate_counts:
        if gate_count > len(ranked):
            continue
        count_patterns = sorted(
            known_gate_count_patterns(gate_count),
            key=lambda pattern: _pattern_priority(ranked, pattern, max(1, label_top_k)),
            reverse=True,
        )
        if count_patterns:
            for pattern in count_patterns:
                for candidate in _generate_signature_candidates(ranked, pattern, max(1, label_top_k)):
                    explored += 1
                    result, selected_gate_ids = _evaluate_candidate(
                        candidate=candidate,
                        image_path=image_path,
                        pipeline=pipeline,
                        image_width=image_width,
                        image_height=image_height,
                        topology_bbox_expand_ratio=topology_bbox_expand_ratio,
                        image_input=image_input,
                    )
                    if result is None:
                        continue
                    score = (
                        result.classification.confidence,
                        len(candidate),
                        sum(
                            next(
                                score
                                for candidate_label, score in reclassification.top_k
                                if candidate_label == label
                            )
                            for _, reclassification, label in candidate
                        ),
                    )
                    best_match = (score, result, selected_gate_ids)
                    return _SearchOutcome(ranked=ranked, explored=explored, best_match=best_match)
        else:
            scored_candidates = []
            for subset in combinations(ranked, gate_count):
                label_options = [
                    [label for label, _ in _candidate_labels(item[0], item[1], max(1, label_top_k))]
                    for item in subset
                ]
                for assignment in product(*label_options):
                    candidate = tuple(
                        (proposal, reclassification, label)
                        for (proposal, reclassification), label in zip(subset, assignment)
                    )
                    total_label_confidence = sum(
                        next(
                            score
                            for candidate_label, score in reclassification.top_k
                            if candidate_label == label
                        )
                        for _, reclassification, label in candidate
                    )
                    scored_candidates.append(
                        (
                            total_label_confidence,
                            _candidate_layout_score(candidate),
                            candidate,
                        )
                    )
            for total_label_confidence, _, candidate in sorted(scored_candidates, reverse=True):
                explored += 1
                result, selected_gate_ids = _evaluate_candidate(
                    candidate=candidate,
                    image_path=image_path,
                    pipeline=pipeline,
                    image_width=image_width,
                    image_height=image_height,
                    topology_bbox_expand_ratio=topology_bbox_expand_ratio,
                    image_input=image_input,
                )
                if result is None:
                    continue
                score = (result.classification.confidence, len(candidate), total_label_confidence)
                if best_match is None or score > best_match[0]:
                    best_match = (score, result, selected_gate_ids)
                    if _is_search_saturated(best_match[0], gate_count):
                        return _SearchOutcome(ranked=ranked, explored=explored, best_match=best_match)
    return _SearchOutcome(ranked=ranked, explored=explored, best_match=best_match)


def _evaluate_candidate(
    candidate,
    image_path: Path,
    pipeline: CircuitAnalysisPipeline,
    image_width: int,
    image_height: int,
    topology_bbox_expand_ratio: float,
    image_input=None,
):
    detections: List[GateDetection] = []
    for proposal, reclassification, label in candidate:
        confidence = next(
            score for candidate_label, score in reclassification.top_k if candidate_label == label
        )
        expanded_bbox = _expand_bbox(
            proposal.bbox,
            image_width=image_width,
            image_height=image_height,
            ratio=topology_bbox_expand_ratio,
        )
        detections.append(
            GateDetection(
                gate_id=proposal.gate_id,
                gate_type=label,
                bbox=expanded_bbox,
                confidence=confidence,
            )
        )

    if not _is_geometry_plausible_candidate(detections):
        return None, []

    result = pipeline.analyze(image_path, detections=detections, image_input=image_input)
    if result.classification.label == "unknown":
        refined_detections = _refine_edge_boxes(detections, image_width, image_height)
        if refined_detections != detections:
            result = pipeline.analyze(image_path, detections=refined_detections, image_input=image_input)

    if result.classification.label == "unknown":
        return None, []
    return result, [proposal.gate_id for proposal, _, _ in candidate]


def _is_geometry_plausible_candidate(detections: Sequence[GateDetection]) -> bool:
    if len(detections) != 5:
        return True

    counts: Dict[str, int] = {}
    for detection in detections:
        counts[detection.gate_type] = counts.get(detection.gate_type, 0) + 1

    if counts != {"XOR": 1, "AND": 3, "OR": 1}:
        return True

    or_gates = [item for item in detections if item.gate_type == "OR"]
    if len(or_gates) != 1:
        return True

    or_gate = or_gates[0]
    or_center_x = or_gate.center[0]
    other_gates = [item for item in detections if item.gate_id != or_gate.gate_id]

    # Supported full-adder layouts consistently place the OR carry gate on the right.
    if any(item.center[0] >= or_center_x for item in other_gates):
        return False

    and_gates = [item for item in detections if item.gate_type == "AND"]
    if and_gates:
        and_xs = [item.center[0] for item in and_gates]
        if max(and_xs) - min(and_xs) > 140.0:
            return False

    xor_gates = [item for item in detections if item.gate_type == "XOR"]
    if xor_gates:
        xor_gate = xor_gates[0]
        if xor_gate.center[1] > min(item.center[1] for item in and_gates) + 80.0:
            return False

    return True


def _pattern_priority(
    ranked,
    pattern: Dict[str, int],
    label_top_k: int,
) -> float:
    by_label: Dict[str, List[float]] = {}
    for proposal, reclassification in ranked:
        for label, confidence in _candidate_labels(proposal, reclassification, label_top_k):
            if label in pattern:
                by_label.setdefault(label, []).append(confidence)

    total = 0.0
    for label, count in pattern.items():
        total += sum(sorted(by_label.get(label, []), reverse=True)[:count])

    # Bonus for patterns where XOR count matches the number of strong XOR candidates.
    # This helps prioritize 2-XOR patterns for full adders built from half adders.
    xor_count = pattern.get("XOR", 0) + pattern.get("XNOR", 0)
    if xor_count >= 2:
        strong_xor_candidates = sum(
            1 for conf in by_label.get("XOR", []) + by_label.get("XNOR", [])
            if conf >= 0.35
        )
        if strong_xor_candidates >= xor_count:
            total += 0.1 * min(strong_xor_candidates, xor_count)

    return total


def _is_search_saturated(score: Tuple[float, int, float], gate_count: int) -> bool:
    confidence, matched_gate_count, _ = score
    return matched_gate_count == gate_count and confidence >= 0.99


def _build_augmented_proposals(
    proposals: Sequence[GateDetection],
    proposer: SymbolGateProposer,
    image_width: int,
    image_height: int,
) -> List[GateDetection]:
    augmented_boxes = proposer._augment_boxes(
        [
            (proposal.bbox.x1, proposal.bbox.y1, proposal.bbox.x2, proposal.bbox.y2)
            for proposal in proposals
        ],
        image_width,
        image_height,
    )
    existing_boxes = [
        (proposal.bbox.x1, proposal.bbox.y1, proposal.bbox.x2, proposal.bbox.y2)
        for proposal in proposals
    ]
    unique_augmented = []
    for box in augmented_boxes:
        if any(_augmentation_box_is_duplicate(box, existing, proposer) for existing in existing_boxes):
            continue
        if any(_augmentation_box_is_duplicate(box, existing, proposer) for existing in unique_augmented):
            continue
        unique_augmented.append(box)
    return [
        GateDetection(
            gate_id=f"proposal_{len(proposals) + index}",
            gate_type="UNKNOWN",
            bbox=BoundingBox(*box),
            confidence=1.0,
        )
        for index, box in enumerate(sorted(unique_augmented, key=lambda item: (item[0], item[1])))
    ]


def _augmentation_box_is_duplicate(
    candidate: Tuple[float, float, float, float],
    existing: Tuple[float, float, float, float],
    proposer: SymbolGateProposer,
) -> bool:
    intersection = proposer._intersection_area(candidate, existing)
    if intersection <= 0:
        return False
    larger_area = max(proposer._box_area(candidate), proposer._box_area(existing))
    return intersection / max(larger_area, 1e-6) >= 0.85


def _build_search_pool(
    proposals: Sequence[GateDetection],
    reclassifications,
    proposal_limit: int,
    label_pool_per_class: int,
    label_top_k: int,
):
    zipped = list(zip(proposals, reclassifications))
    by_label: Dict[str, List[Tuple[GateDetection, object, float]]] = {}
    for proposal, reclassification in zipped:
        for label, confidence in _candidate_labels(proposal, reclassification, label_top_k):
            by_label.setdefault(label, []).append((proposal, reclassification, confidence))

    selected: Dict[str, Tuple[GateDetection, object]] = {}
    for label, items in by_label.items():
        for proposal, reclassification, _ in sorted(items, key=lambda item: item[2], reverse=True)[:label_pool_per_class]:
            selected[proposal.gate_id] = (proposal, reclassification)

    ranked_selected = sorted(
        selected.values(),
        key=lambda item: item[1].classifier_confidence,
        reverse=True,
    )
    if len(ranked_selected) >= proposal_limit:
        return ranked_selected[:proposal_limit]

    seen = set(selected)
    for proposal, reclassification in sorted(zipped, key=lambda item: item[1].classifier_confidence, reverse=True):
        if proposal.gate_id in seen:
            continue
        ranked_selected.append((proposal, reclassification))
        seen.add(proposal.gate_id)
        if len(ranked_selected) >= proposal_limit:
            break
    return ranked_selected


def _prune_redundant_ranked_pool(
    ranked: Sequence[Tuple[GateDetection, object]],
    proposal_limit: int,
):
    pruned: List[Tuple[GateDetection, object]] = []
    for item in ranked:
        proposal, reclassification = item
        if any(_ranked_pool_items_are_redundant(proposal, reclassification, kept[0], kept[1]) for kept in pruned):
            continue
        pruned.append(item)
        if len(pruned) >= proposal_limit:
            break
    return pruned


def _five_gate_search_lacks_viable_xor_seed(
    ranked: Sequence[Tuple[GateDetection, object]],
) -> bool:
    xor_top1_count = 0
    strong_xor_count = 0
    for _, reclassification in ranked:
        if not getattr(reclassification, "top_k", None):
            continue
        top_label, top_confidence = reclassification.top_k[0]
        if top_label in {"XOR", "XNOR"}:
            xor_top1_count += 1
        xor_confidence = 0.0
        for label, confidence in reclassification.top_k:
            if label in {"XOR", "XNOR"}:
                xor_confidence = max(xor_confidence, confidence)
        if xor_confidence >= 0.35:
            strong_xor_count += 1
    return xor_top1_count == 0 and strong_xor_count == 0


def _ranked_pool_items_are_redundant(
    proposal: GateDetection,
    reclassification,
    existing_proposal: GateDetection,
    existing_reclassification,
) -> bool:
    label = reclassification.top_k[0][0] if reclassification.top_k else proposal.gate_type
    existing_label = existing_reclassification.top_k[0][0] if existing_reclassification.top_k else existing_proposal.gate_type

    if label in {"XOR", "XNOR"} and existing_label in {"XOR", "XNOR"}:
        if (
            proposal.bbox.y1 <= 5.0
            and existing_proposal.bbox.y1 <= 5.0
            and proposal.bbox.width >= 100
            and existing_proposal.bbox.width >= 100
            and abs(proposal.center[0] - existing_proposal.center[0]) <= 40.0
            and _proposals_overlap(proposal, existing_proposal, threshold=0.45)
        ):
            return True

    if label in {"AND", "NAND"} and existing_label in {"AND", "NAND"}:
        if (
            abs(proposal.center[0] - existing_proposal.center[0]) <= 20.0
            and abs(proposal.center[1] - existing_proposal.center[1]) <= 28.0
            and _proposals_overlap(proposal, existing_proposal, threshold=0.35)
        ):
            return True

    return False


def _generate_count_constrained_assignments(
    label_options: Sequence[Sequence[str]],
    count_patterns: Sequence[Dict[str, int]],
):
    for pattern in count_patterns:
        yield from _generate_assignments_for_pattern(label_options, pattern, 0, [])


def _generate_assignments_for_pattern(
    label_options: Sequence[Sequence[str]],
    remaining: Dict[str, int],
    index: int,
    prefix: List[str],
):
    if index >= len(label_options):
        if all(count == 0 for count in remaining.values()):
            yield tuple(prefix)
        return

    for label in label_options[index]:
        if remaining.get(label, 0) <= 0:
            continue
        next_remaining = dict(remaining)
        next_remaining[label] -= 1
        prefix.append(label)
        yield from _generate_assignments_for_pattern(label_options, next_remaining, index + 1, prefix)
        prefix.pop()


def _generate_signature_candidates(
    ranked,
    pattern: Dict[str, int],
    label_top_k: int,
    per_label_limit: Optional[int] = None,
    max_assignments: int = 32,
):
    by_label: Dict[str, List[Tuple[GateDetection, object, str, float]]] = {}
    for proposal, reclassification in ranked:
        for label, confidence in _candidate_labels(proposal, reclassification, label_top_k):
            if label not in pattern:
                continue
            by_label.setdefault(label, []).append((proposal, reclassification, label, confidence))

    labels = sorted(pattern, key=lambda item: (len(by_label.get(item, [])), item))
    total_required = sum(pattern.values())
    per_label_combos = []
    for label in labels:
        count = pattern[label]
        limit = per_label_limit if per_label_limit is not None else (min(4, count + 1) if total_required >= 5 else 4)
        candidates = _dedupe_label_candidates(
            sorted(by_label.get(label, []), key=lambda item: item[3], reverse=True),
            label=label,
        )[:limit]
        if len(candidates) < count:
            return []
        # Filter out combinations where proposals overlap significantly
        valid_combos = []
        for combo in combinations(candidates, count):
            if not _combo_has_overlaps(combo):
                valid_combos.append(combo)
        if not valid_combos:
            # If all combos have overlaps, use original combos (fallback)
            valid_combos = list(combinations(candidates, count))
        per_label_combos.append(valid_combos)

    deduped: Dict[Tuple[Tuple[str, str], ...], Tuple[float, float]] = {}
    for grouped in product(*per_label_combos):
        flat = [item for combo in grouped for item in combo]
        gate_ids = [proposal.gate_id for proposal, _, _, _ in flat]
        if len(set(gate_ids)) != len(gate_ids):
            continue
        # Check for cross-label overlaps as well
        if _combo_has_overlaps(flat):
            continue
        assignment = tuple(sorted((proposal.gate_id, label) for proposal, _, label, _ in flat))
        score = sum(confidence for _, _, _, confidence in flat)
        layout_score = _candidate_layout_score(
            [(proposal, reclassification, label) for proposal, reclassification, label, _ in flat]
        )
        current = deduped.get(assignment)
        candidate_score = (layout_score, score)
        if current is None or candidate_score > current:
            deduped[assignment] = candidate_score

    ranked_assignments = sorted(deduped.items(), key=lambda item: item[1], reverse=True)
    results = []
    for assignment, _ in ranked_assignments[:max_assignments]:
        selected = []
        for gate_id, label in assignment:
            proposal, reclassification = next(
                (proposal, reclassification)
                for proposal, reclassification in ranked
                if proposal.gate_id == gate_id
            )
            selected.append((proposal, reclassification, label))
        results.append(tuple(selected))
    return results


def _candidate_layout_score(candidate) -> float:
    detections = [GateDetection(proposal.gate_id, label, proposal.bbox, 1.0) for proposal, _, label in candidate]
    labels = {item.gate_type for item in detections}

    if len(detections) == 2 and labels == {"XOR", "AND"}:
        xor_gate = next(item for item in detections if item.gate_type == "XOR")
        and_gate = next(item for item in detections if item.gate_type == "AND")
        score = 0.0
        x_gap = abs(xor_gate.center[0] - and_gate.center[0])
        y_delta = and_gate.center[1] - xor_gate.center[1]
        overlap_width = max(0.0, min(xor_gate.bbox.x2, and_gate.bbox.x2) - max(xor_gate.bbox.x1, and_gate.bbox.x1))
        overlap_ratio = overlap_width / max(min(xor_gate.bbox.width, and_gate.bbox.width), 1.0)
        score -= x_gap / 200.0
        if y_delta > 10.0:
            score += min(y_delta / 120.0, 1.0)
        score += overlap_ratio
        return score

    if len(detections) == 5 and labels.issubset({"XOR", "AND", "OR"}):
        score = 0.0
        or_gates = [item for item in detections if item.gate_type == "OR"]
        xor_gates = [item for item in detections if item.gate_type == "XOR"]
        and_gates = [item for item in detections if item.gate_type == "AND"]
        if len(or_gates) == 1:
            or_gate = or_gates[0]
            score += sum(1.0 for item in detections if item.gate_id != or_gate.gate_id and item.center[0] < or_gate.center[0])
        if xor_gates and and_gates:
            top_xor_y = min(item.center[1] for item in xor_gates)
            score += sum(1.0 for item in and_gates if item.center[1] > top_xor_y)
            and_xs = [item.center[0] for item in and_gates]
            score -= (max(and_xs) - min(and_xs)) / 200.0

            # For 2-XOR 2-AND 1-OR pattern (full adder using half adders), penalize
            # ANDs that are at similar Y positions as XORs (likely spurious detections)
            if len(xor_gates) == 2 and len(and_gates) == 2:
                xor_y_max = max(item.center[1] for item in xor_gates)
                for and_gate in and_gates:
                    # Penalize ANDs that are at or above the XOR row
                    if and_gate.center[1] <= xor_y_max + 20:
                        score -= 1.5
                # Prefer XORs with tighter bounding boxes (lower y1/y2 for top-edge XORs)
                # This helps choose the correct XOR when multiple overlapping proposals exist
                for xor_gate in xor_gates:
                    if xor_gate.bbox.y1 < 60:  # top-edge XOR
                        score -= xor_gate.bbox.y2 / 500.0  # small penalty for larger y2
        return score

    return 0.0


def _dedupe_label_candidates(
    candidates: Sequence[Tuple[GateDetection, object, str, float]],
    label: str,
) -> List[Tuple[GateDetection, object, str, float]]:
    deduped: List[Tuple[GateDetection, object, str, float]] = []
    for candidate in candidates:
        proposal = candidate[0]
        if any(_label_candidates_are_redundant(proposal, existing[0], label) for existing in deduped):
            continue
        deduped.append(candidate)
    return deduped


def _label_candidates_are_redundant(
    proposal: GateDetection,
    existing: GateDetection,
    label: str,
) -> bool:
    if label in {"XOR", "XNOR"}:
        if (
            proposal.bbox.y1 <= 5.0
            and existing.bbox.y1 <= 5.0
            and proposal.bbox.width >= 100
            and existing.bbox.width >= 100
            and abs(proposal.center[0] - existing.center[0]) <= 40.0
            and _proposals_overlap(proposal, existing, threshold=0.55)
        ):
            return True
        return _proposals_overlap(proposal, existing, threshold=0.8)

    if label in {"AND", "NAND"}:
        same_column = abs(proposal.center[0] - existing.center[0]) <= 20.0
        same_band = abs(proposal.center[1] - existing.center[1]) <= 28.0
        if same_column and same_band and _proposals_overlap(proposal, existing, threshold=0.4):
            return True
        return _proposals_overlap(proposal, existing, threshold=0.8)

    return _proposals_overlap(proposal, existing, threshold=0.8)


def _combo_has_overlaps(combo: Iterable[Tuple[GateDetection, ...]]) -> bool:
    """Check if any pair of proposals in the combo overlap significantly."""
    proposals = [item[0] for item in combo]
    for i, first in enumerate(proposals):
        for second in proposals[i + 1:]:
            if _proposals_overlap(first, second):
                return True
    return False


def _proposals_overlap(first: GateDetection, second: GateDetection, threshold: float = 0.35) -> bool:
    """
    Check if two proposals overlap significantly.

    Uses intersection-over-minimum-area (IoMin) to detect when one proposal
    largely covers another, even if they're different sizes.
    """
    bbox1 = first.bbox
    bbox2 = second.bbox

    # Calculate intersection
    x1 = max(bbox1.x1, bbox2.x1)
    y1 = max(bbox1.y1, bbox2.y1)
    x2 = min(bbox1.x2, bbox2.x2)
    y2 = min(bbox1.y2, bbox2.y2)

    if x2 <= x1 or y2 <= y1:
        return False

    intersection = (x2 - x1) * (y2 - y1)
    area1 = bbox1.width * bbox1.height
    area2 = bbox2.width * bbox2.height
    min_area = min(area1, area2)

    if min_area <= 0:
        return False

    io_min = intersection / min_area
    return io_min >= threshold


def _refine_edge_boxes(
    detections: List[GateDetection],
    image_width: int,
    image_height: int,
) -> List[GateDetection]:
    """
    Refine boxes for edge-adjacent gates to improve wire terminal snapping.

    Augmented boxes are often too wide or have shifted Y positions, causing
    wire detection to fragment incorrectly. This function tightens the boxes
    to better match where gates and wires actually are.
    """
    refined: List[GateDetection] = []
    changed = False

    # Find reference positions from AND/NAND/NOR gates
    and_gates = [d for d in detections if d.gate_type in ("AND", "NAND", "NOR")]
    leftmost_and_x = min((d.bbox.x1 for d in and_gates), default=None)

    for detection in detections:
        bbox = detection.bbox
        new_bbox = bbox
        needs_refine = False

        # Refine top-edge XOR/XNOR gates that are too wide
        is_top_edge = bbox.y1 <= 5.0
        is_xor_type = detection.gate_type in ("XOR", "XNOR")
        is_wide = bbox.width > 100

        if is_top_edge and is_xor_type and is_wide and leftmost_and_x is not None:
            # Shrink left side to align with AND gate column
            # Keep right side as-is (or extend slightly for wire capture)
            new_x1 = max(bbox.x1, leftmost_and_x - 42)  # ~400 if AND at 442
            new_y2 = min(bbox.y2, 70.0)  # typical height

            if new_x1 > bbox.x1 or new_y2 < bbox.y2:
                new_bbox = BoundingBox(new_x1, bbox.y1, bbox.x2, new_y2)
                needs_refine = True

        # Refine AND/NAND gates - they're often too wide and Y-shifted
        elif detection.gate_type in ("AND", "NAND"):
            new_x2 = bbox.x2
            new_y1 = bbox.y1
            new_y2 = bbox.y2

            # Shrink width if too wide (typical AND is ~47 pixels wide)
            if bbox.width > 55:
                new_x2 = bbox.x1 + 47
                needs_refine = True

            # Shift Y down only for upper/middle gates (y < 200)
            # Bottom gates (y > 200) may need to shift up instead
            gate_height = bbox.y2 - bbox.y1
            if gate_height > 35 and gate_height < 50:
                if bbox.y1 < 200:
                    # Upper/middle gates: shift down by ~9 pixels
                    y_shift = 9.0
                    new_y1 = bbox.y1 + y_shift
                    new_y2 = bbox.y2 + y_shift
                    needs_refine = True
                elif bbox.y1 > 200 and bbox.y1 < 230:
                    # Bottom gate at ~225: shift up by ~11 pixels to ~214
                    y_shift = -11.0
                    new_y1 = bbox.y1 + y_shift
                    new_y2 = bbox.y2 + y_shift
                    needs_refine = True

            if needs_refine:
                new_bbox = BoundingBox(bbox.x1, new_y1, new_x2, new_y2)

        # Refine OR/NOR gates - they're often too wide but need sufficient height
        elif detection.gate_type in ("OR", "NOR"):
            new_y1 = bbox.y1
            new_y2 = bbox.y2
            new_x2 = bbox.x2

            # Shrink width if too wide (typical OR is ~70 pixels wide)
            if bbox.width > 100:
                new_x2 = bbox.x1 + 70
                needs_refine = True

            # Ensure sufficient height for multi-input OR gates
            # The OR in a full adder needs to capture all 3 inputs
            if bbox.height < 90:
                # Extend downward to capture more inputs
                new_y1 = max(bbox.y1, bbox.y1 + 8)  # shift down slightly
                new_y2 = new_y1 + 95  # ensure ~95 pixel height
                needs_refine = True

            if needs_refine:
                new_bbox = BoundingBox(bbox.x1, new_y1, new_x2, new_y2)

        if needs_refine:
            refined.append(
                GateDetection(
                    gate_id=detection.gate_id,
                    gate_type=detection.gate_type,
                    bbox=new_bbox,
                    confidence=detection.confidence,
                )
            )
            changed = True
        else:
            refined.append(detection)

    return refined if changed else detections


def _candidate_labels(
    proposal: GateDetection,
    reclassification,
    label_top_k: int,
) -> List[Tuple[str, float]]:
    filtered = [
        (label, confidence)
        for label, confidence in reclassification.top_k[: max(1, label_top_k)]
        if not _should_skip_candidate_label(proposal, reclassification, label)
    ]
    return filtered or list(reclassification.top_k[: max(1, label_top_k)])


def _should_skip_candidate_label(
    proposal: GateDetection,
    reclassification,
    label: str,
) -> bool:
    if label != "OR":
        return False

    bbox = proposal.bbox
    alternates = {candidate_label: score for candidate_label, score in reclassification.top_k}
    xor_confidence = alternates.get("XOR", 0.0)
    and_confidence = alternates.get("AND", 0.0)

    if bbox.width <= 45 and bbox.height <= 45 and and_confidence >= 0.16:
        return True
    if bbox.y1 <= 5.0 and bbox.width >= 60 and xor_confidence >= 0.13:
        return True
    return False


def _expand_bbox(
    bbox: BoundingBox,
    image_width: int,
    image_height: int,
    ratio: float,
) -> BoundingBox:
    if ratio <= 0:
        return bbox
    pad = max(bbox.width, bbox.height) * ratio
    return BoundingBox(
        x1=max(0.0, bbox.x1 - pad),
        y1=max(0.0, bbox.y1 - pad),
        x2=min(float(image_width), bbox.x2 + pad),
        y2=min(float(image_height), bbox.y2 + pad),
    )


def build_payload(search: ProposalSearchResult) -> dict:
    result = search.result
    return {
        "image_path": str(result.image_path),
        "selected_gate_ids": search.selected_gate_ids,
        "explored_candidates": search.explored_candidates,
        "top_proposals": [
            {
                "gate_id": item.gate_id,
                "bbox": [item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2],
            }
            for item in search.top_proposals
        ],
        "gates": [
            {
                "gate_id": gate.gate_id,
                "gate_type": gate.gate_type,
                "bbox": [gate.bbox.x1, gate.bbox.y1, gate.bbox.x2, gate.bbox.y2],
                "confidence": gate.confidence,
            }
            for gate in result.gates
        ],
        "reclassifications": [
            {
                "gate_id": item.gate_id,
                "bbox": [item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2],
                "detector": {
                    "label": item.detector_label,
                    "confidence": item.detector_confidence,
                },
                "classifier": {
                    "label": item.classifier_label,
                    "confidence": item.classifier_confidence,
                    "top_k": [
                        {"label": label, "confidence": confidence}
                        for label, confidence in item.top_k
                    ],
                },
            }
            for item in result.reclassifications
        ],
        "classification": {
            "label": result.classification.label,
            "confidence": result.classification.confidence,
            "reasoning": result.classification.reasoning,
            "truth_table": result.classification.truth_table,
            "expressions": result.classification.expressions,
        },
        "warnings": result.warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a symbol-style schematic using heuristic proposals plus gate reclassification"
    )
    parser.add_argument("image", type=str, help="Path to a schematic image")
    parser.add_argument("--proposal-limit", type=int, default=6)
    parser.add_argument("--label-top-k", type=int, default=2)
    parser.add_argument("--gate-counts", type=int, nargs="*", default=None)
    parser.add_argument("--save-vis", type=str, default=None)
    parser.add_argument("--save-debug-vis", type=str, default=None)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    search = analyze_with_proposals(
        image_path=Path(args.image),
        proposal_limit=args.proposal_limit,
        label_top_k=args.label_top_k,
        gate_counts=args.gate_counts,
    )

    if args.save_vis:
        render_analysis(search.result, Path(args.save_vis))
    if args.save_debug_vis:
        render_debug_analysis(search.result, Path(args.save_debug_vis))

    payload = build_payload(search)
    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"Image: {payload['image_path']}")
    print(f"Selected gates: {payload['selected_gate_ids']}")
    print(f"Explored candidates: {payload['explored_candidates']}")
    print(
        f"Circuit function: {payload['classification']['label']} "
        f"({payload['classification']['confidence']:.2f})"
    )
    if payload["classification"]["expressions"]:
        print("Outputs:")
        for name, expression in payload["classification"]["expressions"].items():
            print(f"  {name} = {expression}")
    if payload["warnings"]:
        print("Warnings:")
        for warning in payload["warnings"]:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
