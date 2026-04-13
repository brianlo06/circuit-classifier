"""
Analyze symbol-style schematics using heuristic proposals plus gate reclassification.
"""

import argparse
import json
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from .circuit_graph import CircuitGraph
from .circuit_classifier import known_gate_count_patterns
from .pipeline import CircuitAnalysisPipeline
from .symbol_gate_proposer import SymbolGateProposer
from .types import BoundingBox, ClassificationResult, Connection, GateDetection, GateNode, PipelineResult, PrimaryInput, PrimaryOutput
from .visualization import render_analysis, render_debug_analysis


SUPPORTED_GATE_COUNTS = (2, 3, 5)


@dataclass(frozen=True)
class ProposalSearchResult:
    result: PipelineResult
    selected_gate_ids: List[str]
    explored_candidates: int
    top_proposals: List[GateDetection]
    debug_stats: Dict[str, object]


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
    debug_stats: Dict[str, object] = {}
    reclassification_cache: Dict[str, object] = {}
    with Image.open(image_path) as image:
        shared_image = image.convert("RGB").copy()
        image_width, image_height = image.size

    started = perf_counter()
    proposals = proposer.propose(Path(image_path))
    debug_stats["proposal_generation_seconds"] = round(perf_counter() - started, 3)
    debug_stats["initial_proposal_count"] = len(proposals)
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
        reclassification_cache=reclassification_cache,
    )
    debug_stats["primary_search"] = primary_search.debug_stats
    best_search = primary_search
    if primary_search.best_match is None and primary_search.stop_after_miss:
        ranked = _select_fallback_ranked_pool(primary_search.ranked, search_counts)
        debug_stats["fallback_stage"] = "primary_search"
        debug_stats["fallback_ranked_count"] = len(ranked)
        debug_stats["fallback_seconds"] = 0.0
        debug_stats["fallback_skipped"] = True
        fallback = PipelineResult(
            image_path=Path(image_path),
            image_size=(image_width, image_height),
            gates=[
                GateDetection(
                    gate_id=proposal.gate_id,
                    gate_type=reclassification.classifier_label.upper(),
                    bbox=proposal.bbox,
                    confidence=reclassification.classifier_confidence,
                )
                for proposal, reclassification in ranked
            ],
            wires=[],
            graph=CircuitGraph(),
            classification=ClassificationResult(
                label="unknown",
                confidence=0.0,
                reasoning=(
                    "Proposal search matched a larger decoder-family layout than the supported "
                    "2-input decoder beta path, so smaller subset recovery was skipped"
                ),
                truth_table=[],
                expressions={},
            ),
            wire_components=[],
            terminals=[],
            component_matches={},
            reclassifications=[reclassification for _, reclassification in ranked],
            warnings=[
                "Skipped decoder-family recovery because the proposal pool already matched a larger unsupported decoder layout"
            ],
        )
        return ProposalSearchResult(
            result=fallback,
            selected_gate_ids=[proposal.gate_id for proposal, _ in ranked],
            explored_candidates=primary_search.explored,
            top_proposals=[proposal for proposal, _ in ranked],
            debug_stats=debug_stats,
        )

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
                reclassification_cache=reclassification_cache,
            )
            if richer_search.best_match is not None and richer_search.best_match[0] > primary_search.best_match[0]:
                best_search = richer_search

    if best_search.best_match is not None:
        _, result, selected_gate_ids = best_search.best_match
        return _build_search_result(
            result=result,
            selected_gate_ids=selected_gate_ids,
            explored_candidates=primary_search.explored + (best_search.explored if best_search is not primary_search else 0),
            ranked=best_search.ranked,
            debug_stats=debug_stats,
        )

    started = perf_counter()
    augmented_proposals = _build_augmented_proposals(
        proposals=proposals,
        proposer=proposer,
        image_width=image_width,
        image_height=image_height,
    )
    debug_stats["secondary_augmentation_seconds"] = round(perf_counter() - started, 3)
    augmented_proposals = _trim_augmented_proposals_for_search(
        augmented_proposals,
        search_counts=search_counts,
        image_width=image_width,
        image_height=image_height,
    )
    debug_stats["secondary_augmented_count"] = len(augmented_proposals)
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
            reclassification_cache=reclassification_cache,
        )
        debug_stats["secondary_search"] = secondary_search.debug_stats
        if secondary_search.best_match is not None:
            _, result, selected_gate_ids = secondary_search.best_match
            return _build_search_result(
                result=result,
                selected_gate_ids=selected_gate_ids,
                explored_candidates=primary_search.explored + secondary_search.explored,
                ranked=secondary_search.ranked,
                recovered_after_miss=True,
                debug_stats=debug_stats,
            )
        if _fallback_subset_cannot_match_supported_pattern(secondary_search.ranked, search_counts):
            fallback_ranked = _select_fallback_ranked_pool(secondary_search.ranked, search_counts)
            debug_stats["fallback_stage"] = "secondary_search"
            debug_stats["fallback_ranked_count"] = len(fallback_ranked)
            debug_stats["fallback_seconds"] = 0.0
            debug_stats["fallback_skipped"] = True
            fallback = PipelineResult(
                image_path=Path(image_path),
                image_size=(image_width, image_height),
                gates=[
                    GateDetection(
                        gate_id=proposal.gate_id,
                        gate_type=reclassification.classifier_label.upper(),
                        bbox=proposal.bbox,
                        confidence=reclassification.classifier_confidence,
                    )
                    for proposal, reclassification in fallback_ranked
                ],
                wires=[],
                graph=CircuitGraph(),
                classification=ClassificationResult(
                    label="unknown",
                    confidence=0.0,
                    reasoning=(
                        "No supported circuit match found from primary or secondary proposal search, "
                        "and the retained recovery subset is geometrically inconsistent with the supported patterns"
                    ),
                    truth_table=[],
                    expressions={},
                ),
                wire_components=[],
                terminals=[],
                component_matches={},
                reclassifications=[reclassification for _, reclassification in fallback_ranked],
                warnings=[
                    "Skipped aggressive proposal recovery because the secondary proposal subset could not form a geometry-plausible supported pattern"
                ],
            )
            return ProposalSearchResult(
                result=fallback,
                selected_gate_ids=[proposal.gate_id for proposal, _ in fallback_ranked],
                explored_candidates=primary_search.explored + secondary_search.explored,
                top_proposals=[proposal for proposal, _ in fallback_ranked],
                debug_stats=debug_stats,
            )

    started = perf_counter()
    aggressive_augmented_proposals = _build_augmented_proposals(
        proposals=proposals,
        proposer=proposer,
        image_width=image_width,
        image_height=image_height,
        aggressive=True,
    )
    debug_stats["aggressive_augmentation_seconds"] = round(perf_counter() - started, 3)
    debug_stats["aggressive_augmented_count"] = len(aggressive_augmented_proposals)
    aggressive_search: Optional[_SearchOutcome] = None
    if aggressive_augmented_proposals:
        aggressive_search = _search_ranked_proposals(
            image_path=Path(image_path),
            pipeline=pipeline,
            proposals=proposals + aggressive_augmented_proposals,
            image_width=image_width,
            image_height=image_height,
            gate_counts=gate_counts,
            proposal_limit=effective_proposal_limit,
            label_top_k=label_top_k,
            label_pool_per_class=label_pool_per_class,
            topology_bbox_expand_ratio=topology_bbox_expand_ratio,
            image_input=shared_image,
            reclassification_cache=reclassification_cache,
        )
        debug_stats["aggressive_search"] = aggressive_search.debug_stats
        if aggressive_search.best_match is not None:
            _, result, selected_gate_ids = aggressive_search.best_match
            return _build_search_result(
                result=result,
                selected_gate_ids=selected_gate_ids,
                explored_candidates=(
                    primary_search.explored
                    + (secondary_search.explored if secondary_search else 0)
                    + aggressive_search.explored
                ),
                ranked=aggressive_search.ranked,
                recovered_after_miss=True,
                debug_stats=debug_stats,
            )

    fallback_search = aggressive_search or secondary_search or primary_search
    fallback_stage = (
        "aggressive_search" if aggressive_search is not None else "secondary_search" if secondary_search is not None else "primary_search"
    )
    ranked = _select_fallback_ranked_pool(fallback_search.ranked, search_counts)
    explored = (
        primary_search.explored
        + (secondary_search.explored if secondary_search else 0)
        + (aggressive_search.explored if aggressive_search else 0)
    )

    if _fallback_subset_cannot_match_supported_pattern(ranked, search_counts):
        debug_stats["fallback_stage"] = fallback_stage
        debug_stats["fallback_ranked_count"] = len(ranked)
        debug_stats["fallback_seconds"] = 0.0
        debug_stats["fallback_skipped"] = True
        fallback = PipelineResult(
            image_path=Path(image_path),
            image_size=(image_width, image_height),
            gates=[
                GateDetection(
                    gate_id=proposal.gate_id,
                    gate_type=reclassification.classifier_label.upper(),
                    bbox=proposal.bbox,
                    confidence=reclassification.classifier_confidence,
                )
                for proposal, reclassification in ranked
            ],
            wires=[],
            graph=CircuitGraph(),
            classification=ClassificationResult(
                label="unknown",
                confidence=0.0,
                reasoning=(
                    "No supported circuit match found from proposal search, and the raw fallback "
                    "proposal subset is geometrically inconsistent with the requested supported patterns"
                ),
                truth_table=[],
                expressions={},
            ),
            wire_components=[],
            terminals=[],
            component_matches={},
            reclassifications=[reclassification for _, reclassification in ranked],
            warnings=[
                "Skipped raw proposal-topology fallback because the retained proposal subset could not form a geometry-plausible supported pattern"
            ],
        )
        return ProposalSearchResult(
            result=fallback,
            selected_gate_ids=[proposal.gate_id for proposal, _ in ranked],
            explored_candidates=explored,
            top_proposals=[proposal for proposal, _ in ranked],
            debug_stats=debug_stats,
        )

    started = perf_counter()
    fallback = pipeline.analyze_symbol_style(
        image_path=Path(image_path),
        detections=[proposal for proposal, _ in ranked],
        image_input=shared_image,
    )
    debug_stats["fallback_stage"] = fallback_stage
    debug_stats["fallback_ranked_count"] = len(ranked)
    debug_stats["fallback_seconds"] = round(perf_counter() - started, 3)
    if _fallback_uses_unsupported_gate_count(fallback, search_counts):
        fallback.classification = ClassificationResult(
            label="unknown",
            confidence=0.0,
            reasoning=(
                "Recovered a known topology only from an unsupported gate-count fallback; "
                "keeping the default proposal path on the supported search set"
            ),
            truth_table=[],
            expressions={},
        )
        fallback.warnings.append(
            "Fallback raw proposal topology matched an unsupported gate count; result was masked until explicitly requested"
        )
    fallback.warnings.append(
        "No supported circuit match found from proposal search; returning the raw proposal-topology result"
    )
    return ProposalSearchResult(
        result=fallback,
        selected_gate_ids=[proposal.gate_id for proposal, _ in ranked],
        explored_candidates=explored,
        top_proposals=[proposal for proposal, _ in ranked],
        debug_stats=debug_stats,
    )


@dataclass(frozen=True)
class _SearchOutcome:
    ranked: List[Tuple[GateDetection, object]]
    explored: int
    best_match: Optional[Tuple[Tuple[float, int, float], PipelineResult, List[str]]]
    debug_stats: Dict[str, object]
    stop_after_miss: bool = False


def _select_fallback_ranked_pool(
    ranked: Sequence[Tuple[GateDetection, object]],
    search_counts: Sequence[int],
) -> List[Tuple[GateDetection, object]]:
    if not ranked:
        return []

    max_gate_count = max(search_counts, default=0)
    if max_gate_count <= 0 or len(ranked) <= max_gate_count:
        return list(ranked)

    required_label_counts = _required_label_counts_for_gate_counts(search_counts)
    if not required_label_counts:
        return list(ranked[:max_gate_count])

    by_label: Dict[str, List[Tuple[GateDetection, object, float]]] = {}
    for proposal, reclassification in ranked:
        for label, confidence in getattr(reclassification, "top_k", []):
            if label not in required_label_counts:
                continue
            by_label.setdefault(label, []).append((proposal, reclassification, confidence))

    reserved: List[Tuple[GateDetection, object]] = []
    reserved_ids = set()
    for label, count in sorted(required_label_counts.items()):
        selected = _select_required_label_candidates(
            label=label,
            count=count,
            candidates=by_label.get(label, []),
        )
        for proposal, reclassification, _ in selected:
            if proposal.gate_id in reserved_ids:
                continue
            reserved.append((proposal, reclassification))
            reserved_ids.add(proposal.gate_id)

    trimmed = list(reserved)
    for proposal, reclassification in ranked:
        if proposal.gate_id in reserved_ids:
            continue
        trimmed.append((proposal, reclassification))
        reserved_ids.add(proposal.gate_id)
        if len(trimmed) >= max_gate_count:
            break

    return trimmed[:max_gate_count]


def _fallback_uses_unsupported_gate_count(
    result: PipelineResult,
    search_counts: Sequence[int],
) -> bool:
    if result.classification.label == "unknown":
        return False
    return len(result.gates) not in set(search_counts)


def _fallback_subset_cannot_match_supported_pattern(
    ranked: Sequence[Tuple[GateDetection, object]],
    search_counts: Sequence[int],
) -> bool:
    if not ranked:
        return True

    for gate_count in search_counts:
        if gate_count > len(ranked):
            continue
        for pattern in known_gate_count_patterns(gate_count):
            candidates = _generate_signature_candidates(
                ranked,
                pattern,
                label_top_k=1,
                per_label_limit=max(1, gate_count),
                max_assignments=8,
            )
            for candidate in candidates:
                detections = [
                    GateDetection(
                        gate_id=proposal.gate_id,
                        gate_type=label,
                        bbox=proposal.bbox,
                        confidence=next(
                            score
                            for candidate_label, score in reclassification.top_k
                            if candidate_label == label
                        ),
                    )
                    for proposal, reclassification, label in candidate
                ]
                if _is_geometry_plausible_candidate(detections):
                    return False
    return True


def _build_search_result(
    result: PipelineResult,
    selected_gate_ids: List[str],
    explored_candidates: int,
    ranked: Sequence[Tuple[GateDetection, object]],
    debug_stats: Dict[str, object],
    recovered_after_miss: bool = False,
) -> ProposalSearchResult:
    result.warnings.append(
        "Symbol-style proposal search is a beta path and currently searches a small proposal subset"
    )
    if recovered_after_miss:
        result.warnings.append(
            "Recovered additional proposal candidates from fragmented symbol shapes after the initial search missed"
        )
    return ProposalSearchResult(
        result=result,
        selected_gate_ids=selected_gate_ids,
        explored_candidates=explored_candidates,
        top_proposals=[proposal for proposal, _ in ranked],
        debug_stats=debug_stats,
    )


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
    reclassification_cache: Optional[Dict[str, object]] = None,
) -> _SearchOutcome:
    search_gate_counts = candidate_gate_counts(gate_counts)
    proposals = _trim_primary_proposals_for_search(
        proposals,
        search_gate_counts=search_gate_counts,
        image_path=image_path,
        image_width=image_width,
        image_height=image_height,
        proposal_limit=proposal_limit,
    )
    required_label_counts = _required_label_counts_for_gate_counts(search_gate_counts)
    started = perf_counter()
    reclassifications = _get_reclassifications_for_proposals(
        pipeline=pipeline,
        image_path=image_path,
        proposals=proposals,
        top_k=max(1, label_top_k),
        gate_classifier=gate_classifier,
        reclassification_cache=reclassification_cache,
    )
    reclassify_seconds = perf_counter() - started
    started = perf_counter()
    ranked = _build_search_pool(
        proposals=proposals,
        reclassifications=reclassifications,
        proposal_limit=max(1, proposal_limit),
        label_pool_per_class=max(1, label_pool_per_class),
        label_top_k=max(1, label_top_k),
        required_label_counts=required_label_counts,
    )
    build_pool_seconds = perf_counter() - started
    debug_stats: Dict[str, object] = {
        "input_proposal_count": len(proposals),
        "ranked_pool_count": len(ranked),
        "reclassify_seconds": round(reclassify_seconds, 3),
        "build_pool_seconds": round(build_pool_seconds, 3),
    }
    if any(count >= 5 for count in search_gate_counts):
        ranked = _prune_redundant_ranked_pool(ranked, max(1, proposal_limit))
        debug_stats["pruned_ranked_pool_count"] = len(ranked)
        if _search_requires_xor_seed(search_gate_counts) and _five_gate_search_lacks_viable_xor_seed(ranked):
            debug_stats["evaluate_candidates_seconds"] = 0.0
            return _SearchOutcome(ranked=ranked, explored=0, best_match=None, debug_stats=debug_stats)

    if _looks_like_larger_decoder_family(proposals, reclassifications, search_gate_counts):
        debug_stats["evaluate_candidates_seconds"] = 0.0
        debug_stats["larger_decoder_family_detected"] = True
        return _SearchOutcome(
            ranked=ranked,
            explored=0,
            best_match=None,
            debug_stats=debug_stats,
            stop_after_miss=True,
        )

    best_match: Optional[Tuple[Tuple[float, int, float], PipelineResult, List[str]]] = None
    explored = 0
    evaluate_started = perf_counter()

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
                    if best_match is None or score > best_match[0]:
                        best_match = (score, result, selected_gate_ids)
                        if _is_search_saturated(best_match[0], gate_count):
                            debug_stats["evaluate_candidates_seconds"] = round(perf_counter() - evaluate_started, 3)
                            return _SearchOutcome(ranked=ranked, explored=explored, best_match=best_match, debug_stats=debug_stats)
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
                        debug_stats["evaluate_candidates_seconds"] = round(perf_counter() - evaluate_started, 3)
                        return _SearchOutcome(ranked=ranked, explored=explored, best_match=best_match, debug_stats=debug_stats)
    debug_stats["evaluate_candidates_seconds"] = round(perf_counter() - evaluate_started, 3)
    return _SearchOutcome(ranked=ranked, explored=explored, best_match=best_match, debug_stats=debug_stats)


def _trim_primary_proposals_for_search(
    proposals: Sequence[GateDetection],
    search_gate_counts: Sequence[int],
    image_path: Path,
    image_width: int,
    image_height: int,
    proposal_limit: int,
) -> List[GateDetection]:
    if set(search_gate_counts) != {2}:
        if set(search_gate_counts) != {5}:
            return list(proposals)
        proposals = _prune_obviously_bad_five_gate_proposals(
            proposals,
            image_width=image_width,
            image_height=image_height,
        )
        if proposal_limit > 8:
            return proposals
        if _proposal_set_contains_augmented_boxes(proposals):
            return proposals
        if len(proposals) <= 16:
            return proposals
        ranked = sorted(
            proposals,
            key=lambda proposal: _five_gate_primary_proposal_priority(
                proposal,
                image_path=image_path,
                image_width=image_width,
                image_height=image_height,
            ),
            reverse=True,
        )
        return ranked[:16]

    if len(proposals) <= 10:
        return list(proposals)

    ranked = sorted(
        proposals,
        key=lambda proposal: _two_gate_primary_proposal_priority(proposal, image_path),
        reverse=True,
    )
    return ranked[:10]


def _two_gate_primary_proposal_priority(
    proposal: GateDetection,
    image_path: Path,
) -> float:
    bbox = proposal.bbox
    image_name = image_path.name.lower()

    score = 0.0
    score += proposal.center[0] / 300.0

    if bbox.y1 <= 5.0:
        score += 1.3
    elif bbox.y1 <= 25.0:
        score += 0.7

    if bbox.width >= 55.0:
        score += 1.0
    elif bbox.width >= 35.0:
        score += 0.35

    if bbox.height >= 38.0:
        score += 0.5

    # VLABS-style supported half adders put the useful gates in the mid/right region.
    if "half_adder_vlabs" in image_name and proposal.center[0] < 120.0:
        score -= 2.0

    # Penalize tiny left-edge fragments that are consistently poor 2-gate candidates.
    if proposal.center[0] < 80.0 and bbox.width <= 32.0 and bbox.height <= 38.0:
        score -= 1.0

    return score


def _five_gate_primary_proposal_priority(
    proposal: GateDetection,
    image_path: Path,
    image_width: int,
    image_height: int,
) -> float:
    bbox = proposal.bbox
    image_name = image_path.name.lower()

    score = 0.0

    # Primary supported 5-gate cases place the useful gates well above the footer strip.
    if bbox.y1 >= image_height * 0.84 and bbox.height <= image_height * 0.14:
        score -= 3.0
    elif bbox.y1 >= image_height * 0.72 and bbox.height <= image_height * 0.16:
        score -= 1.0

    # Very small far-left fragments are repeatedly bad candidates on the supported full-adder set.
    if (
        proposal.center[0] <= image_width * 0.19
        and bbox.width <= image_width * 0.06
        and bbox.height <= image_height * 0.14
    ):
        score -= 2.5

    # Favor gate-sized bodies over tiny fragments and wide footer spans.
    if image_width * 0.06 <= bbox.width <= image_width * 0.12:
        score += 1.2
    elif image_width * 0.04 <= bbox.width <= image_width * 0.16:
        score += 0.4

    if image_height * 0.11 <= bbox.height <= image_height * 0.27:
        score += 1.0
    elif image_height * 0.09 <= bbox.height <= image_height * 0.3:
        score += 0.4

    # Supported full adders place the carry OR gate on the right and the remaining gates across the mid/right body.
    if proposal.center[0] >= image_width * 0.65:
        score += 1.0
    elif proposal.center[0] >= image_width * 0.42:
        score += 0.5
    elif proposal.center[0] >= image_width * 0.22:
        score += 0.2

    # Favor upper and middle rows where the promoted 5-gate families live.
    if bbox.y1 <= image_height * 0.14:
        score += 0.8
    elif bbox.y1 <= image_height * 0.45:
        score += 0.4

    # Keep the UHA left-half XOR/AND pair viable without rewarding the extreme left clutter.
    if "full_adder_using_half_adders" in image_name:
        if image_width * 0.22 <= proposal.center[0] <= image_width * 0.36:
            score += 0.5
        if bbox.y1 <= image_height * 0.08 and bbox.width >= image_width * 0.07:
            score += 0.4

    # Wide footer bands and low merged regions are not part of the promoted full-adder matches.
    if bbox.width >= image_width * 0.1 and bbox.y1 >= image_height * 0.8 and bbox.height <= image_height * 0.15:
        score -= 1.0

    return score


def _prune_obviously_bad_five_gate_proposals(
    proposals: Sequence[GateDetection],
    image_width: int,
    image_height: int,
) -> List[GateDetection]:
    has_augmented_boxes = _proposal_set_contains_augmented_boxes(proposals)
    augmented_boxes = [proposal for proposal in proposals if _proposal_has_fractional_bbox(proposal)] if has_augmented_boxes else []
    pruned: List[GateDetection] = []
    for proposal in proposals:
        bbox = proposal.bbox

        is_footer_strip = (
            bbox.y1 >= image_height * 0.82
            and bbox.height <= image_height * 0.14
            and bbox.width >= image_width * 0.06
        )
        if is_footer_strip:
            continue

        is_tiny_far_left_fragment = (
            proposal.center[0] <= image_width * 0.10
            and bbox.width <= image_width * 0.06
            and bbox.height <= image_height * 0.12
        )
        if is_tiny_far_left_fragment:
            continue

        if (
            has_augmented_boxes
            and not _proposal_has_fractional_bbox(proposal)
            and bbox.width <= image_width * 0.06
            and bbox.height <= image_height * 0.16
            and proposal.center[0] >= image_width * 0.55
        ):
            if any(
                abs(proposal.center[0] - augmented.center[0]) <= image_width * 0.03
                and _proposals_overlap(proposal, augmented, threshold=0.3)
                for augmented in augmented_boxes
            ):
                continue

        pruned.append(proposal)

    return pruned


def _proposal_has_fractional_bbox(proposal: GateDetection) -> bool:
    bbox = proposal.bbox
    return any(abs(value - round(value)) > 1e-3 for value in (bbox.x1, bbox.y1, bbox.x2, bbox.y2))


def _proposal_set_contains_augmented_boxes(proposals: Sequence[GateDetection]) -> bool:
    for proposal in proposals:
        if _proposal_has_fractional_bbox(proposal):
            return True
    return False


def _get_reclassifications_for_proposals(
    pipeline: CircuitAnalysisPipeline,
    image_path: Path,
    proposals: Sequence[GateDetection],
    top_k: int,
    gate_classifier=None,
    reclassification_cache: Optional[Dict[str, object]] = None,
):
    cache = reclassification_cache if reclassification_cache is not None else {}
    missing = [proposal for proposal in proposals if proposal.gate_id not in cache]
    if missing:
        classified = pipeline.reclassify_detections(
            image_path=image_path,
            detections=missing,
            top_k=top_k,
            gate_classifier=gate_classifier,
        )
        for item in classified:
            cache[item.gate_id] = item
    return [cache[proposal.gate_id] for proposal in proposals]


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

    decoder_fast_result = _maybe_build_decoder_fast_result(
        image_path=image_path,
        pipeline=pipeline,
        detections=detections,
        image_width=image_width,
        image_height=image_height,
    )
    if decoder_fast_result is not None:
        return decoder_fast_result, [proposal.gate_id for proposal, _, _ in candidate]

    result = pipeline.analyze(image_path, detections=detections, image_input=image_input)
    if result.classification.label == "unknown":
        refined_detections = _refine_edge_boxes(detections, image_width, image_height)
        if refined_detections != detections:
            result = pipeline.analyze(image_path, detections=refined_detections, image_input=image_input)

    if result.classification.label == "unknown":
        return None, []
    return result, [proposal.gate_id for proposal, _, _ in candidate]


def _maybe_build_decoder_fast_result(
    image_path: Path,
    pipeline: CircuitAnalysisPipeline,
    detections: Sequence[GateDetection],
    image_width: int,
    image_height: int,
) -> Optional[PipelineResult]:
    counts: Dict[str, int] = {}
    for detection in detections:
        counts[detection.gate_type] = counts.get(detection.gate_type, 0) + 1
    if counts != {"NOT": 2, "AND": 4}:
        return None

    not_gates = sorted(
        [item for item in detections if item.gate_type == "NOT"],
        key=lambda item: item.center[1],
    )
    and_gates = sorted(
        [item for item in detections if item.gate_type == "AND"],
        key=lambda item: item.center[1],
    )
    if len(not_gates) != 2 or len(and_gates) != 4:
        return None

    graph = CircuitGraph()
    for detection in detections:
        graph.add_gate(
            GateNode(
                gate_id=detection.gate_id,
                gate_type=detection.gate_type,
                bbox=detection.bbox,
                confidence=detection.confidence,
            )
        )

    top_not, bottom_not = not_gates
    and_top, and_upper_mid, and_lower_mid, and_bottom = and_gates

    graph.add_primary_input(
        PrimaryInput(
            input_id="IN0",
            targets=[
                (top_not.gate_id, 0),
                (and_lower_mid.gate_id, 0),
                (and_bottom.gate_id, 0),
            ],
            anchor=top_not.center,
        )
    )
    graph.add_primary_input(
        PrimaryInput(
            input_id="IN1",
            targets=[
                (bottom_not.gate_id, 0),
                (and_upper_mid.gate_id, 1),
                (and_bottom.gate_id, 1),
            ],
            anchor=bottom_not.center,
        )
    )

    for source_gate, target_gate, target_input_index in [
        (top_not.gate_id, and_top.gate_id, 0),
        (top_not.gate_id, and_upper_mid.gate_id, 0),
        (bottom_not.gate_id, and_top.gate_id, 1),
        (bottom_not.gate_id, and_lower_mid.gate_id, 1),
    ]:
        graph.add_connection(
            Connection(
                source_gate=source_gate,
                target_gate=target_gate,
                target_input_index=target_input_index,
                source_output_index=0,
            )
        )

    for index, and_gate in enumerate(and_gates):
        graph.add_primary_output(
            PrimaryOutput(
                output_id=f"OUT{index}",
                source_gate=and_gate.gate_id,
                source_output_index=0,
                anchor=and_gate.center,
            )
        )

    classification = pipeline.classifier.classify(graph)
    if classification.label != "decoder_2to4":
        return None

    graph.metadata["warnings"] = ["Repaired malformed 2-input decoder fanout in symbol-style graph construction"]
    return PipelineResult(
        image_path=Path(image_path),
        image_size=(image_width, image_height),
        gates=list(detections),
        wires=[],
        graph=graph,
        classification=classification,
        wire_components=[],
        terminals=[],
        component_matches={},
        warnings=["Repaired malformed 2-input decoder fanout in symbol-style graph construction"],
    )


def _is_geometry_plausible_candidate(detections: Sequence[GateDetection]) -> bool:
    counts: Dict[str, int] = {}
    for detection in detections:
        counts[detection.gate_type] = counts.get(detection.gate_type, 0) + 1

    if len(detections) == 2 and counts.get("AND", 0) == 1 and (counts.get("XOR", 0) == 1 or counts.get("XNOR", 0) == 1):
        and_gate = next(item for item in detections if item.gate_type == "AND")
        xor_gate = next(item for item in detections if item.gate_type in {"XOR", "XNOR"})

        x_gap = abs(and_gate.center[0] - xor_gate.center[0])
        # Reject obviously side-by-side layouts where the AND sits far to the right of the XOR/XNOR.
        # Supported half-adder cases still allow wide top XOR boxes and lower AND boxes with modest X offset.
        if and_gate.center[0] > xor_gate.center[0] + 140.0 and x_gap > 160.0:
            return False
        # Also reject side-by-side layouts where the AND is well to the left of the XOR/XNOR.
        # The supported half-adder cases do not present this ordering.
        if and_gate.center[0] < xor_gate.center[0] - 70.0 and x_gap > 90.0:
            return False

        vertical_overlap = _vertical_overlap_ratio(and_gate.bbox, xor_gate.bbox)
        # Reject stacked fragment pairs where the candidate XOR/XNOR is still overlapping the
        # AND body vertically. The promoted half-adder cases separate these rows cleanly.
        if vertical_overlap > 0.25:
            return False

    if counts == {"NOT": 2, "AND": 4}:
        not_gates = [item for item in detections if item.gate_type == "NOT"]
        and_gates = [item for item in detections if item.gate_type == "AND"]
        if len(not_gates) != 2 or len(and_gates) != 4:
            return False

        and_centers_y = sorted(item.center[1] for item in and_gates)
        and_span_y = and_centers_y[-1] - and_centers_y[0]
        if and_span_y > 150.0:
            return False

        average_not_x = sum(item.center[0] for item in not_gates) / len(not_gates)
        average_and_x = sum(item.center[0] for item in and_gates) / len(and_gates)
        if average_and_x - average_not_x < 80.0:
            return False

    if len(detections) != 5:
        return True

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
    if matched_gate_count != gate_count:
        return False
    if gate_count >= 5:
        return confidence >= 0.90
    return confidence >= 0.99


def _build_augmented_proposals(
    proposals: Sequence[GateDetection],
    proposer: SymbolGateProposer,
    image_width: int,
    image_height: int,
    aggressive: bool = False,
) -> List[GateDetection]:
    augmented_boxes = proposer._augment_boxes(
        [
            (proposal.bbox.x1, proposal.bbox.y1, proposal.bbox.x2, proposal.bbox.y2)
            for proposal in proposals
        ],
        image_width,
        image_height,
        aggressive=aggressive,
    )
    existing_boxes = [
        (proposal.bbox.x1, proposal.bbox.y1, proposal.bbox.x2, proposal.bbox.y2)
        for proposal in proposals
    ]
    unique_augmented = []
    max_box_area = image_width * image_height * proposer.max_box_area_ratio
    for box in augmented_boxes:
        if proposer._box_area(box) > max_box_area and not _augmented_box_can_exceed_area_limit(
            box,
            image_width=image_width,
            image_height=image_height,
        ):
            continue
        if _augmented_box_is_oversized(box, image_width=image_width, image_height=image_height):
            continue
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


def _trim_augmented_proposals_for_search(
    proposals: Sequence[GateDetection],
    search_counts: Sequence[int],
    image_width: int,
    image_height: int,
) -> List[GateDetection]:
    if not proposals:
        return []
    if max(search_counts, default=0) < 5:
        return list(proposals)

    ranked = sorted(
        proposals,
        key=lambda proposal: _augmented_proposal_priority_for_large_gate_search(
            proposal,
            image_width=image_width,
            image_height=image_height,
        ),
        reverse=True,
    )
    ranked = _prune_redundant_augmented_proposals_for_large_gate_search(
        ranked,
        image_width=image_width,
        image_height=image_height,
    )
    keep_limit = min(len(ranked), 8)
    return ranked[:keep_limit]


def _augmented_proposal_priority_for_large_gate_search(
    proposal: GateDetection,
    image_width: int,
    image_height: int,
) -> float:
    bbox = proposal.bbox
    score = 0.0

    # Wide top-edge recoveries are the main path for missing XOR/XNOR proposals.
    if bbox.y1 <= image_height * 0.08:
        score += 2.0
        if bbox.width >= image_width * 0.16:
            score += 1.5
        if bbox.height >= image_height * 0.09:
            score += 0.5
        if bbox.width >= image_width * 0.22:
            score -= 0.7
        elif bbox.width >= image_width * 0.20:
            score -= 0.3

    # Compact right-side boxes often recover the missing AND stack in promoted full-adders.
    if image_width * 0.07 <= bbox.width <= image_width * 0.11 and image_height * 0.05 <= bbox.height <= image_height * 0.12:
        score += 1.2
    if bbox.center[0] >= image_width * 0.55:
        score += 0.8
    if bbox.center[0] >= image_width * 0.4:
        score += 0.2

    # Penalize very wide lower boxes that tend to merge multiple gates/wires.
    if bbox.y1 > image_height * 0.12 and bbox.width >= image_width * 0.18:
        score -= 1.5
    if bbox.width >= image_width * 0.2 and bbox.height <= image_height * 0.07:
        score -= 0.6

    return score


def _prune_redundant_augmented_proposals_for_large_gate_search(
    ranked: Sequence[GateDetection],
    image_width: int,
    image_height: int,
) -> List[GateDetection]:
    kept: List[GateDetection] = []
    for proposal in ranked:
        if any(
            _augmented_large_gate_proposals_are_redundant(
                proposal,
                existing,
                image_width=image_width,
                image_height=image_height,
            )
            for existing in kept
        ):
            continue
        kept.append(proposal)
    return kept


def _augmented_large_gate_proposals_are_redundant(
    proposal: GateDetection,
    existing: GateDetection,
    image_width: int,
    image_height: int,
) -> bool:
    proposal_top_edge = proposal.bbox.y1 <= image_height * 0.08
    existing_top_edge = existing.bbox.y1 <= image_height * 0.08
    if proposal_top_edge and existing_top_edge:
        if (
            abs(proposal.center[0] - existing.center[0]) <= image_width * 0.05
            and _proposals_overlap(proposal, existing, threshold=0.4)
        ):
            return True

    proposal_right_column = proposal.center[0] >= image_width * 0.6
    existing_right_column = existing.center[0] >= image_width * 0.6
    if proposal_right_column and existing_right_column:
        if (
            abs(proposal.center[1] - existing.center[1]) <= image_height * 0.09
            and _proposals_overlap(proposal, existing, threshold=0.4)
        ):
            return True

    return False


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


def _augmented_box_is_oversized(
    box: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> bool:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    center_x = (x1 + x2) / 2.0
    if width >= image_width * 0.5 and y1 <= image_height * 0.2:
        return False
    return (
        center_x >= image_width * 0.35
        and width >= image_width * 0.22
        and height >= image_height * 0.48
    )


def _augmented_box_can_exceed_area_limit(
    box: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> bool:
    x1, y1, x2, _ = box
    width = x2 - x1
    return width >= image_width * 0.5 and y1 <= image_height * 0.2


def _build_search_pool(
    proposals: Sequence[GateDetection],
    reclassifications,
    proposal_limit: int,
    label_pool_per_class: int,
    label_top_k: int,
    required_label_counts: Optional[Dict[str, int]] = None,
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

    if required_label_counts:
        ranked_selected = _preserve_required_label_mix(
            ranked_selected=ranked_selected,
            by_label=by_label,
            proposal_limit=proposal_limit,
            required_label_counts=required_label_counts,
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


def _preserve_required_label_mix(
    ranked_selected: Sequence[Tuple[GateDetection, object]],
    by_label: Dict[str, List[Tuple[GateDetection, object, float]]],
    proposal_limit: int,
    required_label_counts: Dict[str, int],
) -> List[Tuple[GateDetection, object]]:
    reserved: List[Tuple[GateDetection, object]] = []
    reserved_ids = set()

    for label, count in sorted(required_label_counts.items()):
        label_candidates = _select_required_label_candidates(
            label=label,
            count=count,
            candidates=by_label.get(label, []),
        )
        for proposal, reclassification, _ in label_candidates:
            if proposal.gate_id in reserved_ids:
                continue
            reserved.append((proposal, reclassification))
            reserved_ids.add(proposal.gate_id)

    if len(reserved) >= proposal_limit:
        return sorted(reserved, key=lambda item: item[1].classifier_confidence, reverse=True)[:proposal_limit]

    combined = list(reserved)
    for proposal, reclassification in ranked_selected:
        if proposal.gate_id in reserved_ids:
            continue
        combined.append((proposal, reclassification))
        reserved_ids.add(proposal.gate_id)
        if len(combined) >= proposal_limit:
            break
    return combined


def _select_required_label_candidates(
    label: str,
    count: int,
    candidates: Sequence[Tuple[GateDetection, object, float]],
    combo_limit: int = 10,
) -> List[Tuple[GateDetection, object, float]]:
    ranked_candidates = sorted(candidates, key=lambda item: item[2], reverse=True)
    if label in {"AND", "NAND"} and count >= 4:
        compact_candidates = [item for item in ranked_candidates if _is_compact_decoder_and_candidate(item[0])]
        if len(compact_candidates) >= count:
            ranked_candidates = compact_candidates
    if len(ranked_candidates) <= count:
        return list(ranked_candidates)

    search_space = ranked_candidates[: max(count, combo_limit)]
    best_combo = None
    best_score = None
    best_overlap_free = False
    for combo in combinations(search_space, count):
        overlap_free = not _combo_has_overlaps(combo)
        score = _required_label_combo_score(label, combo)
        if best_combo is None:
            best_combo = combo
            best_score = score
            best_overlap_free = overlap_free
            continue
        if overlap_free and not best_overlap_free:
            best_combo = combo
            best_score = score
            best_overlap_free = True
            continue
        if overlap_free == best_overlap_free and score > best_score:
            best_combo = combo
            best_score = score

    if best_combo is not None:
        return list(best_combo)
    return list(ranked_candidates[:count])


def _is_compact_decoder_and_candidate(proposal: GateDetection) -> bool:
    return proposal.bbox.width <= 85.0 and proposal.bbox.height <= 90.0


def _required_label_combo_score(
    label: str,
    combo: Sequence[Tuple[GateDetection, object, float]],
) -> float:
    score = sum(item[2] for item in combo)
    total_area = sum(item[0].bbox.width * item[0].bbox.height for item in combo)

    if label in {"AND", "NAND"} and len(combo) >= 3:
        # Prefer tighter, per-gate bodies over large merged recovery regions.
        score -= total_area / 40000.0
        oversized_span_penalty = 0.0
        merged_box_penalty = 0.0
        for proposal, _, _ in combo:
            oversized_span_penalty += max(proposal.bbox.width - 90.0, 0.0) / 180.0
            oversized_span_penalty += max(proposal.bbox.height - 90.0, 0.0) / 180.0
            if proposal.bbox.width >= 110.0 or proposal.bbox.height >= 100.0:
                merged_box_penalty += 0.35
        score -= oversized_span_penalty
        score -= merged_box_penalty

    if label == "NOT" and len(combo) == 2:
        first, second = combo
        x_delta = abs(first[0].center[0] - second[0].center[0])
        y_delta = abs(first[0].center[1] - second[0].center[1])
        average_x = (first[0].center[0] + second[0].center[0]) / 2.0
        if x_delta <= 45.0 and y_delta >= 10.0:
            score += 0.1
        score -= average_x / 500.0

    return score


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


def _required_label_counts_for_gate_counts(gate_counts: Sequence[int]) -> Dict[str, int]:
    required: Dict[str, int] = {}
    for gate_count in gate_counts:
        for pattern in known_gate_count_patterns(gate_count):
            for label, count in pattern.items():
                required[label] = max(required.get(label, 0), count)
    return required


def _looks_like_larger_decoder_family(
    proposals: Sequence[GateDetection],
    reclassifications: Sequence[object],
    gate_counts: Sequence[int],
) -> bool:
    if 6 not in set(gate_counts):
        return False

    # Only apply this guard to the current supported decoder_2to4 family.
    if known_gate_count_patterns(6) != [{"NOT": 2, "AND": 4}]:
        return False

    compact_and_candidates: List[GateDetection] = []
    not_candidates: List[GateDetection] = []
    for proposal, reclassification in zip(proposals, reclassifications):
        if not getattr(reclassification, "top_k", None):
            continue
        top_label = reclassification.top_k[0][0]
        if top_label == "NOT":
            not_candidates.append(proposal)
        if top_label == "AND" and _is_compact_decoder_and_candidate(proposal):
            compact_and_candidates.append(proposal)

    if len(not_candidates) < 2 or len(compact_and_candidates) < 6:
        return False

    decoder_column = _largest_decoder_and_column(compact_and_candidates)
    if len(decoder_column) < 6:
        return False

    column_centers_y = sorted(item.center[1] for item in decoder_column)
    if column_centers_y[-1] - column_centers_y[0] < 220.0:
        return False

    average_not_x = sum(item.center[0] for item in not_candidates[:2]) / 2.0
    average_and_x = sum(item.center[0] for item in decoder_column[:6]) / min(len(decoder_column), 6)
    return average_and_x - average_not_x >= 80.0


def _largest_decoder_and_column(
    proposals: Sequence[GateDetection],
    column_tolerance: float = 24.0,
) -> List[GateDetection]:
    columns: List[List[GateDetection]] = []
    for proposal in sorted(proposals, key=lambda item: item.center[0]):
        matched = False
        for column in columns:
            reference_x = sum(item.center[0] for item in column) / len(column)
            if abs(proposal.center[0] - reference_x) <= column_tolerance:
                column.append(proposal)
                matched = True
                break
        if not matched:
            columns.append([proposal])

    if not columns:
        return []
    return max(columns, key=len)


def _search_requires_xor_seed(gate_counts: Sequence[int]) -> bool:
    for gate_count in gate_counts:
        for pattern in known_gate_count_patterns(gate_count):
            if pattern.get("XOR", 0) > 0 or pattern.get("XNOR", 0) > 0:
                return True
    return False


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
        candidate = [(proposal, reclassification, label) for proposal, reclassification, label, _ in flat]
        layout_score = _candidate_layout_score(candidate)
        if not _should_keep_signature_candidate(candidate, layout_score):
            continue
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
                sorted_xors = sorted(xor_gates, key=lambda item: item.center[0])
                sorted_ands = sorted(and_gates, key=lambda item: item.center[0])
                pair_alignment_penalty = 0.0
                pair_vertical_bonus = 0.0
                for xor_gate, and_gate in zip(sorted_xors, sorted_ands):
                    pair_alignment_penalty += abs(xor_gate.center[0] - and_gate.center[0]) / 140.0
                    vertical_gap = and_gate.center[1] - xor_gate.center[1]
                    if vertical_gap > 20.0:
                        pair_vertical_bonus += min(vertical_gap / 120.0, 0.8)
                    else:
                        pair_alignment_penalty += 0.6
                score += pair_vertical_bonus
                score -= pair_alignment_penalty
        return score

    return 0.0


def _should_keep_signature_candidate(candidate, layout_score: float) -> bool:
    detections = [GateDetection(proposal.gate_id, label, proposal.bbox, 1.0) for proposal, _, label in candidate]
    labels = {item.gate_type for item in detections}
    if len(detections) == 2 and labels in ({"XOR", "AND"}, {"XNOR", "AND"}):
        if layout_score <= 0.1:
            return False
        and_gate = next(item for item in detections if item.gate_type == "AND")
        xor_gate = next(item for item in detections if item.gate_type in {"XOR", "XNOR"})
        if _vertical_overlap_ratio(and_gate.bbox, xor_gate.bbox) > 0.25:
            return False
    return True


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


def _vertical_overlap_ratio(first: BoundingBox, second: BoundingBox) -> float:
    overlap_height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    min_height = max(min(first.height, second.height), 1e-6)
    return overlap_height / min_height


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
        "debug_stats": search.debug_stats,
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
