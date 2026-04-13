"""
Circuit-level classification from graph structure and truth tables.
"""

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from .circuit_graph import CircuitGraph
from .types import ClassificationResult


@dataclass(frozen=True)
class CircuitSignature:
    label: str
    gate_counts: Dict[str, int]
    input_count: int
    output_columns: List[str]


KNOWN_SIGNATURES = [
    CircuitSignature(
        label="half_adder",
        gate_counts={"XOR": 1, "AND": 1},
        input_count=2,
        output_columns=["0110", "0001"],
    ),
    CircuitSignature(
        label="half_subtractor",
        gate_counts={"XOR": 1, "NOT": 1, "AND": 1},
        input_count=2,
        output_columns=["0110", "0100"],
    ),
    CircuitSignature(
        label="half_subtractor",
        gate_counts={"XOR": 1, "NOT": 1, "AND": 1},
        input_count=2,
        output_columns=["0110", "0010"],
    ),
    CircuitSignature(
        label="full_adder",
        gate_counts={"XOR": 2, "AND": 2, "OR": 1},
        input_count=3,
        output_columns=["01101001", "00010111"],
    ),
    CircuitSignature(
        label="full_adder",
        gate_counts={"XOR": 1, "AND": 3, "OR": 1},
        input_count=3,
        output_columns=["01101001", "00010111"],
    ),
    CircuitSignature(
        label="decoder_2to4",
        gate_counts={"NOT": 2, "AND": 4},
        input_count=2,
        output_columns=["0001", "0010", "0100", "1000"],
    ),
]


def known_gate_count_patterns(gate_count: int) -> List[Dict[str, int]]:
    patterns: List[Dict[str, int]] = []
    seen = set()
    for signature in KNOWN_SIGNATURES:
        if sum(signature.gate_counts.values()) != gate_count:
            continue
        key = tuple(sorted(signature.gate_counts.items()))
        if key in seen:
            continue
        seen.add(key)
        patterns.append(dict(signature.gate_counts))
    return patterns


class CircuitClassifier:
    """Classify simple known circuits using truth-table and structural signatures."""

    def classify(self, graph: CircuitGraph) -> ClassificationResult:
        if not graph.primary_inputs or not graph.primary_outputs:
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                reasoning="Graph is missing primary inputs or outputs",
                truth_table=[],
                expressions={},
            )

        try:
            truth_table = graph.get_truth_table()
            expressions = graph.describe_outputs()
        except Exception as exc:
            return ClassificationResult(
                label="unknown",
                confidence=0.0,
                reasoning=f"Graph could not be evaluated: {exc}",
                truth_table=[],
                expressions={},
            )

        actual_counts = Counter(gate.gate_type.upper() for gate in graph.gates.values())
        output_columns = self._extract_output_columns(graph, truth_table)

        best_label = "unknown"
        best_confidence = 0.15
        reasoning = "No known topology matched"

        for signature in KNOWN_SIGNATURES:
            score = self._score_signature(graph, actual_counts, output_columns, signature)
            if score > best_confidence:
                best_label = signature.label
                best_confidence = score
                reasoning = (
                    f"Matched {signature.label} using input count {signature.input_count}, "
                    f"gate counts {dict(signature.gate_counts)}, and output truth columns {signature.output_columns}"
                )

        alias_match = self._maybe_match_with_split_input_alias(graph, actual_counts, truth_table)
        if alias_match and alias_match[1] > best_confidence:
            best_label, best_confidence, reasoning = alias_match

        # Try multi-input aliasing for circuits with many split inputs
        multi_alias_match = self._maybe_match_with_multi_input_alias(graph, actual_counts, truth_table)
        if multi_alias_match and multi_alias_match[1] > best_confidence:
            best_label, best_confidence, reasoning = multi_alias_match

        return ClassificationResult(
            label=best_label,
            confidence=min(best_confidence, 0.99),
            reasoning=reasoning,
            truth_table=truth_table,
            expressions=expressions,
        )

    def _extract_output_columns(self, graph: CircuitGraph, truth_table: List[Dict[str, int]]) -> List[str]:
        output_names = [item.output_id for item in graph.primary_outputs]
        columns = []
        for name in output_names:
            columns.append("".join(str(row[name]) for row in truth_table))
        return sorted(columns)

    def _score_signature(
        self,
        graph: CircuitGraph,
        actual_counts: Counter,
        output_columns: List[str],
        signature: CircuitSignature,
    ) -> float:
        if len(graph.primary_inputs) != signature.input_count:
            return 0.05

        # Require output count to match expected for multi-gate circuits
        expected_output_count = len(signature.output_columns)
        actual_output_count = len(output_columns)
        gate_count = sum(signature.gate_counts.values())
        if gate_count >= 3 and actual_output_count != expected_output_count:
            return 0.08

        expected_counts = Counter(signature.gate_counts)
        shared = sum(min(actual_counts[gate], expected_counts[gate]) for gate in expected_counts)
        count_score = shared / max(sum(expected_counts.values()), 1)

        expected_columns = sorted(signature.output_columns)
        matching_columns = sum(min(output_columns.count(column), expected_columns.count(column)) for column in set(expected_columns))
        truth_score = matching_columns / max(len(expected_columns), 1)

        # Require at least one matching column for multi-gate circuits
        if gate_count >= 3 and matching_columns == 0:
            return 0.10

        return 0.35 * count_score + 0.65 * truth_score

    def _maybe_match_with_split_input_alias(
        self,
        graph: CircuitGraph,
        actual_counts: Counter,
        truth_table: List[Dict[str, int]],
    ) -> Optional[Tuple[str, float, str]]:
        input_names = [item.input_id for item in graph.primary_inputs]
        if len(graph.primary_outputs) != 2:
            return None

        best_match: Optional[Tuple[str, float, str]] = None
        output_names = [item.output_id for item in graph.primary_outputs]
        alias_targets = [
            (
                "half_adder",
                Counter({"XOR": 1, "AND": 1}),
                3,
                sorted(KNOWN_SIGNATURES[0].output_columns),
            ),
            (
                "full_adder",
                Counter({"XOR": 2, "AND": 2, "OR": 1}),
                4,
                sorted(KNOWN_SIGNATURES[3].output_columns),
            ),
        ]

        for label, expected_counts, expected_input_count, expected_columns in alias_targets:
            if len(input_names) != expected_input_count:
                continue
            if actual_counts != expected_counts:
                continue

            for alias_a, alias_b in combinations(input_names, 2):
                reduced_rows = [row for row in truth_table if row[alias_a] == row[alias_b]]
                if len(reduced_rows) != 2 ** (expected_input_count - 1):
                    continue

                retained_inputs = [name for name in input_names if name != alias_b]
                projected_rows = self._project_rows(reduced_rows, retained_inputs, output_names)
                projected_columns = self._extract_output_columns_from_rows(output_names, projected_rows)

                matching_columns = sum(
                    min(projected_columns.count(column), expected_columns.count(column))
                    for column in set(expected_columns)
                )
                truth_score = matching_columns / max(len(expected_columns), 1)
                if truth_score < 1.0:
                    continue

                score = 0.90
                reasoning = (
                    f"Matched {label} after aliasing split inputs {alias_a} and {alias_b}; "
                    "this compensates for a symbol-style wire split during topology extraction"
                )
                if best_match is None or score > best_match[1]:
                    best_match = (label, score, reasoning)

        return best_match

    @staticmethod
    def _project_rows(
        rows: List[Dict[str, int]],
        input_names: List[str],
        output_names: List[str],
    ) -> List[Dict[str, int]]:
        projected: List[Dict[str, int]] = []
        seen = set()
        for row in rows:
            key = tuple(row[name] for name in input_names)
            if key in seen:
                continue
            seen.add(key)
            projected.append({name: row[name] for name in input_names + output_names})
        projected.sort(key=lambda item: tuple(item[name] for name in input_names))
        return projected

    @staticmethod
    def _extract_output_columns_from_rows(output_names: List[str], rows: List[Dict[str, int]]) -> List[str]:
        return sorted("".join(str(row[name]) for row in rows) for name in output_names)

    def _maybe_match_with_multi_input_alias(
        self,
        graph: CircuitGraph,
        actual_counts: Counter,
        truth_table: List[Dict[str, int]],
    ) -> Optional[Tuple[str, float, str]]:
        """
        Try to match a circuit with many split inputs by finding input groupings.

        When wire detection creates many primary inputs that should actually be
        fewer signals (due to fan-out), this method tries to find a valid grouping
        that matches a known signature.
        """
        input_names = [item.input_id for item in graph.primary_inputs]
        output_names = [item.output_id for item in graph.primary_outputs]

        # Only try this for circuits with many inputs and expected output count
        if len(input_names) < 5 or len(output_names) not in (2, 3):
            return None

        # Check for full_adder pattern (1 XOR + 3 AND + 1 OR or 2 XOR + 2 AND + 1 OR)
        full_adder_patterns = [
            Counter({"XOR": 1, "AND": 3, "OR": 1}),
            Counter({"XOR": 2, "AND": 2, "OR": 1}),
        ]
        if actual_counts not in full_adder_patterns:
            return None

        expected_columns = sorted(KNOWN_SIGNATURES[3].output_columns)  # full_adder columns
        target_input_count = 3

        # Try to find a grouping of inputs into target_input_count groups
        result = self._try_all_input_partitions(
            input_names, target_input_count, truth_table, output_names, expected_columns
        )

        if result is None:
            return None

        assignment, _ = result
        aliased_groups = self._describe_alias_groups(assignment)

        score = 0.88 if len(output_names) == 2 else 0.85
        reasoning = (
            f"Matched full_adder after grouping {len(input_names)} split inputs "
            f"into {target_input_count} signals: {aliased_groups}; "
            "this compensates for fan-out wire splits during topology extraction"
        )

        return ("full_adder", score, reasoning)

    def _try_all_input_partitions(
        self,
        input_names: List[str],
        target_groups: int,
        truth_table: List[Dict[str, int]],
        output_names: List[str],
        expected_columns: List[str],
    ) -> Optional[Tuple[Dict[str, str], List[Dict[str, int]]]]:
        """
        Try all ways to partition inputs into groups and find one that works.

        For efficiency, use a greedy approach: try each combination of representative
        inputs, then for each remaining input, assign it to the group that produces
        the most consistent rows when filtered.
        """
        from itertools import product

        # Try each combination of inputs as group representatives
        for representatives in combinations(input_names, target_groups):
            remaining = [inp for inp in input_names if inp not in representatives]

            # Try all possible assignments of remaining inputs to representatives
            # This is expensive but necessary for correctness
            if len(remaining) > 6:
                # Too many possibilities, skip
                continue

            for assignment_tuple in product(representatives, repeat=len(remaining)):
                assignment = {rep: rep for rep in representatives}
                for inp, rep in zip(remaining, assignment_tuple):
                    assignment[inp] = rep

                # Filter rows where grouped inputs are consistent
                filtered_rows = self._filter_rows_by_assignment(truth_table, assignment)

                if len(filtered_rows) != 2 ** target_groups:
                    continue

                # Project to representative inputs
                projected_rows = self._project_rows(
                    filtered_rows, list(representatives), output_names
                )

                # Check if output columns match (for 2 outputs)
                if len(output_names) == 2:
                    projected_columns = self._extract_output_columns_from_rows(
                        output_names, projected_rows
                    )
                    matching = sum(
                        min(projected_columns.count(col), expected_columns.count(col))
                        for col in set(expected_columns)
                    )
                    if matching == len(expected_columns):
                        return assignment, projected_rows

                # For 3 outputs, check if any pair of outputs matches
                elif len(output_names) >= 2:
                    for o1, o2 in combinations(output_names, 2):
                        two_output_columns = sorted([
                            "".join(str(row[o1]) for row in projected_rows),
                            "".join(str(row[o2]) for row in projected_rows),
                        ])
                        matching = sum(
                            min(two_output_columns.count(col), expected_columns.count(col))
                            for col in set(expected_columns)
                        )
                        if matching == len(expected_columns):
                            return assignment, projected_rows

        return None

    @staticmethod
    def _filter_rows_by_assignment(
        truth_table: List[Dict[str, int]],
        assignment: Dict[str, str],
    ) -> List[Dict[str, int]]:
        """Filter rows where all inputs in each group have the same value."""
        filtered = []
        for row in truth_table:
            # Check if all inputs assigned to the same representative have equal values
            valid = True
            for inp, rep in assignment.items():
                if inp != rep and row[inp] != row[rep]:
                    valid = False
                    break
            if valid:
                filtered.append(row)
        return filtered

    @staticmethod
    def _describe_alias_groups(assignment: Dict[str, str]) -> str:
        """Create a human-readable description of the alias groups."""
        groups: Dict[str, List[str]] = {}
        for inp, rep in assignment.items():
            groups.setdefault(rep, []).append(inp)
        descriptions = []
        for rep, members in sorted(groups.items()):
            if len(members) > 1:
                descriptions.append(f"{{{', '.join(sorted(members))}}}")
            else:
                descriptions.append(members[0])
        return ", ".join(descriptions)
