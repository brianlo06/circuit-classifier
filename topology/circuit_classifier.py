"""
Circuit-level classification from graph structure and truth tables.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

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
]


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

        expected_counts = Counter(signature.gate_counts)
        shared = sum(min(actual_counts[gate], expected_counts[gate]) for gate in expected_counts)
        count_score = shared / max(sum(expected_counts.values()), 1)

        expected_columns = sorted(signature.output_columns)
        matching_columns = sum(min(output_columns.count(column), expected_columns.count(column)) for column in set(expected_columns))
        truth_score = matching_columns / max(len(expected_columns), 1)

        return 0.35 * count_score + 0.65 * truth_score
