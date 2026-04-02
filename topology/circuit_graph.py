"""
Circuit graph structures and boolean evaluation.
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional

from .types import Connection, GateNode, PrimaryInput, PrimaryOutput


GATE_INPUT_COUNTS = {
    "NOT": 1,
    "AND": 2,
    "NAND": 2,
    "NOR": 2,
    "OR": 2,
    "XNOR": 2,
    "XOR": 2,
}


@dataclass
class CircuitGraph:
    """Directed graph representation of a logic circuit."""

    gates: Dict[str, GateNode] = field(default_factory=dict)
    connections: List[Connection] = field(default_factory=list)
    primary_inputs: List[PrimaryInput] = field(default_factory=list)
    primary_outputs: List[PrimaryOutput] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def add_gate(self, gate: GateNode) -> None:
        self.gates[gate.gate_id] = gate

    def add_connection(self, connection: Connection) -> None:
        self.connections.append(connection)

    def add_primary_input(self, primary_input: PrimaryInput) -> None:
        self.primary_inputs.append(primary_input)

    def add_primary_output(self, primary_output: PrimaryOutput) -> None:
        self.primary_outputs.append(primary_output)

    def get_topological_order(self) -> List[str]:
        indegree = {gate_id: 0 for gate_id in self.gates}
        adjacency = {gate_id: [] for gate_id in self.gates}

        for connection in self.connections:
            indegree[connection.target_gate] += 1
            adjacency[connection.source_gate].append(connection.target_gate)

        queue = sorted([gate_id for gate_id, degree in indegree.items() if degree == 0])
        order: List[str] = []

        while queue:
            gate_id = queue.pop(0)
            order.append(gate_id)
            for neighbor in adjacency[gate_id]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
                    queue.sort()

        if len(order) != len(self.gates):
            raise ValueError("Graph contains a cycle or unresolved connection ambiguity")

        return order

    def evaluate(self, input_values: Dict[str, bool]) -> Dict[str, bool]:
        gate_outputs: Dict[str, bool] = {}
        input_map: Dict[tuple, bool] = {}

        for primary_input in self.primary_inputs:
            if primary_input.input_id not in input_values:
                raise KeyError(f"Missing value for primary input '{primary_input.input_id}'")
            value = bool(input_values[primary_input.input_id])
            for target_gate, target_input_index in primary_input.targets:
                input_map[(target_gate, target_input_index)] = value

        for gate_id in self.get_topological_order():
            gate = self.gates[gate_id]
            input_count = GATE_INPUT_COUNTS.get(gate.gate_type.upper(), 2)
            values: List[bool] = [False] * input_count

            for connection in self.connections:
                if connection.target_gate == gate_id:
                    values[connection.target_input_index] = gate_outputs[connection.source_gate]

            for input_index in range(input_count):
                key = (gate_id, input_index)
                if key in input_map:
                    values[input_index] = input_map[key]
                elif all(
                    not (connection.target_gate == gate_id and connection.target_input_index == input_index)
                    for connection in self.connections
                ):
                    raise ValueError(f"Gate '{gate_id}' input {input_index} is unconnected")

            gate_outputs[gate_id] = self._evaluate_gate(gate.gate_type, values)

        results: Dict[str, bool] = {}
        for primary_output in self.primary_outputs:
            results[primary_output.output_id] = gate_outputs[primary_output.source_gate]

        return results

    def get_truth_table(self) -> List[Dict[str, int]]:
        rows: List[Dict[str, int]] = []
        input_names = [item.input_id for item in self.primary_inputs]
        output_names = [item.output_id for item in self.primary_outputs]

        for values in product([False, True], repeat=len(input_names)):
            inputs = dict(zip(input_names, values))
            outputs = self.evaluate(inputs)
            row: Dict[str, int] = {name: int(bool(inputs[name])) for name in input_names}
            row.update({name: int(bool(outputs[name])) for name in output_names})
            rows.append(row)

        return rows

    def describe_outputs(self) -> Dict[str, str]:
        incoming = {gate_id: [] for gate_id in self.gates}
        for connection in self.connections:
            incoming[connection.target_gate].append(connection)

        expressions: Dict[str, str] = {}

        def expr_for_gate(gate_id: str) -> str:
            gate = self.gates[gate_id]
            gate_type = gate.gate_type.upper()
            input_count = GATE_INPUT_COUNTS.get(gate_type, 2)
            inputs: List[str] = ["?"] * input_count

            for primary_input in self.primary_inputs:
                for target_gate, target_input_index in primary_input.targets:
                    if target_gate == gate_id:
                        inputs[target_input_index] = primary_input.input_id

            for connection in incoming[gate_id]:
                inputs[connection.target_input_index] = expr_for_gate(connection.source_gate)

            if gate_type == "NOT":
                return f"NOT({inputs[0]})"
            if gate_type == "AND":
                return f"({inputs[0]} AND {inputs[1]})"
            if gate_type == "NAND":
                return f"NOT({inputs[0]} AND {inputs[1]})"
            if gate_type == "OR":
                return f"({inputs[0]} OR {inputs[1]})"
            if gate_type == "NOR":
                return f"NOT({inputs[0]} OR {inputs[1]})"
            if gate_type == "XOR":
                return f"({inputs[0]} XOR {inputs[1]})"
            if gate_type == "XNOR":
                return f"NOT({inputs[0]} XOR {inputs[1]})"
            return gate_type

        for primary_output in self.primary_outputs:
            expressions[primary_output.output_id] = expr_for_gate(primary_output.source_gate)
        return expressions

    def to_dict(self) -> Dict[str, object]:
        return {
            "gates": {
                gate_id: {
                    "gate_type": gate.gate_type,
                    "bbox": [gate.bbox.x1, gate.bbox.y1, gate.bbox.x2, gate.bbox.y2],
                    "confidence": gate.confidence,
                }
                for gate_id, gate in self.gates.items()
            },
            "connections": [connection.__dict__.copy() for connection in self.connections],
            "primary_inputs": [primary_input.__dict__.copy() for primary_input in self.primary_inputs],
            "primary_outputs": [primary_output.__dict__.copy() for primary_output in self.primary_outputs],
            "metadata": dict(self.metadata),
        }

    @staticmethod
    def _evaluate_gate(gate_type: str, values: List[bool]) -> bool:
        gate_type = gate_type.upper()
        if gate_type == "NOT":
            return not values[0]
        if gate_type == "AND":
            return values[0] and values[1]
        if gate_type == "NAND":
            return not (values[0] and values[1])
        if gate_type == "OR":
            return values[0] or values[1]
        if gate_type == "NOR":
            return not (values[0] or values[1])
        if gate_type == "XOR":
            return values[0] ^ values[1]
        if gate_type == "XNOR":
            return not (values[0] ^ values[1])
        raise ValueError(f"Unsupported gate type '{gate_type}'")

