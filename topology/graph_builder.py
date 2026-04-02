"""
Construct a circuit graph from detected gates and wires.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .circuit_graph import CircuitGraph
from .gate_terminals import GateTerminalProvider
from .types import Connection, GateDetection, GateNode, Point, PrimaryInput, PrimaryOutput, Terminal, WireComponent


@dataclass
class GraphBuildResult:
    """Graph construction result with diagnostics."""

    graph: CircuitGraph
    warnings: List[str]
    terminals: List[Terminal]
    component_matches: Dict[str, List[Terminal]]


class GraphBuilder:
    """Map wire components to gate terminals and infer a directed logic graph."""

    def __init__(self, terminal_provider: Optional[GateTerminalProvider] = None, terminal_snap_distance: float = 16.0):
        self.terminal_provider = terminal_provider or GateTerminalProvider()
        self.terminal_snap_distance = terminal_snap_distance

    def build_graph(self, gates: Sequence[GateDetection], wire_components: Sequence[WireComponent]) -> GraphBuildResult:
        graph = CircuitGraph()
        warnings: List[str] = []

        for gate in gates:
            graph.add_gate(GateNode(gate_id=gate.gate_id, gate_type=gate.gate_type, bbox=gate.bbox, confidence=gate.confidence))

        terminals = [terminal for gate in gates for terminal in self.terminal_provider.get_terminals(gate)]
        raw_matches: Dict[str, Dict[Tuple[str, str, int], Tuple[Terminal, float]]] = {}
        best_terminal_components: Dict[Tuple[str, str, int], Tuple[str, float]] = {}

        for component in wire_components:
            matches = self._match_terminals(component, terminals)
            raw_matches[component.component_id] = matches
            for terminal_key, (terminal, distance) in matches.items():
                current = best_terminal_components.get(terminal_key)
                if current is None or distance < current[1]:
                    best_terminal_components[terminal_key] = (component.component_id, distance)

        component_matches: Dict[str, List[Terminal]] = {}
        for component in wire_components:
            filtered: List[Terminal] = []
            for terminal_key, (terminal, _) in raw_matches.get(component.component_id, {}).items():
                owner = best_terminal_components.get(terminal_key)
                if owner and owner[0] == component.component_id:
                    filtered.append(terminal)
            component_matches[component.component_id] = list(sorted(filtered, key=lambda item: (item.gate_id, item.kind, item.index)))

        input_counter = 0
        output_counter = 0

        for component in wire_components:
            matched = component_matches.get(component.component_id, [])
            inputs = [terminal for terminal in matched if terminal.kind == "input"]
            outputs = [terminal for terminal in matched if terminal.kind == "output"]

            if not matched:
                continue

            if not outputs and inputs:
                grouped_inputs = self._split_primary_input_terminals(inputs)
                for group in grouped_inputs:
                    graph.add_primary_input(
                        PrimaryInput(
                            input_id=f"IN{input_counter}",
                            targets=[(terminal.gate_id, terminal.index) for terminal in sorted(group, key=lambda item: (item.gate_id, item.index))],
                            anchor=min((terminal.point for terminal in group), key=lambda point: (point[0], point[1])),
                        )
                    )
                    input_counter += 1
                continue

            if outputs and inputs:
                source = min(outputs, key=lambda terminal: terminal.point[0])
                extra_sources = [terminal for terminal in outputs if terminal != source]
                if extra_sources:
                    warnings.append(
                        f"{component.component_id}: multiple output terminals matched; using {source.gate_id} as source"
                    )
                for target in sorted(inputs, key=lambda terminal: (terminal.gate_id, terminal.index)):
                    if target.gate_id == source.gate_id:
                        warnings.append(
                            f"{component.component_id}: ignoring self-connection on gate {source.gate_id}"
                        )
                        continue
                    graph.add_connection(
                        Connection(
                            source_gate=source.gate_id,
                            target_gate=target.gate_id,
                            target_input_index=target.index,
                            source_output_index=0,
                            component_id=component.component_id,
                        )
                    )
                if extra_sources:
                    for extra in extra_sources:
                        graph.add_primary_output(
                            PrimaryOutput(
                                output_id=f"OUT{output_counter}",
                                source_gate=extra.gate_id,
                                source_output_index=extra.index,
                                anchor=extra.point,
                            )
                        )
                        output_counter += 1
            elif outputs:
                source = min(outputs, key=lambda terminal: terminal.point[0])
                graph.add_primary_output(
                    PrimaryOutput(
                        output_id=f"OUT{output_counter}",
                        source_gate=source.gate_id,
                        source_output_index=source.index,
                        anchor=max(component.points, key=lambda point: point[0]) if component.points else source.point,
                    )
                )
                output_counter += 1

        unconnected_outputs = [
            terminal
            for terminal in terminals
            if terminal.kind == "output"
            and terminal.gate_id not in {output.source_gate for output in graph.primary_outputs}
            and terminal.gate_id not in {connection.source_gate for connection in graph.connections}
        ]
        for terminal in unconnected_outputs:
            graph.add_primary_output(
                PrimaryOutput(
                    output_id=f"OUT{output_counter}",
                    source_gate=terminal.gate_id,
                    source_output_index=terminal.index,
                    anchor=terminal.point,
                )
            )
            output_counter += 1
            warnings.append(f"Gate {terminal.gate_id} output was not attached to any wire; exposed as primary output")

        graph.metadata["warnings"] = list(warnings)
        graph.metadata["terminal_snap_distance"] = self.terminal_snap_distance
        return GraphBuildResult(
            graph=graph,
            warnings=warnings,
            terminals=terminals,
            component_matches=component_matches,
        )

    def _match_terminals(self, component: WireComponent, terminals: Sequence[Terminal]) -> Dict[Tuple[str, str, int], Tuple[Terminal, float]]:
        matched: Dict[Tuple[str, str, int], Tuple[Terminal, float]] = {}
        for terminal in terminals:
            distance = self._component_distance(component, terminal.point)
            if distance <= self.terminal_snap_distance:
                matched[(terminal.gate_id, terminal.kind, terminal.index)] = (terminal, distance)
        return matched

    def _component_distance(self, component: WireComponent, point: Point) -> float:
        distances = [self._distance(point, component_point) for component_point in component.points]
        distances.extend(
            self._point_to_segment_distance(point, segment.start, segment.end)
            for segment in component.segments
        )
        return min(distances) if distances else float("inf")

    @staticmethod
    def _split_primary_input_terminals(inputs: Sequence[Terminal]) -> List[List[Terminal]]:
        grouped: Dict[int, List[Terminal]] = {}
        for terminal in inputs:
            grouped.setdefault(terminal.index, []).append(terminal)
        return [grouped[index] for index in sorted(grouped)]

    @staticmethod
    def _distance(a: Point, b: Point) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return (dx * dx + dy * dy) ** 0.5

    @classmethod
    def _point_to_segment_distance(cls, point: Point, start: Point, end: Point) -> float:
        sx, sy = start
        ex, ey = end
        px, py = point
        dx = ex - sx
        dy = ey - sy

        if dx == 0 and dy == 0:
            return cls._distance(point, start)

        t = ((px - sx) * dx + (py - sy) * dy) / float(dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        projection = (sx + t * dx, sy + t * dy)
        return cls._distance(point, projection)
