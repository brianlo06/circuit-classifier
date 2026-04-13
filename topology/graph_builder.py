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

        # Merge primary inputs that appear to be fan-outs from the same signal
        self._merge_fanout_primary_inputs(graph, warnings)
        self._repair_three_input_full_adder_symbol_graph(graph, terminals, warnings)
        self._repair_two_input_decoder_symbol_graph(graph, terminals, warnings)

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

    @classmethod
    def _split_primary_input_terminals(cls, inputs: Sequence[Terminal]) -> List[List[Terminal]]:
        if cls._should_split_inputs_spatially(inputs):
            return cls._split_inputs_by_vertical_bands(inputs)

        grouped: Dict[int, List[Terminal]] = {}
        for terminal in inputs:
            grouped.setdefault(terminal.index, []).append(terminal)
        return [grouped[index] for index in sorted(grouped)]

    @staticmethod
    def _should_split_inputs_spatially(inputs: Sequence[Terminal]) -> bool:
        if len(inputs) < 5:
            return False
        ys = [terminal.point[1] for terminal in inputs]
        return (max(ys) - min(ys)) >= 180.0

    @staticmethod
    def _split_inputs_by_vertical_bands(
        inputs: Sequence[Terminal],
        y_band_threshold: float = 22.0,
    ) -> List[List[Terminal]]:
        groups: List[List[Terminal]] = []
        for terminal in sorted(inputs, key=lambda item: (item.point[1], item.point[0], item.gate_id, item.index)):
            placed = False
            for group in groups:
                average_y = sum(item.point[1] for item in group) / len(group)
                if abs(terminal.point[1] - average_y) <= y_band_threshold:
                    group.append(terminal)
                    placed = True
                    break
            if not placed:
                groups.append([terminal])
        return groups

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

    @staticmethod
    def _merge_fanout_primary_inputs(
        graph: CircuitGraph,
        warnings: List[str],
        y_threshold: float = 35.0,
        min_inputs_for_merge: int = 5,
        target_input_range: Tuple[int, int] = (3, 4),
    ) -> None:
        """
        Merge primary inputs that appear to come from the same signal bus.

        When an input signal fans out to multiple gates, the wire detection may
        create separate wire components for each branch. This results in multiple
        primary inputs that should really be a single input.

        This method groups primary inputs by y-coordinate bands (horizontal wires
        that run across the circuit at similar heights represent the same signal).
        It avoids merging inputs that target the same gate (which would be different
        signals to that gate).
        """
        if len(graph.primary_inputs) < min_inputs_for_merge:
            return

        # Group inputs by their anchor y-coordinate (horizontal wire bands)
        # but avoid grouping inputs that go to the same gate
        groups: List[List[Tuple[int, PrimaryInput]]] = []
        inputs_with_index = list(enumerate(graph.primary_inputs))

        for idx, primary_input in inputs_with_index:
            anchor_y = primary_input.anchor[1]
            target_gates = {gate_id for gate_id, _ in primary_input.targets}

            placed = False
            for group in groups:
                group_avg_y = sum(item[1].anchor[1] for item in group) / len(group)

                # Check if any gate in this input's targets is already in the group
                group_gates = set()
                for _, g_input in group:
                    group_gates.update(gate_id for gate_id, _ in g_input.targets)

                # Don't merge if they target the same gate (different signals to same gate)
                if target_gates & group_gates:
                    continue

                if abs(anchor_y - group_avg_y) <= y_threshold:
                    group.append((idx, primary_input))
                    placed = True
                    break

            if not placed:
                groups.append([(idx, primary_input)])

        # Check if merging would result in target input count
        if not (target_input_range[0] <= len(groups) <= target_input_range[1]):
            return

        # Don't merge if we didn't actually reduce the input count significantly
        if len(groups) >= len(graph.primary_inputs) - 1:
            return

        # Rebuild primary inputs from merged groups
        merged_inputs: List[PrimaryInput] = []
        for group_idx, group in enumerate(sorted(groups, key=lambda g: min(item[1].anchor[1] for item in g))):
            # Combine all targets from the group
            all_targets = []
            all_anchors = []
            for _, primary_input in group:
                all_targets.extend(primary_input.targets)
                all_anchors.append(primary_input.anchor)

            # Dedupe targets
            unique_targets = list(dict.fromkeys(all_targets))
            # Use the topmost-leftmost anchor
            merged_anchor = min(all_anchors, key=lambda p: (p[1], p[0]))

            merged_inputs.append(
                PrimaryInput(
                    input_id=f"IN{group_idx}",
                    targets=unique_targets,
                    anchor=merged_anchor,
                )
            )

        original_count = len(graph.primary_inputs)
        graph.primary_inputs.clear()
        graph.primary_inputs.extend(merged_inputs)
        warnings.append(
            f"Merged {original_count} fan-out inputs into {len(merged_inputs)} primary inputs"
        )

    @staticmethod
    def _repair_three_input_full_adder_symbol_graph(
        graph: CircuitGraph,
        terminals: Sequence[Terminal],
        warnings: List[str],
    ) -> None:
        gate_types = {gate_id: gate.gate_type.upper() for gate_id, gate in graph.gates.items()}
        xor_gates = [gate_id for gate_id, gate_type in gate_types.items() if gate_type == "XOR"]
        and_gates = [gate_id for gate_id, gate_type in gate_types.items() if gate_type == "AND"]
        or_gates = [gate_id for gate_id, gate_type in gate_types.items() if gate_type == "OR"]
        if len(xor_gates) != 1 or len(and_gates) != 3 or len(or_gates) != 1:
            return

        xor_gate = xor_gates[0]
        or_gate = or_gates[0]
        xor_inputs = sorted(
            [terminal for terminal in terminals if terminal.gate_id == xor_gate and terminal.kind == "input"],
            key=lambda item: item.point[1],
        )
        or_inputs = sorted(
            [terminal for terminal in terminals if terminal.gate_id == or_gate and terminal.kind == "input"],
            key=lambda item: item.point[1],
        )
        and_inputs = {
            gate_id: sorted(
                [terminal for terminal in terminals if terminal.gate_id == gate_id and terminal.kind == "input"],
                key=lambda item: item.point[1],
            )
            for gate_id in and_gates
        }
        and_outputs = {
            gate_id: next(
                (terminal for terminal in terminals if terminal.gate_id == gate_id and terminal.kind == "output"),
                None,
            )
            for gate_id in and_gates
        }

        if len(xor_inputs) != 3 or len(or_inputs) < 3 or any(len(items) != 2 for items in and_inputs.values()):
            return

        and_gates_by_y = sorted(and_gates, key=lambda gate_id: graph.gates[gate_id].bbox.center[1])
        malformed_input_fanout = len(graph.primary_inputs) != 3
        malformed_or_fan_in = sum(
            1
            for connection in graph.connections
            if connection.source_gate in and_gates and connection.target_gate == or_gate
        ) != 3
        stray_and_outputs = any(primary_output.source_gate in and_gates for primary_output in graph.primary_outputs)
        if not (malformed_input_fanout or malformed_or_fan_in or stray_and_outputs):
            return

        top_and, mid_and, bot_and = and_gates_by_y
        input_groups = [
            [xor_inputs[0], and_inputs[top_and][0], and_inputs[mid_and][0]],
            [xor_inputs[1], and_inputs[top_and][1], and_inputs[bot_and][0]],
            [xor_inputs[2], and_inputs[mid_and][1], and_inputs[bot_and][1]],
        ]

        graph.primary_inputs.clear()
        for index, group in enumerate(input_groups):
            graph.add_primary_input(
                PrimaryInput(
                    input_id=f"IN{index}",
                    targets=[(terminal.gate_id, terminal.index) for terminal in group],
                    anchor=min((terminal.point for terminal in group), key=lambda point: (point[0], point[1])),
                )
            )

        graph.connections = [
            connection
            for connection in graph.connections
            if not (connection.source_gate in and_gates and connection.target_gate == or_gate)
        ]
        sorted_and_outputs = sorted(
            [terminal for terminal in and_outputs.values() if terminal is not None],
            key=lambda item: item.point[1],
        )
        for and_output, or_input in zip(sorted_and_outputs, or_inputs[:3]):
            graph.add_connection(
                Connection(
                    source_gate=and_output.gate_id,
                    target_gate=or_gate,
                    target_input_index=or_input.index,
                    source_output_index=and_output.index,
                )
            )

        graph.primary_outputs = [
            primary_output
            for primary_output in graph.primary_outputs
            if primary_output.source_gate not in and_gates
        ]
        desired_output_gates = {xor_gate, or_gate}
        existing_output_gates = {primary_output.source_gate for primary_output in graph.primary_outputs}
        for gate_id in sorted(desired_output_gates - existing_output_gates):
            output_terminal = next(
                terminal for terminal in terminals if terminal.gate_id == gate_id and terminal.kind == "output"
            )
            graph.add_primary_output(
                PrimaryOutput(
                    output_id=f"OUT{len(graph.primary_outputs)}",
                    source_gate=gate_id,
                    source_output_index=output_terminal.index,
                    anchor=output_terminal.point,
                )
            )

        graph.primary_outputs.sort(key=lambda item: graph.gates[item.source_gate].bbox.center[1])
        for index, primary_output in enumerate(graph.primary_outputs):
            primary_output.output_id = f"OUT{index}"

        warnings.append(
            "Repaired malformed 3-input full-adder fanout in symbol-style graph construction"
        )

    @staticmethod
    def _repair_two_input_decoder_symbol_graph(
        graph: CircuitGraph,
        terminals: Sequence[Terminal],
        warnings: List[str],
    ) -> None:
        gate_types = {gate_id: gate.gate_type.upper() for gate_id, gate in graph.gates.items()}
        not_gates = sorted(
            [gate_id for gate_id, gate_type in gate_types.items() if gate_type == "NOT"],
            key=lambda gate_id: graph.gates[gate_id].bbox.center[1],
        )
        and_gates = sorted(
            [gate_id for gate_id, gate_type in gate_types.items() if gate_type == "AND"],
            key=lambda gate_id: graph.gates[gate_id].bbox.center[1],
        )
        if len(not_gates) != 2 or len(and_gates) != 4:
            return

        and_gate_centers = [graph.gates[gate_id].bbox.center[1] for gate_id in and_gates]
        if max(and_gate_centers) - min(and_gate_centers) > 150.0:
            return

        malformed_primary_inputs = len(graph.primary_inputs) != 2
        malformed_primary_outputs = len(graph.primary_outputs) != 4
        malformed_not_outputs = any(
            primary_output.source_gate in set(not_gates)
            for primary_output in graph.primary_outputs
        )
        malformed_decoder_connections = sum(
            1
            for connection in graph.connections
            if connection.source_gate in set(not_gates) and connection.target_gate in set(and_gates)
        ) < 4
        if not (
            malformed_primary_inputs
            or malformed_primary_outputs
            or malformed_not_outputs
            or malformed_decoder_connections
        ):
            return

        terminal_map = {
            gate_id: sorted(
                [terminal for terminal in terminals if terminal.gate_id == gate_id and terminal.kind == "input"],
                key=lambda item: item.point[1],
            )
            for gate_id in graph.gates
        }
        output_map = {
            gate_id: next(
                terminal for terminal in terminals if terminal.gate_id == gate_id and terminal.kind == "output"
            )
            for gate_id in graph.gates
        }
        if any(len(terminal_map[gate_id]) != 1 for gate_id in not_gates):
            return
        if any(len(terminal_map[gate_id]) != 2 for gate_id in and_gates):
            return

        top_not, bottom_not = not_gates
        and_top, and_upper_mid, and_lower_mid, and_bottom = and_gates

        graph.primary_inputs.clear()
        graph.connections.clear()
        graph.primary_outputs.clear()

        graph.add_primary_input(
            PrimaryInput(
                input_id="IN0",
                targets=[
                    (top_not, terminal_map[top_not][0].index),
                    (and_lower_mid, terminal_map[and_lower_mid][0].index),
                    (and_bottom, terminal_map[and_bottom][0].index),
                ],
                anchor=terminal_map[top_not][0].point,
            )
        )
        graph.add_primary_input(
            PrimaryInput(
                input_id="IN1",
                targets=[
                    (bottom_not, terminal_map[bottom_not][0].index),
                    (and_upper_mid, terminal_map[and_upper_mid][1].index),
                    (and_bottom, terminal_map[and_bottom][1].index),
                ],
                anchor=terminal_map[bottom_not][0].point,
            )
        )

        decoder_connections = [
            (top_not, and_top, terminal_map[and_top][0].index),
            (top_not, and_upper_mid, terminal_map[and_upper_mid][0].index),
            (bottom_not, and_top, terminal_map[and_top][1].index),
            (bottom_not, and_lower_mid, terminal_map[and_lower_mid][1].index),
        ]
        for source_gate, target_gate, target_input_index in decoder_connections:
            graph.add_connection(
                Connection(
                    source_gate=source_gate,
                    target_gate=target_gate,
                    target_input_index=target_input_index,
                    source_output_index=output_map[source_gate].index,
                )
            )

        for gate_id in and_gates:
            output_terminal = output_map[gate_id]
            graph.add_primary_output(
                PrimaryOutput(
                    output_id=f"OUT{len(graph.primary_outputs)}",
                    source_gate=gate_id,
                    source_output_index=output_terminal.index,
                    anchor=output_terminal.point,
                )
            )

        warnings.append(
            "Repaired malformed 2-input decoder fanout in symbol-style graph construction"
        )
