"""
Wire detection using classical computer vision and a crossing-aware skeleton graph.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from PIL import Image

from .types import BoundingBox, GateDetection, Point, WireComponent, WireSegment


ImageLike = Union[str, Path, np.ndarray, Image.Image]
Pixel = Tuple[int, int]


@dataclass
class WireDetectionResult:
    """Container for wire detection outputs."""

    segments: List[WireSegment]
    components: List[WireComponent]
    mask: np.ndarray


@dataclass(frozen=True)
class _SkeletonNode:
    node_id: int
    pixels: Tuple[Pixel, ...]
    point: Point


@dataclass(frozen=True)
class _SkeletonEdge:
    edge_id: int
    start_node: int
    end_node: int
    path: Tuple[Pixel, ...]


class WireDetector:
    """Detect wire-like topology outside gate bounding boxes."""

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 25,
        min_line_length: int = 20,
        max_line_gap: int = 10,
        gate_mask_padding: int = 2,
        component_join_distance: float = 12.0,
        junction_neighbor_threshold: int = 3,
        node_cluster_distance: float = 6.0,
        crossing_dot_threshold: float = 0.75,
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.gate_mask_padding = gate_mask_padding
        self.component_join_distance = component_join_distance
        self.junction_neighbor_threshold = junction_neighbor_threshold
        self.node_cluster_distance = node_cluster_distance
        self.crossing_dot_threshold = crossing_dot_threshold

    def detect(self, image: ImageLike, gate_boxes: Sequence[Union[GateDetection, BoundingBox]]) -> WireDetectionResult:
        cv2 = self._import_cv2()

        bgr = self._load_bgr(image)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = self._build_mask(gray, gate_boxes)
        skeleton = self._skeletonize(mask)
        components = self._extract_graph_components(skeleton)
        segments = [segment for component in components for segment in component.segments]
        return WireDetectionResult(segments=segments, components=components, mask=skeleton)

    def _build_mask(self, gray: np.ndarray, gate_boxes: Sequence[Union[GateDetection, BoundingBox]]) -> np.ndarray:
        cv2 = self._import_cv2()

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary_inv = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel, iterations=1)

        for item in gate_boxes:
            bbox = item.bbox if isinstance(item, GateDetection) else item
            x1, y1, x2, y2 = bbox.expanded(self.gate_mask_padding).to_int_tuple()
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(cleaned.shape[1] - 1, x2)
            y2 = min(cleaned.shape[0] - 1, y2)
            cleaned[y1:y2, x1:x2] = 0

        return cleaned

    def _skeletonize(self, mask: np.ndarray) -> np.ndarray:
        """Thin wire mask to a 1-pixel skeleton using Zhang-Suen thinning."""
        binary = (mask > 0).astype(np.uint8)
        if not np.any(binary):
            return mask

        thinned = binary.copy()
        changed = True

        while changed:
            changed = False
            for step in (0, 1):
                points_to_remove = []
                for y in range(1, thinned.shape[0] - 1):
                    for x in range(1, thinned.shape[1] - 1):
                        if thinned[y, x] == 0:
                            continue
                        neighbors = self._neighbors(thinned, x, y)
                        count = sum(neighbors)
                        if count < 2 or count > 6:
                            continue
                        transitions = self._neighbor_transitions(neighbors)
                        if transitions != 1:
                            continue
                        p2, _, p4, _, p6, _, p8, _ = neighbors
                        if step == 0:
                            if p2 * p4 * p6 != 0:
                                continue
                            if p4 * p6 * p8 != 0:
                                continue
                        else:
                            if p2 * p4 * p8 != 0:
                                continue
                            if p2 * p6 * p8 != 0:
                                continue
                        points_to_remove.append((y, x))
                if points_to_remove:
                    changed = True
                    for y, x in points_to_remove:
                        thinned[y, x] = 0

        return (thinned * 255).astype(np.uint8)

    def _extract_graph_components(self, mask: np.ndarray) -> List[WireComponent]:
        skeleton = (mask > 0).astype(np.uint8)
        pixels = {(x, y) for y, x in zip(*np.where(skeleton > 0))}
        if not pixels:
            return []

        degree_map = {pixel: self._pixel_degree(skeleton, pixel[0], pixel[1]) for pixel in pixels}
        key_pixels = {pixel for pixel, degree in degree_map.items() if degree != 2}
        if not key_pixels:
            return self._component_from_pixels(pixels)

        nodes = self._build_nodes(key_pixels)
        pixel_to_node = {pixel: node.node_id for node in nodes for pixel in node.pixels}
        edges = self._trace_edges(skeleton, nodes, pixel_to_node)
        if not edges:
            return self._component_from_pixels(pixels)

        return self._build_net_components(nodes, edges)

    def _build_nodes(self, key_pixels: Set[Pixel]) -> List[_SkeletonNode]:
        clusters = self._cluster_key_pixels(key_pixels)
        nodes: List[_SkeletonNode] = []
        for node_id, cluster in enumerate(clusters):
            nodes.append(
                _SkeletonNode(
                    node_id=node_id,
                    pixels=tuple(cluster),
                    point=self._cluster_center(cluster),
                )
            )
        return nodes

    def _trace_edges(
        self,
        skeleton: np.ndarray,
        nodes: Sequence[_SkeletonNode],
        pixel_to_node: Dict[Pixel, int],
    ) -> List[_SkeletonEdge]:
        visited_edges: Set[Tuple[Pixel, Pixel]] = set()
        graph_edges: List[_SkeletonEdge] = []

        for node in nodes:
            cluster_pixels = set(node.pixels)
            for pixel in node.pixels:
                for neighbor in self._neighbor_pixels(skeleton, pixel[0], pixel[1]):
                    if neighbor in cluster_pixels:
                        continue
                    edge_key = self._edge_key(pixel, neighbor)
                    if edge_key in visited_edges:
                        continue
                    path = self._trace_path(skeleton, pixel, neighbor, pixel_to_node, visited_edges)
                    if len(path) < 2:
                        continue
                    end_node = pixel_to_node.get(path[-1])
                    if end_node is None or end_node == node.node_id:
                        continue
                    graph_edges.append(
                        _SkeletonEdge(
                            edge_id=len(graph_edges),
                            start_node=node.node_id,
                            end_node=end_node,
                            path=tuple(path),
                        )
                    )

        deduped: List[_SkeletonEdge] = []
        seen_pairs: Set[Tuple[int, int, Tuple[Pixel, ...]]] = set()
        for edge in graph_edges:
            if edge.start_node <= edge.end_node:
                pair = (edge.start_node, edge.end_node)
                canonical_path = edge.path
            else:
                pair = (edge.end_node, edge.start_node)
                canonical_path = tuple(reversed(edge.path))
            marker = (pair[0], pair[1], canonical_path)
            if marker in seen_pairs:
                continue
            seen_pairs.add(marker)
            deduped.append(
                _SkeletonEdge(
                    edge_id=len(deduped),
                    start_node=pair[0],
                    end_node=pair[1],
                    path=canonical_path,
                )
            )
        return deduped

    def _trace_path(
        self,
        skeleton: np.ndarray,
        start: Pixel,
        next_pixel: Pixel,
        pixel_to_node: Dict[Pixel, int],
        visited_edges: Set[Tuple[Pixel, Pixel]],
    ) -> List[Pixel]:
        path = [start, next_pixel]
        visited_edges.add(self._edge_key(start, next_pixel))
        previous = start
        current = next_pixel

        while True:
            if current in pixel_to_node and current != start:
                break

            neighbors = [pixel for pixel in self._neighbor_pixels(skeleton, current[0], current[1]) if pixel != previous]
            if not neighbors:
                break

            next_candidates = [pixel for pixel in neighbors if self._edge_key(current, pixel) not in visited_edges]
            if not next_candidates:
                break

            if len(next_candidates) > 1:
                break

            nxt = next_candidates[0]
            visited_edges.add(self._edge_key(current, nxt))
            path.append(nxt)
            previous, current = current, nxt

        return path

    def _build_net_components(
        self,
        nodes: Sequence[_SkeletonNode],
        edges: Sequence[_SkeletonEdge],
    ) -> List[WireComponent]:
        if not edges:
            return []

        node_lookup = {node.node_id: node for node in nodes}
        edge_lookup = {edge.edge_id: edge for edge in edges}
        edge_directions = self._edge_directions(node_lookup, edge_lookup)

        parent = list(range(len(edges)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(a: int, b: int) -> None:
            root_a = find(a)
            root_b = find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        incident_edges: Dict[int, List[int]] = {}
        for edge in edges:
            incident_edges.setdefault(edge.start_node, []).append(edge.edge_id)
            incident_edges.setdefault(edge.end_node, []).append(edge.edge_id)

        for node_id, node_edge_ids in incident_edges.items():
            if len(node_edge_ids) <= 1:
                continue
            if self._is_crossing_node(node_id, node_edge_ids, edge_directions):
                for first, second in self._pair_crossing_edges(node_id, node_edge_ids, edge_directions):
                    union(first, second)
                continue
            for other_edge_id in node_edge_ids[1:]:
                union(node_edge_ids[0], other_edge_id)

        grouped: Dict[int, List[_SkeletonEdge]] = {}
        for edge in edges:
            grouped.setdefault(find(edge.edge_id), []).append(edge)

        components: List[WireComponent] = []
        for group in grouped.values():
            segments: List[WireSegment] = []
            representative_points: List[Point] = []

            for edge in group:
                path_points = [self._to_point(pixel) for pixel in edge.path]
                segments.extend(self._path_to_segments(path_points))
                representative_points.extend(self._sample_path_points(path_points))
                representative_points.append(node_lookup[edge.start_node].point)
                representative_points.append(node_lookup[edge.end_node].point)

            if not segments:
                continue

            components.append(
                WireComponent(
                    component_id=f"wire_{len(components)}",
                    segments=segments,
                    points=self._unique_points(representative_points),
                )
            )

        return components

    def _edge_directions(
        self,
        nodes: Dict[int, _SkeletonNode],
        edges: Dict[int, _SkeletonEdge],
    ) -> Dict[Tuple[int, int], np.ndarray]:
        directions: Dict[Tuple[int, int], np.ndarray] = {}
        for edge in edges.values():
            directions[(edge.edge_id, edge.start_node)] = self._direction_from_node(nodes[edge.start_node], edge.path, from_start=True)
            directions[(edge.edge_id, edge.end_node)] = self._direction_from_node(nodes[edge.end_node], edge.path, from_start=False)
        return directions

    def _direction_from_node(self, node: _SkeletonNode, path: Sequence[Pixel], from_start: bool) -> np.ndarray:
        node_center = np.array(node.point, dtype=float)
        samples = path[1: min(len(path), 6)] if from_start else list(reversed(path[max(0, len(path) - 6):-1]))
        for pixel in samples:
            vector = np.array((float(pixel[0]), float(pixel[1])), dtype=float) - node_center
            norm = float(np.linalg.norm(vector))
            if norm > 0.0:
                return vector / norm
        return np.zeros(2, dtype=float)

    def _is_crossing_node(
        self,
        node_id: int,
        edge_ids: Sequence[int],
        edge_directions: Dict[Tuple[int, int], np.ndarray],
    ) -> bool:
        if len(edge_ids) != 4:
            return False

        pairings = self._crossing_pairings(edge_ids)
        for pairing in pairings:
            opposite_scores = []
            orthogonal_scores = []
            valid = True
            for first, second in pairing:
                direction_a = edge_directions.get((first, node_id))
                direction_b = edge_directions.get((second, node_id))
                if direction_a is None or direction_b is None:
                    valid = False
                    break
                dot = float(np.dot(direction_a, direction_b))
                opposite_scores.append(dot)
            if not valid:
                continue

            first_pair = pairing[0]
            second_pair = pairing[1]
            for first in first_pair:
                for second in second_pair:
                    direction_a = edge_directions[(first, node_id)]
                    direction_b = edge_directions[(second, node_id)]
                    orthogonal_scores.append(abs(float(np.dot(direction_a, direction_b))))

            if all(score <= -self.crossing_dot_threshold for score in opposite_scores) and all(
                score <= (1.0 - self.crossing_dot_threshold) for score in orthogonal_scores
            ):
                return True
        return False

    @staticmethod
    def _crossing_pairings(edge_ids: Sequence[int]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        a, b, c, d = edge_ids
        return [
            ((a, b), (c, d)),
            ((a, c), (b, d)),
            ((a, d), (b, c)),
        ]

    def _pair_crossing_edges(
        self,
        node_id: int,
        edge_ids: Sequence[int],
        edge_directions: Dict[Tuple[int, int], np.ndarray],
    ) -> List[Tuple[int, int]]:
        best_score = float("-inf")
        best_pairing: List[Tuple[int, int]] = []
        for pairing in self._crossing_pairings(edge_ids):
            score = 0.0
            for first, second in pairing:
                direction_a = edge_directions[(first, node_id)]
                direction_b = edge_directions[(second, node_id)]
                score += -float(np.dot(direction_a, direction_b))
            if score > best_score:
                best_score = score
                best_pairing = [pairing[0], pairing[1]]
        return best_pairing

    def _component_from_pixels(self, pixels: Set[Pixel]) -> List[WireComponent]:
        points = [self._to_point(pixel) for pixel in sorted(pixels)]
        segments = self._path_to_segments(points[:2]) if len(points) >= 2 else []
        return [WireComponent(component_id="wire_0", segments=segments, points=self._summarize_points(points))]

    def _cluster_key_pixels(self, key_pixels: Set[Pixel]) -> List[List[Pixel]]:
        remaining = set(key_pixels)
        clusters: List[List[Pixel]] = []
        while remaining:
            seed = remaining.pop()
            cluster = [seed]
            stack = [seed]
            while stack:
                pixel = stack.pop()
                neighbors = [candidate for candidate in list(remaining) if self._point_distance(self._to_point(pixel), self._to_point(candidate)) <= self.node_cluster_distance]
                for neighbor in neighbors:
                    remaining.remove(neighbor)
                    cluster.append(neighbor)
                    stack.append(neighbor)
            clusters.append(cluster)
        return clusters

    @staticmethod
    def _cluster_center(cluster: Sequence[Pixel]) -> Point:
        xs = [pixel[0] for pixel in cluster]
        ys = [pixel[1] for pixel in cluster]
        return (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))

    @staticmethod
    def _neighbors(binary: np.ndarray, x: int, y: int) -> List[int]:
        return [
            int(binary[y - 1, x]),
            int(binary[y - 1, x + 1]),
            int(binary[y, x + 1]),
            int(binary[y + 1, x + 1]),
            int(binary[y + 1, x]),
            int(binary[y + 1, x - 1]),
            int(binary[y, x - 1]),
            int(binary[y - 1, x - 1]),
        ]

    @staticmethod
    def _neighbor_transitions(neighbors: Sequence[int]) -> int:
        wrapped = list(neighbors) + [neighbors[0]]
        return sum(1 for a, b in zip(wrapped, wrapped[1:]) if a == 0 and b == 1)

    @staticmethod
    def _adjacent_pixels(pixel: Pixel) -> List[Pixel]:
        x, y = pixel
        return [
            (x + dx, y + dy)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if not (dx == 0 and dy == 0)
        ]

    def _neighbor_pixels(self, skeleton: np.ndarray, x: int, y: int) -> List[Pixel]:
        neighbors: List[Pixel] = []
        for nx, ny in self._adjacent_pixels((x, y)):
            if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1] and skeleton[ny, nx] > 0:
                neighbors.append((nx, ny))
        return neighbors

    def _pixel_degree(self, skeleton: np.ndarray, x: int, y: int) -> int:
        return len(self._neighbor_pixels(skeleton, x, y))

    @staticmethod
    def _edge_key(a: Pixel, b: Pixel) -> Tuple[Pixel, Pixel]:
        return (a, b) if a <= b else (b, a)

    @staticmethod
    def _to_point(pixel: Pixel) -> Point:
        return (float(pixel[0]), float(pixel[1]))

    def _sample_path_points(self, path_points: Sequence[Point]) -> List[Point]:
        if len(path_points) <= 2:
            return list(path_points)
        step = max(1, len(path_points) // 8)
        sampled = [path_points[0]]
        sampled.extend(path_points[index] for index in range(step, len(path_points) - 1, step))
        sampled.append(path_points[-1])
        return sampled

    def _path_to_segments(self, path_points: Sequence[Point]) -> List[WireSegment]:
        segments: List[WireSegment] = []
        for start, end in zip(path_points, path_points[1:]):
            length = self._point_distance(start, end)
            if length == 0:
                continue
            segments.append(WireSegment(start=start, end=end, length=length))
        return segments

    def _summarize_points(self, points: Sequence[Point]) -> List[Point]:
        if not points:
            return []
        left = min(points, key=lambda point: point[0])
        right = max(points, key=lambda point: point[0])
        top = min(points, key=lambda point: point[1])
        bottom = max(points, key=lambda point: point[1])
        return self._unique_points([left, right, top, bottom])

    def _unique_points(self, points: Iterable[Point]) -> List[Point]:
        unique: List[Point] = []
        for point in points:
            if all(self._point_distance(point, existing) > self.component_join_distance for existing in unique):
                unique.append(point)
        return unique

    @staticmethod
    def _point_distance(a: Point, b: Point) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    @staticmethod
    def _load_bgr(image: ImageLike) -> np.ndarray:
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
            rgb = np.array(pil_image)
        elif isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
        else:
            rgb = image
            if rgb.ndim == 2:
                rgb = np.stack([rgb, rgb, rgb], axis=-1)
        return rgb[:, :, ::-1].copy()

    @staticmethod
    def _import_cv2():
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("opencv-python is required for wire detection") from exc
        return cv2
