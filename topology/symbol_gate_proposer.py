"""
Heuristic proposal generator for symbol-style gate schematics.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image

from .types import BoundingBox, GateDetection


@dataclass(frozen=True)
class ProposalDebug:
    mask: np.ndarray
    line_mask: np.ndarray
    shape_mask: np.ndarray


class SymbolGateProposer:
    """Generate candidate gate boxes from symbol-style schematics using classical CV heuristics."""

    def __init__(
        self,
        white_threshold: int = 220,
        dark_background_threshold: int = 105,
        line_kernel: int = 25,
        min_area_ratio: float = 0.00025,
        min_dimension: int = 10,
        merge_gap: int = 18,
        pad: int = 8,
        max_box_area_ratio: float = 0.20,
    ):
        self.white_threshold = white_threshold
        self.dark_background_threshold = dark_background_threshold
        self.line_kernel = line_kernel
        self.min_area_ratio = min_area_ratio
        self.min_dimension = min_dimension
        self.merge_gap = merge_gap
        self.pad = pad
        self.max_box_area_ratio = max_box_area_ratio

    def propose(self, image_path: Path) -> List[GateDetection]:
        proposals, _ = self.propose_with_debug(image_path)
        return proposals

    def propose_with_debug(self, image_path: Path) -> Tuple[List[GateDetection], ProposalDebug]:
        cv2 = self._import_cv2()
        with Image.open(image_path) as image:
            rgb = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        mask = self._build_gate_mask(rgb, gray)
        line_mask = self._estimate_line_mask(mask)
        shape_mask = cv2.subtract(mask, line_mask)
        shape_mask = cv2.morphologyEx(
            shape_mask,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.uint8),
            iterations=1,
        )

        raw_boxes = self._extract_component_boxes(shape_mask)
        merged_boxes = self._merge_boxes(raw_boxes, gray.shape[1], gray.shape[0])
        finalized_raw_boxes = self._finalize_boxes(raw_boxes, gray.shape[1], gray.shape[0])
        candidate_boxes = self._dedupe_boxes(
            list(merged_boxes) + list(finalized_raw_boxes),
            gray.shape[1],
            gray.shape[0],
        )
        proposals = [
            GateDetection(
                gate_id=f"proposal_{index}",
                gate_type="UNKNOWN",
                bbox=BoundingBox(*box),
                confidence=1.0,
            )
            for index, box in enumerate(sorted(candidate_boxes, key=lambda item: (item[0], item[1])))
        ]
        return proposals, ProposalDebug(mask=mask, line_mask=line_mask, shape_mask=shape_mask)

    def _build_gate_mask(self, rgb: np.ndarray, gray: np.ndarray) -> np.ndarray:
        cv2 = self._import_cv2()
        if float(gray.mean()) < self.dark_background_threshold:
            channel_max = rgb.max(axis=2)
            channel_min = rgb.min(axis=2)
            mask = ((channel_max > 80) & ((channel_max - channel_min) > 40)).astype(np.uint8) * 255
        else:
            _, mask = cv2.threshold(gray, self.white_threshold, 255, cv2.THRESH_BINARY_INV)
        return mask.astype(np.uint8)

    def _estimate_line_mask(self, mask: np.ndarray) -> np.ndarray:
        cv2 = self._import_cv2()
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.line_kernel, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.line_kernel))
        horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)
        vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel)
        return cv2.bitwise_or(horizontal, vertical)

    def _extract_component_boxes(self, mask: np.ndarray) -> List[Tuple[float, float, float, float]]:
        cv2 = self._import_cv2()
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        image_area = mask.shape[0] * mask.shape[1]
        boxes: List[Tuple[float, float, float, float]] = []

        for component_id in range(1, component_count):
            x, y, width, height, area = stats[component_id]
            if area < max(25, int(image_area * self.min_area_ratio)):
                continue
            if width < self.min_dimension or height < self.min_dimension:
                continue

            aspect_ratio = width / max(height, 1)
            fill_ratio = area / max(width * height, 1)
            if aspect_ratio < 0.25 or aspect_ratio > 8.0:
                continue
            if fill_ratio < 0.03 or fill_ratio > 0.90:
                continue

            boxes.append((float(x), float(y), float(x + width), float(y + height)))

        return boxes

    def _merge_boxes(
        self,
        boxes: Sequence[Tuple[float, float, float, float]],
        image_width: int,
        image_height: int,
    ) -> List[Tuple[float, float, float, float]]:
        pending = [list(box) for box in boxes]
        merged = True
        while merged:
            merged = False
            output: List[List[float]] = []
            used = [False] * len(pending)
            for index, box in enumerate(pending):
                if used[index]:
                    continue
                x1, y1, x2, y2 = box
                for inner_index in range(index + 1, len(pending)):
                    if used[inner_index]:
                        continue
                    candidate = pending[inner_index]
                    if self._boxes_are_adjacent((x1, y1, x2, y2), tuple(candidate)):
                        x1 = min(x1, candidate[0])
                        y1 = min(y1, candidate[1])
                        x2 = max(x2, candidate[2])
                        y2 = max(y2, candidate[3])
                        used[inner_index] = True
                        merged = True
                used[index] = True
                output.append([x1, y1, x2, y2])
            pending = output

        return self._finalize_boxes(pending, image_width, image_height)

    def _finalize_boxes(
        self,
        boxes: Sequence[Tuple[float, float, float, float]],
        image_width: int,
        image_height: int,
    ) -> List[Tuple[float, float, float, float]]:
        final_boxes: List[Tuple[float, float, float, float]] = []
        max_box_area = image_width * image_height * self.max_box_area_ratio
        for x1, y1, x2, y2 in boxes:
            x1 = max(0.0, x1 - self.pad)
            y1 = max(0.0, y1 - self.pad)
            x2 = min(float(image_width), x2 + self.pad)
            y2 = min(float(image_height), y2 + self.pad)
            area = (x2 - x1) * (y2 - y1)
            if area > max_box_area:
                continue
            final_boxes.append((x1, y1, x2, y2))
        return final_boxes

    def _augment_boxes(
        self,
        boxes: Sequence[Tuple[float, float, float, float]],
        image_width: int,
        image_height: int,
        aggressive: bool = False,
    ) -> List[Tuple[float, float, float, float]]:
        augmented: List[Tuple[float, float, float, float]] = []

        # Real symbol diagrams often leave only the right half of curvy gate bodies after
        # line suppression. Add a conservative asymmetric expansion on right-half fragments.
        for x1, y1, x2, y2 in boxes:
            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + x2) / 2.0
            if center_x < image_width * 0.45:
                continue
            if width < 24 or height < 24 or width > 90 or height > 90:
                continue
            left_pad = max(width * 0.6, float(self.pad * 2))
            right_pad = max(width * 0.35, float(self.pad))
            vertical_pad = max(height * 0.12, 2.0)
            augmented.append(
                (
                    max(0.0, x1 - left_pad),
                    max(0.0, y1 - vertical_pad),
                    min(float(image_width), x2 + right_pad),
                    min(float(image_height), y2 + vertical_pad),
                )
            )

        if aggressive:
            # Line-drawing and ANSI-style schematics often leave a compact inner body after
            # line suppression. Add a broader recovery box so wide curved gates can still be
            # proposed from that surviving core fragment.
            for x1, y1, x2, y2 in boxes:
                width = x2 - x1
                height = y2 - y1
                if width < 28 or height < 20:
                    continue
                if width > 110 or height > 95:
                    continue

                aspect_ratio = width / max(height, 1.0)
                if aspect_ratio < 0.55 or aspect_ratio > 2.6:
                    continue

                center_x = (x1 + x2) / 2.0
                left_bias = 1.8 if center_x >= image_width * 0.40 else 1.1
                right_bias = 0.8 if center_x >= image_width * 0.40 else 0.55
                vertical_bias = 1.0 if height >= 32 else 1.2
                augmented.append(
                    (
                        max(0.0, x1 - max(width * left_bias, float(self.pad * 2))),
                        max(0.0, y1 - max(height * vertical_bias, float(self.pad))),
                        min(float(image_width), x2 + max(width * right_bias, float(self.pad * 2))),
                        min(float(image_height), y2 + max(height * vertical_bias, float(self.pad))),
                    )
                )

            # NAND-heavy constructions often retain only the lower-right body after
            # line suppression. Recover a tighter upper-left shifted box for those cases.
            for x1, y1, x2, y2 in boxes:
                width = x2 - x1
                height = y2 - y1
                center_x = (x1 + x2) / 2.0
                if width < 70 or width > 100:
                    continue
                if height < 45 or height > 65:
                    continue
                if center_x < image_width * 0.18:
                    continue

                augmented.append(
                    (
                        max(0.0, x1 - max(width * 0.30, float(self.pad))),
                        max(0.0, y1 - max(height * 1.05, float(self.pad * 2))),
                        min(float(image_width), x2 + max(width * 0.02, 2.0)),
                        min(float(image_height), y2 + max(height * 0.12, 2.0)),
                    )
                )

        augmented.extend(self._build_row_cluster_boxes(boxes, image_width, image_height))
        augmented.extend(self._build_left_stacked_inverter_boxes(boxes, image_width, image_height))
        return augmented

    def _build_left_stacked_inverter_boxes(
        self,
        boxes: Sequence[Tuple[float, float, float, float]],
        image_width: int,
        image_height: int,
    ) -> List[Tuple[float, float, float, float]]:
        augmented: List[Tuple[float, float, float, float]] = []
        for x1, y1, x2, y2 in boxes:
            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + x2) / 2.0

            # Compact decoder inputs can collapse two stacked inverter triangles into
            # one thin left-side fragment after line suppression. Split that fragment
            # into two overlapping proposals so the NOT candidates can survive ranking.
            if center_x > image_width * 0.45:
                continue
            if width < 24 or width > 58:
                continue
            if height < 38 or height > 82:
                continue
            if height < width * 1.10:
                continue

            split_y = y1 + (height / 2.0)
            overlap = max(height * 0.12, 3.0)
            left_pad = max(width * 0.55, float(self.pad))
            right_pad = max(width * 0.10, 3.0)
            vertical_pad = max(height * 0.10, 2.0)

            augmented.append(
                (
                    max(0.0, x1 - left_pad),
                    max(0.0, y1 - vertical_pad),
                    min(float(image_width), x2 + right_pad),
                    min(float(image_height), split_y + overlap),
                )
            )
            augmented.append(
                (
                    max(0.0, x1 - left_pad),
                    max(0.0, split_y - overlap),
                    min(float(image_width), x2 + right_pad),
                    min(float(image_height), y2 + vertical_pad),
                )
            )
        return augmented

    def _build_row_cluster_boxes(
        self,
        boxes: Sequence[Tuple[float, float, float, float]],
        image_width: int,
        image_height: int,
    ) -> List[Tuple[float, float, float, float]]:
        candidate_indices = [
            index
            for index, box in enumerate(boxes)
            if ((box[0] + box[2]) / 2.0) >= image_width * 0.45
        ]
        if not candidate_indices:
            return []

        adjacency = {index: set() for index in candidate_indices}
        for position, first_index in enumerate(candidate_indices):
            for second_index in candidate_indices[position + 1 :]:
                if self._boxes_share_row_band(boxes[first_index], boxes[second_index]):
                    adjacency[first_index].add(second_index)
                    adjacency[second_index].add(first_index)

        visited = set()
        unions: List[Tuple[float, float, float, float]] = []
        for index in candidate_indices:
            if index in visited:
                continue
            stack = [index]
            cluster = []
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                cluster.append(boxes[current])
                stack.extend(adjacency[current] - visited)

            if len(cluster) < 2:
                continue

            x1 = min(item[0] for item in cluster)
            y1 = min(item[1] for item in cluster)
            x2 = max(item[2] for item in cluster)
            y2 = max(item[3] for item in cluster)
            width = x2 - x1
            height = y2 - y1
            if width < 90 or width > image_width * 0.40:
                continue
            if height < 28 or height > image_height * 0.40:
                continue
            unions.append((x1, y1, x2, y2))
        return self._finalize_boxes(unions, image_width, image_height)


    def _dedupe_boxes(
        self,
        boxes: Sequence[Tuple[float, float, float, float]],
        image_width: int,
        image_height: int,
    ) -> List[Tuple[float, float, float, float]]:
        unique: List[Tuple[float, float, float, float]] = []
        for box in sorted(boxes, key=self._box_area):
            if any(self._boxes_equivalent(box, existing) for existing in unique):
                continue
            unique.append(box)
        return unique

    def _boxes_are_adjacent(
        self,
        first: Tuple[float, float, float, float],
        second: Tuple[float, float, float, float],
    ) -> bool:
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        return not (
            ax2 + self.merge_gap < bx1
            or bx2 + self.merge_gap < ax1
            or ay2 + self.merge_gap < by1
            or by2 + self.merge_gap < ay1
        )

    def _boxes_share_row_band(
        self,
        first: Tuple[float, float, float, float],
        second: Tuple[float, float, float, float],
    ) -> bool:
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        overlap = max(0.0, min(ay2, by2) - max(ay1, by1))
        min_height = min(ay2 - ay1, by2 - by1)
        if overlap / max(min_height, 1.0) < 0.45:
            return False

        horizontal_gap = max(0.0, max(ax1, bx1) - min(ax2, bx2))
        return horizontal_gap <= self.merge_gap * 2.5

    @staticmethod
    def _box_area(box: Tuple[float, float, float, float]) -> float:
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    @classmethod
    def _boxes_equivalent(
        cls,
        first: Tuple[float, float, float, float],
        second: Tuple[float, float, float, float],
    ) -> bool:
        intersection = cls._intersection_area(first, second)
        if intersection <= 0:
            return False
        smaller_area = min(cls._box_area(first), cls._box_area(second))
        return intersection / max(smaller_area, 1e-6) >= 0.85

    @staticmethod
    def _intersection_area(
        first: Tuple[float, float, float, float],
        second: Tuple[float, float, float, float],
    ) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        return (x2 - x1) * (y2 - y1)

    @classmethod
    def _augmentation_box_is_duplicate(
        cls,
        candidate: Tuple[float, float, float, float],
        existing: Tuple[float, float, float, float],
    ) -> bool:
        intersection = cls._intersection_area(candidate, existing)
        if intersection <= 0:
            return False
        larger_area = max(cls._box_area(candidate), cls._box_area(existing))
        return intersection / max(larger_area, 1e-6) >= 0.85


    @staticmethod
    def _import_cv2():
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("opencv-python is required for symbol gate proposal generation") from exc
        return cv2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate heuristic symbol-style gate proposals")
    parser.add_argument("image", type=str, help="Path to a schematic image")
    parser.add_argument("--json", action="store_true", help="Print proposals as JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    image_path = Path(args.image)
    proposer = SymbolGateProposer()
    proposals = proposer.propose(image_path)
    payload = [
        {
            "gate_id": proposal.gate_id,
            "gate_type": proposal.gate_type,
            "bbox": [proposal.bbox.x1, proposal.bbox.y1, proposal.bbox.x2, proposal.bbox.y2],
            "confidence": proposal.confidence,
        }
        for proposal in proposals
    ]
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    print(f"Image: {image_path}")
    print(f"Proposals: {len(payload)}")
    for item in payload:
        print(f"- {item['gate_id']} {item['bbox']}")


if __name__ == "__main__":
    main()
