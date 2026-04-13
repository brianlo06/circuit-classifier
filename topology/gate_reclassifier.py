"""Utilities for reclassifying detected gate crops with the isolated-gate model."""

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from data_loader import prepare_circuit_image
from model import get_model

from .types import BoundingBox, GateDetection, GateReclassification


DEFAULT_GATE_CLASSIFIER_CHECKPOINT = (
    Path(__file__).resolve().parent.parent
    / "checkpoints_384_v3"
    / "best_model.pth"
)

class GateCropClassifier:
    """Run the isolated-gate classifier on crops from a schematic image."""

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_GATE_CLASSIFIER_CHECKPOINT,
        image_size: int = 384,
        crop_padding_ratio: float = 0.12,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.image_size = image_size
        self.crop_padding_ratio = crop_padding_ratio
        self.device = self._pick_device()
        self.model, self.class_names = self._load_model()
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @staticmethod
    def _pick_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Gate classifier checkpoint not found at {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        model_name = checkpoint.get("model_name", "small")
        class_names = checkpoint["class_names"]
        model = get_model(model_name, num_classes=len(class_names))
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()
        return model, class_names

    def classify_image(self, image: Image.Image, top_k: int = 3) -> Tuple[str, float, List[Tuple[str, float]]]:
        processed = prepare_circuit_image(image)
        tensor = self.transform(processed).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.class_names)))

        ranked = [
            (self.class_names[index.item()], float(prob.item()))
            for prob, index in zip(top_probs[0], top_indices[0])
        ]
        label, confidence = ranked[0]
        return label, confidence, ranked

    def classify_image_with_edge_suppression(
        self,
        image: Image.Image,
        top_k: int = 3,
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        cleaned = self._suppress_edge_connected_strokes(image)
        return self.classify_image(cleaned, top_k=top_k)

    def classify_detection_crop(
        self,
        image: Image.Image,
        detection: GateDetection,
        top_k: int = 3,
        suppress_edge_wires: bool = False,
    ) -> GateReclassification:
        crop_box = self._expanded_crop_box(detection.bbox, image.width, image.height)
        crop = image.crop(crop_box.to_int_tuple())
        if suppress_edge_wires:
            label, confidence, ranked = self.classify_image_with_edge_suppression(crop, top_k=top_k)
        else:
            label, confidence, ranked = self.classify_image(crop, top_k=top_k)
        return GateReclassification(
            gate_id=detection.gate_id,
            detector_label=detection.gate_type,
            detector_confidence=detection.confidence,
            classifier_label=label,
            classifier_confidence=confidence,
            bbox=detection.bbox,
            top_k=ranked,
        )

    def classify_detections(
        self,
        image_path: Path,
        detections: Sequence[GateDetection],
        top_k: int = 3,
        suppress_edge_wires: bool = False,
    ) -> List[GateReclassification]:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            return [
                self.classify_detection_crop(
                    rgb,
                    detection,
                    top_k=top_k,
                    suppress_edge_wires=suppress_edge_wires,
                )
                for detection in detections
            ]

    def _expanded_crop_box(self, bbox: BoundingBox, image_width: int, image_height: int) -> BoundingBox:
        pad = max(bbox.width, bbox.height) * self.crop_padding_ratio
        return BoundingBox(
            x1=max(0.0, bbox.x1 - pad),
            y1=max(0.0, bbox.y1 - pad),
            x2=min(float(image_width), bbox.x2 + pad),
            y2=min(float(image_height), bbox.y2 + pad),
        )

    @staticmethod
    def _suppress_edge_connected_strokes(image: Image.Image, threshold: int = 220) -> Image.Image:
        """
        Remove dark connected components that touch the crop border.
        This is an evaluation-time heuristic to suppress schematic wires entering the gate crop.
        """
        rgb = image.convert("RGB")
        gray = np.array(rgb.convert("L"))
        dark = gray < threshold
        if not np.any(dark):
            return rgb

        visited = np.zeros_like(dark, dtype=bool)
        height, width = dark.shape
        erase = np.zeros_like(dark, dtype=bool)

        def neighbors(x: int, y: int):
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx = x + dx
                ny = y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    yield nx, ny

        edge_points = []
        for x in range(width):
            edge_points.append((x, 0))
            edge_points.append((x, height - 1))
        for y in range(height):
            edge_points.append((0, y))
            edge_points.append((width - 1, y))

        for x, y in edge_points:
            if visited[y, x] or not dark[y, x]:
                continue
            stack = [(x, y)]
            component = []
            visited[y, x] = True
            while stack:
                cx, cy = stack.pop()
                component.append((cx, cy))
                for nx, ny in neighbors(cx, cy):
                    if visited[ny, nx] or not dark[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((nx, ny))

            for cx, cy in component:
                erase[cy, cx] = True

        cleaned = np.array(rgb)
        cleaned[erase] = 255
        return Image.fromarray(cleaned)
