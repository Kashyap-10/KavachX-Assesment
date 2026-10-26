"""
Open-Vocabulary Detection with YOLO-World.
Detects arbitrary object classes dynamically without retraining and performs spatial attribute association.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Detection:
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    label: str
    attributes: Dict[str, Any]


def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """Calculates Intersection over Union between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(1e-6, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1e-6, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return interArea / float(boxAArea + boxBArea - interArea)


def is_contained_or_near(inner_box: List[float], outer_box: List[float], margin_ratio: float = 0.2) -> bool:
    """Checks whether inner_box (e.g. bag) is inside or adjacent to outer_box (e.g. person)."""
    # Expand outer box slightly
    w = outer_box[2] - outer_box[0]
    h = outer_box[3] - outer_box[1]
    exp_x1 = outer_box[0] - margin_ratio * w
    exp_y1 = outer_box[1] - margin_ratio * h
    exp_x2 = outer_box[2] + margin_ratio * w
    exp_y2 = outer_box[3] + margin_ratio * h

    inner_center_x = (inner_box[0] + inner_box[2]) / 2.0
    inner_center_y = (inner_box[1] + inner_box[3]) / 2.0

    return (exp_x1 <= inner_center_x <= exp_x2) and (exp_y1 <= inner_center_y <= exp_y2)


class OpenVocabDetector:
    """Open-vocabulary object detector powered by YOLO-World."""

    def __init__(
        self,
        target_class: str = "person",
        attributes: Optional[Dict[str, Any]] = None,
        model_name: str = "yolov8s-worldv2.pt",
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
    ):
        self.target_class = target_class.lower()
        self.attributes = attributes or {}
        self.confidence_threshold = confidence_threshold

        from ultralytics import YOLO
        import torch

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[Detector] Initializing YOLO-World ({model_name}) on device: {self.device}")
        self.model = YOLO(model_name)

        # Assemble custom vocabulary list
        vocab = [self.target_class]
        if "carrying" in self.attributes:
            carry_item = self.attributes["carrying"].lower()
            vocab.extend([carry_item, "bag", "backpack", "handbag", "suitcase"])
        if "wearing" in self.attributes:
            wear_item = self.attributes["wearing"].lower()
            vocab.extend([wear_item, "helmet", "hard hat", "vest"])

        self.vocab = list(dict.fromkeys(vocab))
        print(f"[Detector] Setting open-vocabulary classes: {self.vocab}")
        self.model.set_classes(self.vocab)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Runs open-vocabulary detection and associates target objects with detected attributes."""
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        target_candidates = []
        attribute_objects = []

        if not results or len(results) == 0:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        names = self.model.names
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            cls_idx = int(box.cls[0].cpu().numpy())
            label = names[cls_idx].lower()

            if label == self.target_class:
                target_candidates.append({
                    "bbox": xyxy,
                    "confidence": conf,
                    "label": label,
                })
            else:
                attribute_objects.append({
                    "bbox": xyxy,
                    "confidence": conf,
                    "label": label,
                })

        # Attribute association
        matched_detections: List[Detection] = []
        requires_carrying = "carrying" in self.attributes
        target_carried_item = self.attributes.get("carrying", "").lower()

        for cand in target_candidates:
            matched_attrs = {}
            cand_bbox = cand["bbox"]

            if requires_carrying:
                has_bag = False
                for obj in attribute_objects:
                    if is_contained_or_near(obj["bbox"], cand_bbox):
                        has_bag = True
                        matched_attrs["carrying"] = self.attributes.get("carrying", "bag")
                        break
                if not has_bag:
                    # If the rule specifically requires carrying a bag, skip persons without a bag
                    continue

            matched_detections.append(Detection(
                bbox=cand_bbox,
                confidence=cand["confidence"],
                label=cand["label"],
                attributes=matched_attrs,
            ))

        return matched_detections
