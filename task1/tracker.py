"""
Multi-Object Tracking and Dwell Time Engine.
Implements robust multi-object tracking (IoU & spatial distance association)
and measures dwell time within specified polygon zones.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from shapely.geometry import Point, Polygon

from task1.detector import Detection, calculate_iou
from utils.geometry import get_foot_position, is_point_in_zone


@dataclass
class TrackViolation:
    track_id: int
    label: str
    attributes: Dict[str, Any]
    confidence: float
    dwell_s: float
    zone: str
    ts_start: str
    ts_end: str
    person_bbox: List[float]


class DwellTracker:
    """Tracks detected objects across frames and calculates accumulated dwell time in zones."""

    def __init__(
        self,
        zone_name: str,
        zone_polygon: Polygon,
        dwell_threshold_s: float = 10.0,
        fps: float = 30.0,
        max_lost_frames: int = 30,
        iou_threshold: float = 0.3,
    ):
        self.zone_name = zone_name
        self.zone_polygon = zone_polygon
        self.dwell_threshold_s = dwell_threshold_s
        self.fps = max(1.0, float(fps))
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold

        self.next_track_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def update(
        self,
        detections: List[Detection],
        frame_timestamp: Optional[datetime] = None,
    ) -> List[TrackViolation]:
        """Updates tracker with detections and returns violations triggered on this frame."""
        now = frame_timestamp or datetime.now(timezone.utc)
        iso_now = now.isoformat()
        violations: List[TrackViolation] = []

        for t_id in list(self.tracks.keys()):
            self.tracks[t_id]["lost_frames"] += 1
            if self.tracks[t_id]["lost_frames"] > self.max_lost_frames:
                del self.tracks[t_id]

        if len(detections) == 0:
            return violations

        matched_det_indices = set()
        matched_track_ids = set()

        for t_id, track in self.tracks.items():
            best_iou = self.iou_threshold
            best_idx = None

            for d_idx, det in enumerate(detections):
                if d_idx in matched_det_indices:
                    continue
                iou = calculate_iou(track["last_bbox"], det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = d_idx

            if best_idx is not None:
                matched_track_ids.add(t_id)
                matched_det_indices.add(best_idx)
                det = detections[best_idx]

                track["last_bbox"] = det.bbox
                track["last_conf"] = det.confidence
                track["attributes"] = det.attributes
                track["lost_frames"] = 0

                foot = get_foot_position(det.bbox)
                is_inside = is_point_in_zone(foot, self.zone_polygon)

                if is_inside:
                    if not track["in_zone"]:
                        track["in_zone"] = True
                        track["ts_start"] = iso_now

                    track["dwell_s"] += 1.0 / self.fps

                    if track["dwell_s"] >= self.dwell_threshold_s and not track["emitted"]:
                        track["emitted"] = True
                        violations.append(
                            TrackViolation(
                                track_id=t_id,
                                label=track["label"],
                                attributes=track["attributes"],
                                confidence=track["last_conf"],
                                dwell_s=round(track["dwell_s"], 2),
                                zone=self.zone_name,
                                ts_start=track["ts_start"],
                                ts_end=iso_now,
                                person_bbox=det.bbox,
                            )
                        )
                else:
                    track["in_zone"] = False

        for d_idx, det in enumerate(detections):
            if d_idx not in matched_det_indices:
                foot = get_foot_position(det.bbox)
                is_inside = is_point_in_zone(foot, self.zone_polygon)

                t_id = self.next_track_id
                self.next_track_id += 1

                self.tracks[t_id] = {
                    "last_bbox": det.bbox,
                    "last_conf": det.confidence,
                    "label": det.label,
                    "attributes": det.attributes,
                    "lost_frames": 0,
                    "dwell_s": (1.0 / self.fps) if is_inside else 0.0,
                    "in_zone": is_inside,
                    "ts_start": iso_now,
                    "emitted": False,
                }

                if is_inside and self.tracks[t_id]["dwell_s"] >= self.dwell_threshold_s:
                    self.tracks[t_id]["emitted"] = True
                    violations.append(
                        TrackViolation(
                            track_id=t_id,
                            label=det.label,
                            attributes=det.attributes,
                            confidence=det.confidence,
                            dwell_s=round(self.tracks[t_id]["dwell_s"], 2),
                            zone=self.zone_name,
                            ts_start=iso_now,
                            ts_end=iso_now,
                            person_bbox=det.bbox,
                        )
                    )

        return violations
