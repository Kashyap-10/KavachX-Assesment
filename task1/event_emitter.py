"""
Event Emission and JSON Schema Validation.
Constructs and validates structured event JSON objects matching event_schema.json.
"""

import json
import os
import uuid
from typing import Any, Dict, Optional
import jsonschema

from task1.rule_parser import RuleConfig
from task1.tracker import TrackViolation
from utils.video_utils import frame_to_base64_jpeg
import numpy as np


class EventEmitter:
    """Emits schema-compliant event JSON files for verified rule violations."""

    def __init__(
        self,
        schema_path: str = "event_schema.json",
        device_id: str = "kx-edge-001",
        camera_id: str = "cam-03",
        borderline_confidence_threshold: float = 0.65,
    ):
        self.device_id = device_id
        self.camera_id = camera_id
        self.borderline_threshold = borderline_confidence_threshold
        self.schema: Optional[Dict[str, Any]] = None

        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                self.schema = json.load(f)

    def create_event(
        self,
        rule_config: RuleConfig,
        violation: TrackViolation,
        masked_frame: np.ndarray,
    ) -> Dict[str, Any]:
        """Constructs an event dictionary matching event_schema.json."""
        # Convert masked frame to base64 JPEG
        evidence_b64 = frame_to_base64_jpeg(masked_frame, quality=80)

        # needs_verification set to true if detection confidence is borderline (< 0.65)
        needs_verification = violation.confidence < self.borderline_threshold

        event = {
            "event_id": str(uuid.uuid4()),
            "device_id": self.device_id,
            "camera_id": self.camera_id,
            "rule_id": rule_config.rule_id,
            "rule_text": rule_config.rule_text,
            "ts_start": violation.ts_start,
            "ts_end": violation.ts_end,
            "track_id": int(violation.track_id),
            "label": violation.label,
            "attributes": violation.attributes,
            "confidence": round(float(violation.confidence), 3),
            "dwell_s": round(float(violation.dwell_s), 2),
            "zone": violation.zone,
            "evidence_frame": evidence_b64,
            "needs_verification": bool(needs_verification),
        }

        # Validate against schema if available
        if self.schema:
            try:
                jsonschema.validate(instance=event, schema=self.schema)
            except jsonschema.ValidationError as e:
                print(f"[Warning] Generated event schema validation warning: {e.message}")

        return event

    def save_event(
        self,
        event: Dict[str, Any],
        output_dir: str = "task1/outputs",
    ) -> str:
        """Saves event JSON to disk."""
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"event_{event['event_id'][:8]}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(event, f, indent=2)
        return file_path
