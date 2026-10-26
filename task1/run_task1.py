"""
Task 1: Plain-English Rule to Event JSON Pipeline.
End-to-end pipeline:
1. Parse natural language rule
2. Open-vocabulary detection (YOLO-World)
3. Multi-object tracking & polygon dwell time calculation (ByteTrack)
4. Face & head privacy masking
5. JSON event emission compliant with event_schema.json
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Optional
import cv2
import numpy as np

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from task1.detector import OpenVocabDetector
from task1.event_emitter import EventEmitter
from task1.privacy import PrivacyMasker
from task1.rule_parser import RuleParser
from task1.tracker import DwellTracker
from utils.geometry import load_zones


def create_mock_video_if_missing(video_path: str, duration_sec: int = 15, fps: int = 30):
    """Creates a mock surveillance clip for testing if the user hasn't provided one yet."""
    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
    if os.path.exists(video_path):
        return

    print(f"[Notice] Video '{video_path}' not found. Generating a synthetic CCTV surveillance test clip...")
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    total_frames = duration_sec * fps
    # Draw background, warehouse zone, and a moving person carrying a bag inside the zone
    for frame_idx in range(total_frames):
        frame = np.full((h, w, 3), 40, dtype=np.uint8)  # Dark warehouse background

        # Draw loading bay floor marking
        zone_pts = np.array([[100, 150], [550, 150], [580, 450], [80, 450]], np.int32)
        cv2.polylines(frame, [zone_pts], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.putText(frame, "LOADING BAY", (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw a simulated person standing inside loading bay
        # Position: inside zone (x=300, y=250 to 390)
        px1, py1, px2, py2 = 280, 220, 340, 410
        cv2.rectangle(frame, (px1, py1), (px2, py2), (180, 180, 180), -1)  # Body
        cv2.circle(frame, (310, 205), 20, (200, 200, 200), -1)             # Head

        # Draw a carried bag beside the person
        cv2.rectangle(frame, (335, 300), (370, 360), (0, 140, 255), -1)   # Bag
        cv2.putText(frame, f"Frame {frame_idx}/{total_frames}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        out.write(frame)

    out.release()
    print(f"[Notice] Synthetic clip saved to '{video_path}'.")


def run_pipeline(
    rule_text: str,
    video_path: str,
    zones_path: str,
    output_dir: str = "task1/outputs",
    model_name: str = "yolov8s-worldv2.pt",
    max_frames: Optional[int] = None,
):
    print("=" * 70)
    print("KavachX Task 1: Plain-English Rule to Event JSON Pipeline")
    print("=" * 70)
    print(f"Rule: \"{rule_text}\"")

    # Step 1: Parse rule
    parser = RuleParser()
    config = parser.parse(rule_text)
    print(f"[1/5] Rule Parsed Successfully:")
    print(f"      - ID: {config.rule_id}")
    print(f"      - Target: {config.target_class}")
    print(f"      - Attributes: {config.attributes}")
    print(f"      - Zone: {config.zone}")
    print(f"      - Dwell Threshold: {config.dwell_threshold_s}s")

    # Step 2: Load zones
    if not os.path.exists(zones_path):
        raise FileNotFoundError(f"Zones configuration file not found at: {zones_path}")
    zones = load_zones(zones_path)
    if config.zone not in zones:
        available = list(zones.keys())
        print(f"[Warning] Zone '{config.zone}' not in {available}. Falling back to '{available[0]}'")
        zone_polygon = zones[available[0]]
        config.zone = available[0]
    else:
        zone_polygon = zones[config.zone]
    print(f"[2/5] Zone '{config.zone}' loaded ({len(zone_polygon.exterior.coords)} vertices)")

    # Prepare video
    if not os.path.exists(video_path):
        create_mock_video_if_missing(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[3/5] Video Stream: {video_path} (FPS: {fps:.1f}, Frames: {total_video_frames})")

    # Step 3: Initialize components
    detector = OpenVocabDetector(
        target_class=config.target_class,
        attributes=config.attributes,
        model_name=model_name,
        confidence_threshold=0.25,
    )
    tracker = DwellTracker(
        zone_name=config.zone,
        zone_polygon=zone_polygon,
        dwell_threshold_s=config.dwell_threshold_s,
        fps=fps,
    )
    privacy_masker = PrivacyMasker()
    emitter = EventEmitter()

    # Step 4: Process video
    print("[4/5] Running Detection, Tracking & Dwell Time Analysis...")
    emitted_events = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if max_frames and frame_count > max_frames:
            break

        # Detect
        detections = detector.detect(frame)

        # Track & check dwell violations
        violations = tracker.update(detections)

        # Process any violations
        for violation in violations:
            print(f"\n[VIOLATION DETECTED] Track #{violation.track_id} exceeded dwell time: {violation.dwell_s}s in {violation.zone}")

            # Mask faces for privacy
            person_boxes = [d.bbox for d in detections if d.label == "person"] or [violation.person_bbox]
            masked_frame = privacy_masker.mask_evidence_frame(frame, person_boxes)

            # Emit event
            event = emitter.create_event(config, violation, masked_frame)
            event_path = emitter.save_event(event, output_dir)
            emitted_events.append(event)
            print(f"  -> Saved event JSON: {event_path}")

            # Save visual evidence frame image
            frame_img_path = os.path.join(output_dir, f"evidence_{event['event_id'][:8]}.jpg")
            cv2.imwrite(frame_img_path, masked_frame)
            print(f"  -> Saved privacy-masked frame: {frame_img_path}")

        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{total_video_frames} frames...", end="\r")

    cap.release()
    print(f"\n[5/5] Processing complete. Total frames processed: {frame_count}")
    print(f"      Total violations emitted: {len(emitted_events)}")

    # Summary
    if len(emitted_events) > 0:
        sample_path = os.path.join(output_dir, "sample_event.json")
        with open(sample_path, "w", encoding="utf-8") as f:
            import json
            json.dump(emitted_events[0], f, indent=2)
        print(f"      Sample event saved to: {sample_path}")

    return emitted_events


def main():
    parser = argparse.ArgumentParser(description="Task 1: Plain-English Rule to Event JSON Pipeline")
    parser.add_argument(
        "--rule",
        type=str,
        default="Alert me if a person carrying a bag enters the loading bay and stays more than 10 seconds.",
        help="Natural language surveillance rule",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="clips/loading_bay.mp4",
        help="Path to input surveillance video",
    )
    parser.add_argument(
        "--zones",
        type=str,
        default="zones.json",
        help="Path to zones configuration JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="task1/outputs",
        help="Directory to save emitted JSON events and masked frames",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s-worldv2.pt",
        help="YOLO-World model checkpoint name",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit on number of frames to process",
    )

    args = parser.parse_args()
    run_pipeline(
        rule_text=args.rule,
        video_path=args.video,
        zones_path=args.zones,
        output_dir=args.output_dir,
        model_name=args.model,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
