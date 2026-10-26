"""
Task 2: Motion Gating Simulation for Edge Budget Optimization.
Runs detector execution behind a motion gate with periodic full-frame passes.
Measures detector calls per second with vs. without motion gating.
"""

import argparse
import os
import sys
import time
from typing import Optional, Tuple
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def create_static_heavy_video_if_missing(video_path: str, duration_sec: int = 20, fps: int = 30):
    """Generates synthetic surveillance footage with mostly static background and occasional motion."""
    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
    if os.path.exists(video_path):
        return

    print(f"[Notice] Generating synthetic static-heavy footage: '{video_path}'...")
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    total_frames = duration_sec * fps
    for f in range(total_frames):
        frame = np.full((h, w, 3), 50, dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (540, 380), (70, 70, 70), -1)
        cv2.putText(frame, "CCTV 03 - WAREHOUSE AISLE", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if (120 <= f <= 180) or (360 <= f <= 420):
            x_pos = 150 + int((f % 120) * 3)
            cv2.rectangle(frame, (x_pos, 200), (x_pos + 40, 320), (0, 200, 255), -1)

        out.write(frame)

    out.release()
    print(f"[Notice] Saved '{video_path}'.")


class MotionGate:
    """Detects movement using background subtraction and triggers model execution."""

    def __init__(
        self,
        diff_threshold: int = 25,
        min_motion_ratio: float = 0.004,
        full_frame_interval: int = 30,
    ):
        self.diff_threshold = diff_threshold
        self.min_motion_ratio = min_motion_ratio
        self.full_frame_interval = full_frame_interval
        self.prev_gray: Optional[np.ndarray] = None
        self.frame_idx = 0

    def should_trigger(self, frame: np.ndarray) -> tuple[bool, str]:
        """Determines whether the detector should run on the given frame.
        
        Returns:
            (should_run, reason_string)
        """
        self.frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.frame_idx % self.full_frame_interval == 0:
            self.prev_gray = gray
            return True, "periodic_heartbeat"

        if self.prev_gray is None:
            self.prev_gray = gray
            return True, "initial_frame"

        delta = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(delta, self.diff_threshold, 255, cv2.THRESH_BINARY)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        motion_pixels = cv2.countNonZero(thresh)
        total_pixels = frame.shape[0] * frame.shape[1]
        motion_ratio = motion_pixels / float(total_pixels)

        self.prev_gray = gray

        if motion_ratio >= self.min_motion_ratio:
            return True, f"motion_detected ({motion_ratio*100:.2f}%)"

        return False, "static_scene"


def run_motion_gating_simulation(
    video_path: str = "clips/static_heavy.mp4",
    model_path: Optional[str] = "task2/outputs/model_int8.onnx",
    full_frame_interval: int = 30,
):
    print("=" * 70)
    print("Task 2: Edge Budget Simulation — Motion Gating on CCTV Footage")
    print("=" * 70)

    if not os.path.exists(video_path):
        create_static_heavy_video_if_missing(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps
    print(f"Video: {video_path} | {total_frames} frames | {duration_s:.1f}s duration | FPS: {fps:.1f}")

    session = None
    input_name = None
    if model_path and os.path.exists(model_path):
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            print(f"Loaded INT8 ONNX detector: {model_path}")
        except Exception as e:
            print(f"[Notice] ONNX session init note ({e}). Simulating inference.")

    gater = MotionGate(full_frame_interval=full_frame_interval)

    detector_calls = 0
    heartbeat_calls = 0
    motion_calls = 0
    processed_frames = 0

    t0 = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frames += 1

        trigger, reason = gater.should_trigger(frame)

        if trigger:
            detector_calls += 1
            if "periodic" in reason:
                heartbeat_calls += 1
            else:
                motion_calls += 1

            if session and input_name:
                inp = cv2.resize(frame, (416, 416)).transpose(2, 0, 1).astype(np.float32) / 255.0
                inp = np.expand_dims(inp, axis=0)
                _ = session.run(None, {input_name: inp})

    elapsed = time.perf_counter() - t0
    cap.release()

    calls_per_sec_no_gate = processed_frames / duration_s
    calls_per_sec_with_gate = detector_calls / duration_s
    compute_savings_pct = (1.0 - (detector_calls / processed_frames)) * 100.0

    print("\n" + "=" * 70)
    print("Motion Gating Benchmark Results:")
    print(f"  • Total Video Duration        : {duration_s:.1f} seconds ({processed_frames} frames)")
    print(f"  • Calls WITHOUT Motion Gating : {calls_per_sec_no_gate:.2f} calls/sec ({processed_frames} total)")
    print(f"  • Calls WITH Motion Gating    : {calls_per_sec_with_gate:.2f} calls/sec ({detector_calls} total)")
    print(f"      - Motion-triggered        : {motion_calls}")
    print(f"      - Heartbeat passes        : {heartbeat_calls}")
    print(f"  • Edge Compute Reduction      : {compute_savings_pct:.1f}% savings!")
    print("=" * 70)

    return {
        "calls_per_sec_no_gate": calls_per_sec_no_gate,
        "calls_per_sec_with_gate": calls_per_sec_with_gate,
        "compute_savings_pct": compute_savings_pct,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motion Gating Simulation")
    parser.add_argument("--video", type=str, default="clips/static_heavy.mp4")
    parser.add_argument("--model", type=str, default="task2/outputs/model_int8.onnx")
    parser.add_argument("--interval", type=int, default=30, help="Full-frame heartbeat interval in frames")

    args = parser.parse_args()
    run_motion_gating_simulation(
        video_path=args.video,
        model_path=args.model,
        full_frame_interval=args.interval,
    )
