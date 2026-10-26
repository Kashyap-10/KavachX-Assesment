"""
Video and image encoding utility functions.
"""

import base64
import cv2
import numpy as np


def frame_to_base64_jpeg(frame: np.ndarray, quality: int = 85) -> str:
    """Encodes a BGR OpenCV image frame to a base64 JPEG string."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not success:
        raise ValueError("Failed to encode frame to JPEG format.")
    return base64.b64encode(buffer).decode("utf-8")


def base64_jpeg_to_frame(b64_string: str) -> np.ndarray:
    """Decodes a base64 JPEG string back into an OpenCV BGR numpy array."""
    img_data = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode base64 string to image.")
    return frame
