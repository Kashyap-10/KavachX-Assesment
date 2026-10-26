"""
Privacy Protection and Face Masking Engine.
Applies Gaussian blurring to face and head regions on evidence frames to preserve privacy.
"""

from typing import List, Optional
import cv2
import numpy as np


class PrivacyMasker:
    """Anonymizes evidence frames by detecting and blurring faces and head regions."""

    def __init__(self, blur_kernel_size: int = 51, blur_sigma: float = 30.0):
        self.ksize = (blur_kernel_size, blur_kernel_size) if blur_kernel_size % 2 == 1 else (blur_kernel_size + 1, blur_kernel_size + 1)
        self.sigma = blur_sigma

        # Load OpenCV Haar cascades for face detection
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def mask_evidence_frame(
        self,
        frame: np.ndarray,
        person_bboxes: Optional[List[List[float]]] = None,
    ) -> np.ndarray:
        """Applies privacy blur to faces and upper head regions.
        
        Args:
            frame: BGR numpy image.
            person_bboxes: Optional list of person bounding boxes [x1, y1, x2, y2].
        
        Returns:
            Privacy-masked copy of frame.
        """
        masked = frame.copy()
        h, w = masked.shape[:2]

        # Tier 1: Direct face detection via Haar Cascade
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(20, 20),
        )

        for (fx, fy, fw, fh) in faces:
            fx1, fy1 = max(0, fx), max(0, fy)
            fx2, fy2 = min(w, fx + fw), min(h, fy + fh)
            if fx2 > fx1 and fy2 > fy1:
                roi = masked[fy1:fy2, fx1:fx2]
                masked[fy1:fy2, fx1:fx2] = cv2.GaussianBlur(roi, self.ksize, self.sigma)

        # Tier 2: Head region masking for all detected persons (CCTV failsafe)
        if person_bboxes:
            for bbox in person_bboxes:
                bx1, by1, bx2, by2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(w, bx2), min(h, by2)

                # Head occupies roughly the top 28% of the person's bounding box
                head_height = int((by2 - by1) * 0.28)
                hy2 = min(by2, by1 + head_height)

                if bx2 > bx1 and hy2 > by1:
                    roi = masked[by1:hy2, bx1:bx2]
                    masked[by1:hy2, bx1:bx2] = cv2.GaussianBlur(roi, self.ksize, self.sigma)

        return masked
