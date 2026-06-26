"""
camera.py

Handles webcam initialization and frame capture.
"""

from __future__ import annotations

import cv2
import numpy as np


class Camera:
    """Handles webcam operations."""

    def __init__(self, camera_index: int = 0) -> None:
        """
        Initialize the webcam.

        Args:
            camera_index: Index of the webcam to open.
        """
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")

    def read(self) -> np.ndarray | None:
        """
        Read a single frame from the webcam.

        Returns:
            The captured frame, or None if reading failed.
        """
        success, frame = self.cap.read()

        if not success:
            return None

        return frame

    def release(self) -> None:
        """Release the webcam."""
        self.cap.release()