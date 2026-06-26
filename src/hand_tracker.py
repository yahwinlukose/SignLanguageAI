from __future__ import annotations

import cv2
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision


class HandTracker:
    """Detects hand landmarks using MediaPipe."""

    def __init__(self) -> None:
        """Initialize the MediaPipe Hand Landmarker."""

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path="models/hand_landmarker.task"
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame):
        """
        Detect hands in an OpenCV frame.

        Args:
            frame: OpenCV image (BGR)

        Returns:
            MediaPipe detection result.
        """

        # Convert BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        # Detect hands
        return self.detector.detect(mp_image)