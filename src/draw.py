"""
draw.py

Contains functions for drawing hand landmarks.
"""

from __future__ import annotations

import cv2


def draw_landmarks(frame, result) -> None:
    """
    Draw detected hand landmarks on the frame.

    Args:
        frame: OpenCV image.
        result: MediaPipe HandLandmarkerResult.
    """

    if not result.hand_landmarks:
        return

    height, width, _ = frame.shape

    for hand in result.hand_landmarks:
        for landmark in hand:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1,
            )