"""
dataset.py

Handles saving landmark data for training.
"""

from __future__ import annotations

from pathlib import Path


class DatasetCollector:
    """Collects hand landmark data and stores it as CSV files."""

    def __init__(self, data_dir: str = "data") -> None:
        """
        Initialize the dataset collector.

        Args:
            data_dir: Directory where dataset files are stored.
        """
        self.data_dir = Path(data_dir)

        # Create the data directory if it doesn't exist
        self.data_dir.mkdir(exist_ok=True)

    def save(self, result, label: str) -> None:
        """
        Save one detected hand sample.

        Args:
            result: MediaPipe HandLandmarkerResult.
            label: Gesture label (e.g. "A").
        """

        # No hand detected
        if not result.hand_landmarks:
            print("No hand detected.")
            return

        # Take the first detected hand
        hand = result.hand_landmarks[0]

        # Store the 63 features
        features = []

        # Extract x, y, z coordinates
        for landmark in hand:
            features.append(landmark.x)
            features.append(landmark.y)
            features.append(landmark.z)

        print(f"Extracted {len(features)} features.")
        print(features)