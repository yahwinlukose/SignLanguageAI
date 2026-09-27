"""
dataset.py

Handles saving landmark data for training.
"""

from __future__ import annotations

import csv
import math
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

        self.csv_path = self.data_dir / "landmarks.csv"

        # Create CSV with a header if it doesn't exist
        if not self.csv_path.exists():
            headers = ["label"]
            for i in range(21):
                headers.extend([f"x{i}", f"y{i}", f"z{i}"])
            
            with open(self.csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

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

        # Validate that exactly 21 landmarks are present
        if len(hand) != 21:
            print(f"Expected 21 landmarks, but got {len(hand)}.")
            return

        # Extract and normalize coordinates relative to the wrist (landmark 0)
        wrist = hand[0]
        relative_landmarks = []
        for landmark in hand:
            relative_landmarks.append((
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z
            ))

        # Scale by the maximum Euclidean distance from the wrist
        scale = max(
            math.sqrt(x*x + y*y + z*z)
            for x, y, z in relative_landmarks
        )

        # Store the 63 normalized features
        features = []
        for x, y, z in relative_landmarks:
            if scale > 0:
                features.extend([x / scale, y / scale, z / scale])
            else:
                features.extend([x, y, z])

        # Append the sample to the CSV file
        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([label] + features)

        print(f"Saved sample: label={label}, features={len(features)}")