import os
# Suppress heavy TensorFlow startup logging to keep terminal clean
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import csv
import cv2
import time
import json
import math
import numpy as np
import collections

# Load TensorFlow gracefully if available
try:
    import tensorflow as tf
except ImportError:
    tf = None

from camera import Camera
from hand_tracker import HandTracker
from draw import draw_landmarks
from dataset import DatasetCollector


def main() -> None:
    """Main application entry point."""

    # Create objects
    camera = Camera()
    tracker = HandTracker()
    collector = DatasetCollector()

    current_label = "A"

    # Application Mode: PREDICTING or COLLECTING
    mode = "PREDICTING"
    last_collect_time = 0.0

    # Load Model & Labels for Prediction
    model = None
    labels_dict = {}
    
    model_path = "models/sign_language_model.keras"
    labels_path = "models/labels.json"
    
    if tf is not None and os.path.exists(model_path) and os.path.exists(labels_path):
        print("Loading trained sign language model...")
        model = tf.keras.models.load_model(model_path)
        with open(labels_path, "r") as f:
            labels_dict = json.load(f)
        print("Model loaded successfully. Entering PREDICTING mode.")
    else:
        print("Warning: Model, labels, or TensorFlow not found. Predictions disabled.")

    # Prediction smoothing: track last 5 frames
    prediction_history = collections.deque(maxlen=5)

    # Initialize sample counts for dataset collection mode
    counts = {chr(i): 0 for i in range(ord('A'), ord('Z') + 1)}
    csv_path = "data/landmarks.csv"
    if os.path.exists(csv_path):
        try:
            with open(csv_path, mode="r") as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    if row and row[0] in counts:
                        counts[row[0]] += 1
        except Exception as e:
            print(f"Could not load existing counts: {e}")

    while True:
        # Read a frame from the webcam
        frame = camera.read()

        if frame is None:
            break

        # Mirror the image for intuitive interaction
        frame = cv2.flip(frame, 1)

        # Detect hands using MediaPipe
        result = tracker.detect(frame)

        # Draw hand landmarks safely
        draw_landmarks(frame, result)

        display_text = []

        if mode == "COLLECTING":
            # --- Dataset Collection Mode ---
            current_time = time.monotonic()
            # Enforce 200ms throttle for approx 5 samples/sec
            if current_time - last_collect_time >= 0.2:
                collector.save(result, current_label)
                # Verify an actual hand was stored
                if result.hand_landmarks and len(result.hand_landmarks[0]) == 21:
                    counts[current_label] += 1
                last_collect_time = current_time

            display_text.append("Mode: COLLECTING")
            display_text.append(f"Label: {current_label}")
            display_text.append(f"Samples: {counts[current_label]}")

        elif mode == "PREDICTING":
            # --- Live Inference Mode ---
            display_text.append("Mode: PREDICTING")
            
            pred_text = "Unknown"
            conf_text = ""
            
            if model and result.hand_landmarks:
                hand = result.hand_landmarks[0]
                if len(hand) == 21:
                    # 1. EXACT SAME Normalization Strategy (translate wrist to origin)
                    wrist = hand[0]
                    relative_landmarks = []
                    for landmark in hand:
                        relative_landmarks.append((
                            landmark.x - wrist.x,
                            landmark.y - wrist.y,
                            landmark.z - wrist.z
                        ))

                    # 2. Scale by the maximum Euclidean distance from the wrist
                    scale = max(
                        math.sqrt(x*x + y*y + z*z)
                        for x, y, z in relative_landmarks
                    )

                    # 3. Store the 63 normalized feature floats
                    features = []
                    for x, y, z in relative_landmarks:
                        if scale > 0:
                            features.extend([x / scale, y / scale, z / scale])
                        else:
                            features.extend([x, y, z])
                            
                    # 4. Cast to NumPy array & reshape exactly to (1, 63)
                    X = np.array(features, dtype=np.float32).reshape(1, 63)
                    
                    # 5. Model Inference (silencing the progress bar log spam)
                    pred_probs = model.predict(X, verbose=0)[0]
                    pred_idx = np.argmax(pred_probs)
                    confidence = pred_probs[pred_idx]
                    
                    # 6. Apply 70% Confidence Threshold
                    if confidence >= 0.70:
                        raw_label = labels_dict.get(str(pred_idx), "Unknown")
                        prediction_history.append((raw_label, confidence))
                    else:
                        prediction_history.append(("Unknown", confidence))
                else:
                    prediction_history.append(("Unknown", 0.0))
            else:
                # Hand is missing from frame entirely
                prediction_history.append(("Unknown", 0.0))
                
            # 7. Smoothing Mechanism: evaluate the majority vote of the last 5 frames
            if prediction_history:
                # Extract just the label strings for the vote
                labels_only = [item[0] for item in prediction_history]
                majority_label = collections.Counter(labels_only).most_common(1)[0][0]
                
                if majority_label != "Unknown":
                    pred_text = majority_label
                    # Find the most recent confidence score belonging to this winning label
                    for label, conf in reversed(prediction_history):
                        if label == majority_label:
                            conf_text = f"{conf * 100:.1f}%"
                            break
            
            display_text.append(f"Prediction: {pred_text}")
            if pred_text != "Unknown":
                display_text.append(f"Confidence: {conf_text}")

        # Render UI Overlay text stack cleanly
        y_offset = 30
        for text in display_text:
            cv2.putText(
                frame, 
                text, 
                (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                (0, 255, 0), 
                2, 
                cv2.LINE_AA
            )
            y_offset += 40

        # Display the frame to screen
        cv2.imshow("Camera", frame)

        # Poll keyboard input (ONLY ONCE per loop)
        key = cv2.waitKey(1) & 0xFF

        if key != 255:
            # SPACE toggles cleanly between data collection and inference modes
            if key == ord(" "):
                if mode == "PREDICTING":
                    mode = "COLLECTING"
                    print("Switched to COLLECTING mode")
                else:
                    mode = "PREDICTING"
                    print("Switched to PREDICTING mode")
                    # Clear smoothing queue when returning to predict to avoid stale data
                    prediction_history.clear()
            
            # Q triggers quit
            elif key == ord("q") or key == ord("Q"):
                break
            
            # S enables manual save override (can be done in either mode technically)
            elif key == ord("s") or key == ord("S"):
                collector.save(result, current_label)
                if result.hand_landmarks and len(result.hand_landmarks[0]) == 21:
                    counts[current_label] += 1
                    
            # A-Z controls target collection label
            elif ord("a") <= key <= ord("z"):
                current_label = chr(key).upper()
                print(f"Selected target label: {current_label}")
            elif ord("A") <= key <= ord("Z"):
                current_label = chr(key).upper()
                print(f"Selected target label: {current_label}")

    # Memory cleanup
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()