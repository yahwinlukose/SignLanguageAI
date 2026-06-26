import cv2

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

    while True:
        # Read a frame from the webcam
        frame = camera.read()

        if frame is None:
            break

        # Mirror the image
        frame = cv2.flip(frame, 1)

        # Detect hands
        result = tracker.detect(frame)

        # Draw landmarks
        draw_landmarks(frame, result)

        # Display the frame
        cv2.imshow("Camera", frame)

        # Read keyboard input (ONLY ONCE)
        key = cv2.waitKey(1) & 0xFF

        # Save one sample
        if key == ord("s"):
            collector.save(result, "A")

        # Quit
        elif key == ord("q"):
            break

    # Clean up
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()