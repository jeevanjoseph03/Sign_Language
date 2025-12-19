import os
import time
from pathlib import Path

import cv2
import numpy as np

from utils import (
    create_holistic,
    mediapipe_detection,
    extract_keypoints,
    draw_landmarks,
)


# Configuration
ACTIONS = [
    "Hello",
    "How are you",
    "I need help",
    "Thank you",
    "Goodbye",
]
NO_SEQUENCES = 30
SEQUENCE_LENGTH = 30
DATA_PATH = Path("Sign_Language_Data")


def ensure_directories():
    for action in ACTIONS:
        for sequence in range(NO_SEQUENCES):
            seq_dir = DATA_PATH / action / str(sequence)
            os.makedirs(seq_dir, exist_ok=True)


def main():
    ensure_directories()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not accessible: ensure camera permissions and device availability.")

    # Reduce lag by lowering resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    try:
        with create_holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for action in ACTIONS:
                for sequence in range(NO_SEQUENCES):
                    for frame_num in range(SEQUENCE_LENGTH):
                        ret, frame = cap.read()
                        if not ret:
                            raise RuntimeError("Failed to read frame from webcam.")

                        image, results = mediapipe_detection(frame, holistic)
                        draw_landmarks(image, results)

                        # Display text on frame
                        if frame_num == 0:
                            cv2.putText(
                                image,
                                "STARTING COLLECTION",
                                (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                (0, 255, 0),
                                3,
                                cv2.LINE_AA,
                            )
                            cv2.putText(
                                image,
                                f"Action: {action} | Seq: {sequence} | Frame: {frame_num}/{SEQUENCE_LENGTH}",
                                (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 0),
                                2,
                            )
                            cv2.imshow("OmniSign - Data Collection", image)
                            cv2.waitKey(2000)  # Wait 2 seconds for start
                        else:
                            cv2.putText(
                                image,
                                f"Action: {action} | Seq: {sequence} | Frame: {frame_num}/{SEQUENCE_LENGTH}",
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2,
                            )
                            cv2.imshow("OmniSign - Data Collection", image)
                            cv2.waitKey(1)  # Minimal wait for smooth video

                        # Save keypoints
                        keypoints = extract_keypoints(results)
                        save_path = DATA_PATH / action / str(sequence) / f"{frame_num}.npy"
                        np.save(save_path, keypoints)

                        # Check for 'q' key to quit
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            print("Quit key pressed. Exiting...")
                            cap.release()
                            cv2.destroyAllWindows()
                            return
    except KeyboardInterrupt:
        print("\nCollection interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Data collection complete!")


if __name__ == "__main__":
    main()
