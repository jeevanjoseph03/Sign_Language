import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from collections import deque
from pathlib import Path
from utils import mediapipe_detection, extract_keypoints, draw_landmarks, create_holistic

# Configuration
SEQUENCE_LENGTH = 30
KEYPOINT_DIM = 258
CONFIDENCE_THRESHOLD = 0.7

# Load model and labels
try:
    model = load_model('sign_language_model.h5')
    with open('action_labels.pkl', 'rb') as f:
        ACTIONS = pickle.load(f)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Make sure you ran: python .\\train_model.py")
    exit(1)

print("=" * 60)
print("REAL-TIME SIGN LANGUAGE RECOGNITION")
print("=" * 60)
print(f"Model loaded! Recognizing: {', '.join(ACTIONS)}")
print("Press 'Q' to quit")
print("=" * 60)

# Initialize
sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
prediction_text = ""
confidence_text = ""

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not found!")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try:
    with create_holistic() as holistic:
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame")
                    break

                image, results = mediapipe_detection(frame, holistic)
                draw_landmarks(image, results)

                # Extract keypoints
                keypoints = extract_keypoints(results)
                sequence_buffer.append(keypoints)

                # Make prediction when buffer is full
                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    X = np.array([list(sequence_buffer)])
                    predictions = model.predict(X, verbose=0)[0]
                    action_idx = np.argmax(predictions)
                    confidence = predictions[action_idx]

                    if confidence > CONFIDENCE_THRESHOLD:
                        prediction_text = ACTIONS[action_idx]
                        confidence_text = f"{confidence * 100:.1f}%"
                    else:
                        prediction_text = "Uncertain"
                        confidence_text = f"{confidence * 100:.1f}%"

                # Display prediction
                h, w = image.shape[:2]
                cv2.rectangle(image, (10, 30), (400, 120), (0, 0, 0), -1)
                cv2.putText(
                    image,
                    "PREDICTION",
                    (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    image,
                    prediction_text,
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3,
                )
                cv2.putText(
                    image,
                    f"Confidence: {confidence_text}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (200, 200, 0),
                    2,
                )

                # Display buffer status
                cv2.putText(
                    image,
                    f"Buffer: {len(sequence_buffer)}/{SEQUENCE_LENGTH}",
                    (w - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (100, 100, 255),
                    1,
                )

                cv2.imshow("Sign Language Recognition", image)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # Q or ESC
                    break

            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break

except KeyboardInterrupt:
    print("\nClosing...")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Closed.")
