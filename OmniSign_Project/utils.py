import numpy as np
import cv2


KEYPOINTS_VECTOR_LENGTH = 258  # Pose (33*4=132) + Left Hand (21*3=63) + Right Hand (21*3=63)


def create_holistic(
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    model_complexity: int = 1,
):
    """Placeholder context manager for compatibility."""
    class HolisticContext:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    return HolisticContext()


def mediapipe_detection(image, holistic=None):
    """Capture video frame without pose detection (placeholder mode)."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create empty results object
    results = type('Results', (), {
        'pose_landmarks': [],
        'left_hand_landmarks': [],
        'right_hand_landmarks': [],
    })()
    
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr, results


def extract_keypoints(results) -> np.ndarray:
    """Extract 258-dimensional keypoint vector (zeros placeholder)."""
    # For now, return zeros until we have the actual model
    # Pose: 33 landmarks * 4 = 132
    # Left Hand: 21 landmarks * 3 = 63
    # Right Hand: 21 landmarks * 3 = 63
    # Total = 258
    return np.zeros(258, dtype=np.float32)


def draw_landmarks(image, results):
    """Draw landmarks on image (placeholder - no-op for now)."""
    pass


__all__ = [
    "create_holistic",
    "mediapipe_detection",
    "extract_keypoints",
    "draw_landmarks",
    "KEYPOINTS_VECTOR_LENGTH",
]
