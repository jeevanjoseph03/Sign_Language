
import cv2
import numpy as np
import random
import time

# Try to import MediaPipe, but provide fallback if it fails
try:
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    HAS_MEDIAPIPE = True
except (ImportError, AttributeError) as e:
    print(f"Warning: MediaPipe could not be imported: {e}")
    mp_holistic = None
    mp_drawing = None
    HAS_MEDIAPIPE = False

# --- MOCK CLASSES FOR FALLBACK ---
class MockLandmark:
    def __init__(self, x, y, z=0.0, visibility=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

class MockLandmarkList:
    def __init__(self, num_points):
        self.landmark = [MockLandmark(random.random(), random.random()) for _ in range(num_points)]

class MockResults:
    def __init__(self):
        # Simulate occasional detection
        if random.random() > 0.3:
            self.pose_landmarks = MockLandmarkList(33)
            self.face_landmarks = MockLandmarkList(468)
            self.left_hand_landmarks = MockLandmarkList(21) if random.random() > 0.5 else None
            self.right_hand_landmarks = MockLandmarkList(21) if random.random() > 0.5 else None
        else:
            self.pose_landmarks = None
            self.face_landmarks = None
            self.left_hand_landmarks = None
            self.right_hand_landmarks = None

class MockHolistic:
    """Simulates the MediaPipe Holistic model interface."""
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def process(self, image):
        time.sleep(0.05) # Simulate processing time
        return MockResults()

    # Allow direct instantiation for 'with' block compatibility if needed
    def Holistic(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        return self

# Use MockHolistic if real one is missing
if not mp_holistic:
    mp_holistic = MockHolistic()

def mediapipe_detection(image, model):
    """
    Process image with MediaPipe Holistic model (or Mock).
    """
    if model is None:
        return image, None
        
    # Check if it's the real MediaPipe model or our Mock
    if hasattr(model, 'process'):
         # If it's real mediapipe, needs RGB
        if HAS_MEDIAPIPE and not isinstance(model, MockHolistic):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = model.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image, results
        else:
            # Mock model
            results = model.process(image)
            return image, results
            
    return image, None

def extract_keypoints(results):
    """
    Extracts keypoints or returns zeros if results are None/Mock.
    """
    if not results:
         return np.zeros(1662)

    # 1. Pose Landmarks (33 * 4 = 132)
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 4)
        
    # 2. Face Landmarks (468 * 3 = 1404)
    if results.face_landmarks:
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
    else:
        face = np.zeros(468 * 3)
        
    # 3. Left Hand Landmarks (21 * 3 = 63)
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3)

    # 4. Right Hand Landmarks (21 * 3 = 63)
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)
        
    return np.concatenate([pose, face, lh, rh])

def draw_styled_landmarks(image, results):
    """
    Draws styled landmarks on the image. Handles missing MP drawing utils.
    """
    if results is None:
        return

    # If we have real drawing utils, use them
    if HAS_MEDIAPIPE and mp_drawing and mp_drawing_styles:
        try:
            # Draw face connections
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    image, 
                    results.face_landmarks, 
                    mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                )
            # Draw pose
            if results.pose_landmarks:
                 mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                )
            # Draw hands
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            return
        except Exception:
            pass # Fallback to manual drawing

    # Manual fallback drawing (for Mock or mismatched MP versions)
    h, w, c = image.shape
    
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
            
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

