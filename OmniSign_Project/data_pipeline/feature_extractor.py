"""
Data Pipeline: Feature Extraction using MediaPipe Holistic

Extracts comprehensive landmark data from video frames:
- Hand landmarks (21 keypoints per hand)
- Facial landmarks (468 keypoints)
- Body landmarks (33 keypoints)
- Pose and orientation information
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List

try:
    import mediapipe as mp
    from mediapipe.solutions import holistic, drawing_utils
except ImportError:
    print("Warning: MediaPipe not available. Feature extraction will use dummy data.")
    mp = None


class MediaPipeFeatureExtractor:
    """
    Extract hand, face, and body landmarks from video frames using MediaPipe Holistic.
    """
    
    def __init__(self):
        """Initialize MediaPipe Holistic."""
        if mp is None:
            self.holistic = None
            self.mp_drawing = None
            print("MediaPipe not available - using dummy feature extraction")
            return
            
        self.mp_holistic = holistic
        self.mp_drawing = drawing_utils
        
        # Create holistic detector
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,  # 0 = lite, 1 = full
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def get_dummy_landmarks(self) -> Dict[str, np.ndarray]:
        """Return dummy landmarks when MediaPipe is not available."""
        return {
            'hands': np.zeros((42, 4)),
            'hand_confidence': 0.0,
            'face': np.zeros((100, 4)),
            'face_confidence': 0.0,
            'pose': np.zeros((33, 4)),
            'pose_confidence': 0.0
        }
    
    def extract_landmarks(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all landmarks from a single frame.
        
        Args:
            frame (np.ndarray): Input image frame (RGB format)
            
        Returns:
            Dict with keys:
                - 'hands': (42, 4) - 21 pts × 2 hands × 4 values (x, y, z, confidence)
                - 'face': (~100, 4) - Selected facial landmarks
                - 'pose': (33, 4) - Body/pose landmarks
                - 'hand_confidence': Hand detection confidence
                - 'face_confidence': Face detection confidence
                - 'pose_confidence': Pose detection confidence
        """
        
        # Return dummy data if MediaPipe not available
        if self.holistic is None:
            return self.get_dummy_landmarks()
        
        # RGB format required
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        
        landmarks = {}
        
        # ==================== HAND LANDMARKS ====================
        hand_landmarks = np.zeros((42, 4))  # 21 pts × 2 hands × 4 values
        hand_confidence = 0.0
        
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark):
                hand_landmarks[i] = [lm.x, lm.y, lm.z, lm.confidence]
            hand_confidence = 0.5
        
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark):
                hand_landmarks[21 + i] = [lm.x, lm.y, lm.z, lm.confidence]
            hand_confidence = max(hand_confidence, 0.5)
        
        landmarks['hands'] = hand_landmarks
        landmarks['hand_confidence'] = hand_confidence
        
        # ==================== FACIAL LANDMARKS ====================
        # Select key facial landmarks (eyes, nose, mouth, jaw)
        # Full face has 468 landmarks; we'll use ~100 key ones
        face_landmarks = np.zeros((100, 4))  # Will pad with zeros
        face_confidence = 0.0
        
        if results.face_landmarks:
            # Key facial landmarks indices
            key_indices = [
                *range(0, 10),      # Silhouette start
                *range(17, 48),     # Eyebrows and eyes
                *range(48, 68),     # Mouth
                *range(68, 73),     # Nose
                *range(73, 82),     # Additional jaw points
            ]
            
            for idx, i in enumerate(key_indices[:100]):
                if i < len(results.face_landmarks.landmark):
                    lm = results.face_landmarks.landmark[i]
                    face_landmarks[idx] = [lm.x, lm.y, lm.z, lm.confidence]
            
            face_confidence = 0.5
        
        landmarks['face'] = face_landmarks
        landmarks['face_confidence'] = face_confidence
        
        # ==================== BODY/POSE LANDMARKS ====================
        # 33 body keypoints (full body pose)
        body_landmarks = np.zeros((33, 4))
        body_confidence = 0.0
        
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                body_landmarks[i] = [lm.x, lm.y, lm.z, lm.confidence]
            body_confidence = 0.5
        
        landmarks['pose'] = body_landmarks
        landmarks['pose_confidence'] = body_confidence
        
        return landmarks
    
    def concatenate_landmarks(self, landmarks: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Concatenate all landmarks into a single feature vector.
        
        Args:
            landmarks (Dict): Output from extract_landmarks()
            
        Returns:
            np.ndarray: Flattened feature vector of shape (500,)
                - Hands: 42 × 4 = 168 dims
                - Face: 100 × 4 = 400 dims
                - Pose: 33 × 4 = 132 dims
                - Total: 700 dims (will trim to 500)
        """
        
        # Flatten each component
        hands_flat = landmarks['hands'].flatten()  # 168 dims
        face_flat = landmarks['face'].flatten()    # 400 dims
        pose_flat = landmarks['pose'].flatten()    # 132 dims
        
        # Concatenate
        feature_vector = np.concatenate([hands_flat, face_flat[:332]])  # 500 dims total
        
        return feature_vector
    
    def extract_sequence(self, video_path: str, num_frames: int = 30) -> np.ndarray:
        """
        Extract landmark sequence from a video file.
        
        Args:
            video_path (str): Path to video file
            num_frames (int): Number of frames to extract
            
        Returns:
            np.ndarray: Shape (num_frames, 500) - sequence of feature vectors
        """
        
        cap = cv2.VideoCapture(video_path)
        sequence = []
        
        while len(sequence) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract landmarks
            landmarks = self.extract_landmarks(frame)
            feature_vector = self.concatenate_landmarks(landmarks)
            sequence.append(feature_vector)
        
        cap.release()
        
        # Pad with zeros if not enough frames
        while len(sequence) < num_frames:
            sequence.append(np.zeros(500))
        
        return np.array(sequence[:num_frames])
    
    def extract_from_webcam(self, duration_seconds: int = 5, num_frames: int = 30) -> np.ndarray:
        """
        Capture landmarks from webcam for a specified duration.
        
        Args:
            duration_seconds (int): Duration to capture in seconds
            num_frames (int): Target number of frames
            
        Returns:
            np.ndarray: Shape (num_frames, 500) - sequence of feature vectors
        """
        
        cap = cv2.VideoCapture(0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        target_frame_count = int(duration_seconds * fps)
        
        sequence = []
        frame_count = 0
        
        while frame_count < target_frame_count and len(sequence) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for selfie view
            frame = cv2.flip(frame, 1)
            
            # Extract landmarks
            landmarks = self.extract_landmarks(frame)
            feature_vector = self.concatenate_landmarks(landmarks)
            sequence.append(feature_vector)
            
            # Display frame with landmarks
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(frame_rgb)
            
            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                frame, 
                results.face_landmarks, 
                self.mp_holistic.FACEMESH_TESSELATION
            )
            self.mp_drawing.draw_landmarks(
                frame, 
                results.left_hand_landmarks, 
                self.mp_holistic.HAND_CONNECTIONS
            )
            self.mp_drawing.draw_landmarks(
                frame, 
                results.right_hand_landmarks, 
                self.mp_holistic.HAND_CONNECTIONS
            )
            self.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_holistic.POSE_CONNECTIONS
            )
            
            # Display
            cv2.putText(frame, f"Frames: {len(sequence)}/{num_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Capture Landmarks', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Pad with zeros if not enough frames
        while len(sequence) < num_frames:
            sequence.append(np.zeros(500))
        
        return np.array(sequence[:num_frames])
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks to range [0, 1].
        
        Args:
            landmarks (np.ndarray): Landmark data
            
        Returns:
            np.ndarray: Normalized landmarks
        """
        min_val = np.min(landmarks)
        max_val = np.max(landmarks)
        
        if max_val - min_val > 0:
            normalized = (landmarks - min_val) / (max_val - min_val)
        else:
            normalized = landmarks
        
        return normalized


# Example usage
if __name__ == "__main__":
    print("MediaPipe Feature Extractor Test")
    print("=" * 50)
    
    extractor = MediaPipeFeatureExtractor()
    
    # Test webcam capture
    print("Starting webcam capture (5 seconds)...")
    print("Press 'q' to stop early")
    sequence = extractor.extract_from_webcam(duration_seconds=5, num_frames=30)
    
    print(f"Sequence shape: {sequence.shape}")
    print(f"Sequence min: {np.min(sequence):.4f}")
    print(f"Sequence max: {np.max(sequence):.4f}")
    print(f"Sequence mean: {np.mean(sequence):.4f}")
    
    # Normalize
    normalized = extractor.normalize_landmarks(sequence)
    print(f"\nAfter normalization:")
    print(f"Normalized min: {np.min(normalized):.4f}")
    print(f"Normalized max: {np.max(normalized):.4f}")
    print(f"Normalized mean: {np.mean(normalized):.4f}")
