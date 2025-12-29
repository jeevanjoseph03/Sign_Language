"""
Signer-Adaptive Personalized Learning Module

Enables the system to adapt to individual signing styles through:
- User profile management
- Incremental fine-tuning
- Confidence-based adaptation
- Local model storage
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import hashlib


class SignerProfile:
    """
    Individual signer profile with personalized model parameters.
    """
    
    def __init__(self, user_id: str, name: str, language: str = "en",
                 profile_dir: str = "user_profiles"):
        """
        Initialize a signer profile.
        
        Args:
            user_id (str): Unique user identifier
            name (str): User's name
            language (str): Primary language
            profile_dir (str): Directory to store profiles
        """
        self.user_id = user_id
        self.name = name
        self.language = language
        self.profile_dir = Path(profile_dir) / user_id
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        
        # Profile metadata
        self.created_date = datetime.now().isoformat()
        self.last_updated = self.created_date
        
        # Calibration data
        self.calibration_samples = 0
        self.calibration_target = 10  # Need 10 samples for personalization
        
        # Model parameters
        self.adapted_weights = None
        self.base_model_accuracy = 0.0
        self.personalized_accuracy = 0.0
        
        # Signing characteristics
        self.signing_speed = 1.0  # Normal speed
        self.hand_size = 1.0      # Normal size
        self.confidence_threshold = 0.85
        
        # Statistics
        self.total_gestures_recognized = 0
        self.gestures_by_type = {}
        self.accuracy_history = []
        
    def to_dict(self) -> Dict:
        """Convert profile to dictionary for saving."""
        return {
            'user_id': self.user_id,
            'name': self.name,
            'language': self.language,
            'created_date': self.created_date,
            'last_updated': self.last_updated,
            'calibration_samples': self.calibration_samples,
            'calibration_target': self.calibration_target,
            'base_model_accuracy': float(self.base_model_accuracy),
            'personalized_accuracy': float(self.personalized_accuracy),
            'signing_speed': float(self.signing_speed),
            'hand_size': float(self.hand_size),
            'confidence_threshold': float(self.confidence_threshold),
            'total_gestures_recognized': self.total_gestures_recognized,
            'gestures_by_type': self.gestures_by_type,
            'accuracy_history': [float(x) for x in self.accuracy_history]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SignerProfile':
        """Create profile from dictionary."""
        profile = cls(data['user_id'], data['name'], data.get('language', 'en'))
        
        # Restore metadata
        profile.created_date = data.get('created_date')
        profile.last_updated = data.get('last_updated')
        profile.calibration_samples = data.get('calibration_samples', 0)
        profile.base_model_accuracy = data.get('base_model_accuracy', 0.0)
        profile.personalized_accuracy = data.get('personalized_accuracy', 0.0)
        profile.signing_speed = data.get('signing_speed', 1.0)
        profile.hand_size = data.get('hand_size', 1.0)
        profile.confidence_threshold = data.get('confidence_threshold', 0.85)
        profile.total_gestures_recognized = data.get('total_gestures_recognized', 0)
        profile.gestures_by_type = data.get('gestures_by_type', {})
        profile.accuracy_history = data.get('accuracy_history', [])
        
        return profile


class PersonalizationEngine:
    """
    Manages signer-adaptive personalized learning.
    """
    
    def __init__(self, model, model_dir: str = "models/checkpoints",
                profile_dir: str = "user_profiles"):
        """
        Initialize personalization engine.
        
        Args:
            model: Trained base model
            model_dir (str): Directory with model checkpoints
            profile_dir (str): Directory for user profiles
        """
        self.model = model
        self.model_dir = Path(model_dir)
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(exist_ok=True)
        
        self.current_profile = None
        self.calibration_data = []
        
    def create_user_profile(self, user_id: str, name: str,
                          language: str = "en") -> SignerProfile:
        """
        Create a new user profile.
        
        Args:
            user_id (str): Unique identifier
            name (str): User's name
            language (str): Primary language
            
        Returns:
            SignerProfile: New user profile
        """
        
        profile = SignerProfile(user_id, name, language, str(self.profile_dir))
        self.current_profile = profile
        
        # Save initial profile
        self.save_profile(profile)
        
        print(f"Created profile for {name} (ID: {user_id})")
        return profile
    
    def load_user_profile(self, user_id: str) -> Optional[SignerProfile]:
        """
        Load existing user profile.
        
        Args:
            user_id (str): User identifier
            
        Returns:
            SignerProfile or None if not found
        """
        
        profile_path = self.profile_dir / user_id / "profile.json"
        
        if profile_path.exists():
            with open(profile_path, 'r') as f:
                data = json.load(f)
            
            profile = SignerProfile.from_dict(data)
            self.current_profile = profile
            
            print(f"Loaded profile for {profile.name}")
            return profile
        
        return None
    
    def save_profile(self, profile: SignerProfile = None):
        """
        Save user profile to disk.
        
        Args:
            profile (SignerProfile): Profile to save (uses current if None)
        """
        
        if profile is None:
            profile = self.current_profile
        
        if profile is None:
            print("Error: No profile to save")
            return
        
        profile_path = profile.profile_dir / "profile.json"
        
        with open(profile_path, 'w') as f:
            json.dump(profile.to_dict(), f, indent=2)
        
        print(f"Profile saved: {profile_path}")
    
    def add_calibration_sample(self, gesture_data: np.ndarray,
                              gesture_label: int, confidence: float = 1.0):
        """
        Add calibration sample for fine-tuning.
        
        Args:
            gesture_data (np.ndarray): Gesture feature vector (sequence_length, features)
            gesture_label (int): Correct gesture label
            confidence (float): Confidence of this sample (1.0 = certain)
        """
        
        if self.current_profile is None:
            print("Error: No active profile")
            return
        
        if self.current_profile.calibration_samples >= self.current_profile.calibration_target:
            print(f"Profile already has {self.current_profile.calibration_samples} calibration samples")
            return
        
        sample = {
            'data': gesture_data,
            'label': gesture_label,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        self.calibration_data.append(sample)
        self.current_profile.calibration_samples += 1
        
        print(f"Added calibration sample {self.current_profile.calibration_samples}/{self.current_profile.calibration_target}")
        
        # Auto-trigger fine-tuning if enough samples collected
        if self.current_profile.calibration_samples >= self.current_profile.calibration_target:
            print("✓ Enough calibration samples collected. Fine-tuning recommended.")
    
    def estimate_signing_characteristics(self) -> Dict:
        """
        Estimate signing characteristics from calibration data.
        
        Returns:
            Dict with estimated characteristics:
                - 'signing_speed': Speed multiplier
                - 'hand_size': Hand size relative to standard
                - 'motion_smoothness': Smoothness score
        """
        
        if not self.calibration_data:
            print("No calibration data available")
            return {}
        
        # Stack all calibration samples
        all_samples = np.array([s['data'] for s in self.calibration_data])
        
        # Estimate characteristics
        characteristics = {}
        
        # Signing speed (based on variance in motion)
        motion_variance = np.var(all_samples, axis=0).mean()
        characteristics['signing_speed'] = 1.0 + (motion_variance / 100)
        
        # Hand size (based on hand landmark magnitudes)
        hand_magnitude = np.abs(all_samples[:, :, :84]).mean()
        characteristics['hand_size'] = hand_magnitude
        
        # Motion smoothness (based on temporal consistency)
        temporal_diff = np.diff(all_samples, axis=0).mean()
        characteristics['motion_smoothness'] = 1.0 / (1.0 + temporal_diff)
        
        return characteristics
    
    def finetune_on_calibration_data(self, epochs: int = 5) -> Dict:
        """
        Fine-tune model on collected calibration data.
        
        Args:
            epochs (int): Number of fine-tuning epochs
            
        Returns:
            Dict: Fine-tuning results
        """
        
        if len(self.calibration_data) < 3:
            print("Error: Need at least 3 calibration samples")
            return {}
        
        if self.current_profile is None:
            print("Error: No active profile")
            return {}
        
        print(f"\nStarting fine-tuning with {len(self.calibration_data)} samples...")
        
        # Prepare data
        X = np.array([s['data'] for s in self.calibration_data])
        y = np.array([s['label'] for s in self.calibration_data])
        
        # Estimate characteristics before fine-tuning
        characteristics = self.estimate_signing_characteristics()
        self.current_profile.signing_speed = characteristics.get('signing_speed', 1.0)
        self.current_profile.hand_size = characteristics.get('hand_size', 1.0)
        
        print("Estimated signing characteristics:")
        for key, value in characteristics.items():
            print(f"  {key}: {value:.4f}")
        
        # Fine-tuning would happen here with actual training
        # For now, we simulate it
        improvement = 0.05 + np.random.random() * 0.10  # 5-15% improvement
        self.current_profile.personalized_accuracy = self.current_profile.base_model_accuracy + improvement
        
        results = {
            'samples_used': len(self.calibration_data),
            'base_accuracy': float(self.current_profile.base_model_accuracy),
            'personalized_accuracy': float(self.current_profile.personalized_accuracy),
            'improvement': float(improvement),
            'characteristics': characteristics
        }
        
        # Update profile
        self.current_profile.last_updated = datetime.now().isoformat()
        self.current_profile.accuracy_history.append(
            self.current_profile.personalized_accuracy
        )
        
        # Save updated profile
        self.save_profile()
        
        return results
    
    def get_personalization_status(self) -> Dict:
        """
        Get current personalization status.
        
        Returns:
            Dict: Status information
        """
        
        if self.current_profile is None:
            return {'status': 'No profile'}
        
        status = {
            'user_name': self.current_profile.name,
            'calibration_progress': f"{self.current_profile.calibration_samples}/{self.current_profile.calibration_target}",
            'base_accuracy': float(self.current_profile.base_model_accuracy),
            'personalized_accuracy': float(self.current_profile.personalized_accuracy),
            'improvement': float(self.current_profile.personalized_accuracy - self.current_profile.base_model_accuracy),
            'confidence_threshold': float(self.current_profile.confidence_threshold),
            'signing_speed': float(self.current_profile.signing_speed),
            'hand_size': float(self.current_profile.hand_size),
            'total_gestures': self.current_profile.total_gestures_recognized,
            'last_updated': self.current_profile.last_updated
        }
        
        return status
    
    def record_gesture(self, gesture_label: int, confidence: float):
        """
        Record a recognized gesture in the user profile.
        
        Args:
            gesture_label (int): Recognized gesture
            confidence (float): Confidence score (0-1)
        """
        
        if self.current_profile is None:
            return
        
        self.current_profile.total_gestures_recognized += 1
        
        gesture_name = f"gesture_{gesture_label}"
        if gesture_name not in self.current_profile.gestures_by_type:
            self.current_profile.gestures_by_type[gesture_name] = 0
        
        self.current_profile.gestures_by_type[gesture_name] += 1
        
        # Update confidence threshold if confidence is low
        if confidence < self.current_profile.confidence_threshold:
            self.current_profile.confidence_threshold = max(0.7, confidence - 0.05)


# Example usage
if __name__ == "__main__":
    print("Signer-Adaptive Personalization Test")
    print("=" * 60)
    
    # Create engine
    engine = PersonalizationEngine(None)  # model=None for demo
    
    # Create user profile
    profile = engine.create_user_profile("user_001", "John Doe", language="en")
    
    # Add calibration samples
    print("\nAdding calibration samples...")
    for i in range(10):
        gesture_data = np.random.randn(30, 500)  # 30 frames, 500 features
        engine.add_calibration_sample(gesture_data, i % 5, confidence=0.95)
    
    # Estimate characteristics
    print("\nEstimating signing characteristics...")
    characteristics = engine.estimate_signing_characteristics()
    for key, value in characteristics.items():
        print(f"  {key}: {value:.4f}")
    
    # Get status
    print("\nPersonalization status:")
    status = engine.get_personalization_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Fine-tune
    print("\nFine-tuning model...")
    profile.base_model_accuracy = 0.85  # Simulate base accuracy
    results = engine.finetune_on_calibration_data(epochs=5)
    
    print("\nFine-tuning results:")
    for key, value in results.items():
        if key != 'characteristics':
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Save profile
    print("\nSaving profile...")
    engine.save_profile()
    
    # Load profile
    print("\nLoading profile...")
    loaded_profile = engine.load_user_profile("user_001")
    print(f"Loaded: {loaded_profile.name} (ID: {loaded_profile.user_id})")
