"""
OmniSign: Main Application
Bidirectional Communication Framework for Sign Language

Features:
- Real-time sign language recognition
- Multilingual translation
- Bidirectional communication (Sign ↔ Speech/Text)
- Signer-adaptive personalization
- Confidence-based error handling
"""

import os
import sys
import numpy as np
import cv2
from typing import Tuple, Optional, Dict
from pathlib import Path
import json
from datetime import datetime

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import OmniSign modules
from models.dual_stream_model import DualStreamSignRecognizer
from data_pipeline.feature_extractor import MediaPipeFeatureExtractor
from modules.translator import BidirectionalCommunicationEngine, Language
from modules.personalization import PersonalizationEngine, SignerProfile


class OmniSignApp:
    """
    Main OmniSign Application for bidirectional sign language communication.
    """
    
    def __init__(self, model_path: Optional[str] = None,
                 actions: list = None):
        """
        Initialize OmniSign application.
        
        Args:
            model_path (str): Path to pre-trained model
            actions (list): List of recognizable signs
        """
        
        if actions is None:
            self.actions = ["Hello", "Goodbye", "Thank you", "How are you", "I need help"]
        else:
            self.actions = actions
        
        self.num_classes = len(self.actions)
        
        # Initialize components
        print("Initializing OmniSign components...")
        
        # 1. Model
        self.model = DualStreamSignRecognizer(
            num_classes=self.num_classes,
            manual_features=168,
            non_manual_features=332,
            sequence_length=30
        )
        self.model.build_model()
        
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model.model.load_weights(model_path)
        
        # 2. Feature Extractor
        self.feature_extractor = MediaPipeFeatureExtractor()
        
        # 3. Translator
        self.translator = BidirectionalCommunicationEngine()
        
        # 4. Personalization
        self.personalization = PersonalizationEngine(self.model.model)
        self.current_user = None
        
        # Session data
        self.session_start = datetime.now()
        self.recognition_history = []
        self.confidence_threshold = 0.85
        
        print("[OK] OmniSign initialized successfully")
    
    def create_session(self, user_id: str, user_name: str,
                      language: str = "en") -> SignerProfile:
        """
        Create a new session for a user.
        
        Args:
            user_id (str): User identifier
            user_name (str): User's name
            language (str): Preferred language
            
        Returns:
            SignerProfile: User profile
        """
        
        # Try to load existing profile
        profile = self.personalization.load_user_profile(user_id)
        
        if profile is None:
            # Create new profile
            profile = self.personalization.create_user_profile(
                user_id, user_name, language
            )
        
        self.current_user = profile
        self.confidence_threshold = profile.confidence_threshold
        
        print(f"\n{'='*60}")
        print(f"Session started for: {user_name}")
        print(f"Language: {language}")
        print(f"Calibration: {profile.calibration_samples}/{profile.calibration_target} samples")
        print(f"{'='*60}\n")
        
        return profile
    
    def recognize_sign_from_webcam(self, duration_seconds: int = 5,
                                  num_frames: int = 30) -> Dict:
        """
        Capture and recognize sign language from webcam.
        
        Args:
            duration_seconds (int): Duration to capture
            num_frames (int): Number of frames to collect
            
        Returns:
            Dict with recognition results:
                - 'sign': Recognized sign
                - 'confidence': Confidence score
                - 'translation': Text translation
                - 'status': Success/failure status
        """
        
        print("\nCapturing gesture (press 'q' to stop)...")
        
        # Extract feature sequence
        sequence = self.feature_extractor.extract_from_webcam(
            duration_seconds=duration_seconds,
            num_frames=num_frames
        )
        
        # Normalize
        sequence = self.feature_extractor.normalize_landmarks(sequence)
        
        # Split into manual and non-manual streams
        manual_stream = sequence[:, :168]      # Hand features
        non_manual_stream = sequence[:, 168:]  # Facial + body features
        
        # Make prediction
        manual_input = np.expand_dims(manual_stream, 0)  # Add batch dimension
        non_manual_input = np.expand_dims(non_manual_stream, 0)
        
        predictions = self.model.predict(manual_input, non_manual_input)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        sign = self.actions[predicted_class]
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            status = "low_confidence"
            translation = None
            print(f"\n⚠️ Low confidence: {confidence:.4f} < {self.confidence_threshold}")
        else:
            status = "success"
            
            # Translate
            translation_result = self.translator.process_sign_input(
                sign, self.current_user.language
            )
            translation = translation_result['translation']
        
        # Record in profile
        if self.current_user:
            self.personalization.record_gesture(predicted_class, confidence)
        
        result = {
            'sign': sign,
            'confidence': confidence,
            'translation': translation,
            'status': status,
            'language': self.current_user.language if self.current_user else 'en',
            'timestamp': datetime.now().isoformat()
        }
        
        self.recognition_history.append(result)
        
        return result
    
    def generate_sign_from_text(self, text: str, source_language: str = "en") -> Dict:
        """
        Convert text to sign language representation.
        
        Args:
            text (str): Input text
            source_language (str): Language code
            
        Returns:
            Dict with signs to generate
        """
        
        print(f"\nProcessing text: '{text}'")
        
        result = self.translator.process_text_input(text, source_language)
        
        print(f"English translation: {result['english']}")
        print(f"Signs to generate: {result['signs_to_generate']}")
        
        return result
    
    def personalization_wizard(self) -> Dict:
        """
        Interactive personalization wizard to calibrate for user.
        
        Returns:
            Dict: Calibration results
        """
        
        if not self.current_user:
            print("Error: No active user session")
            return {}
        
        print("\n" + "="*60)
        print("PERSONALIZATION WIZARD")
        print("="*60)
        
        print(f"\nThis will calibrate the system for {self.current_user.name}")
        print(f"We need {self.current_user.calibration_target} gesture samples")
        
        for gesture_idx, gesture_name in enumerate(self.actions):
            if self.current_user.calibration_samples >= self.current_user.calibration_target:
                break
            
            print(f"\n[{gesture_idx+1}] Gesture: {gesture_name}")
            print(f"    Remaining: {self.current_user.calibration_target - self.current_user.calibration_samples}")
            
            input("Press Enter to capture gesture...")
            
            # Capture
            sequence = self.feature_extractor.extract_from_webcam(
                duration_seconds=3, num_frames=30
            )
            sequence = self.feature_extractor.normalize_landmarks(sequence)
            
            # Verify by showing prediction
            manual_stream = sequence[:, :168]
            non_manual_stream = sequence[:, 168:]
            
            predictions = self.model.predict(
                np.expand_dims(manual_stream, 0),
                np.expand_dims(non_manual_stream, 0)
            )
            predicted_idx = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            predicted_name = self.actions[predicted_idx]
            
            print(f"    Predicted: {predicted_name} (confidence: {confidence:.4f})")
            
            # Add to calibration
            self.personalization.add_calibration_sample(
                sequence, gesture_idx, confidence=0.95
            )
        
        # Fine-tune if enough samples
        if self.current_user.calibration_samples >= self.current_user.calibration_target:
            print("\n[OK] Enough samples collected. Fine-tuning model...")
            self.current_user.base_model_accuracy = 0.85  # Assume base accuracy
            
            results = self.personalization.finetune_on_calibration_data(epochs=5)
            
            print("\nFine-tuning results:")
            print(f"  Base accuracy: {results['base_accuracy']:.4f}")
            print(f"  Personalized accuracy: {results['personalized_accuracy']:.4f}")
            print(f"  Improvement: +{results['improvement']:.4f}")
            
            return results
        
        return {}
    
    def interactive_session(self, interactive: bool = True):
        """
        Run interactive communication session.
        
        Args:
            interactive (bool): Whether to prompt for actions
        """
        
        if not self.current_user:
            print("Error: Create session first")
            return
        
        print("\n" + "="*60)
        print("INTERACTIVE COMMUNICATION SESSION")
        print("="*60)
        print("\nCommands:")
        print("  'sign'  - Recognize sign from webcam")
        print("  'text'  - Enter text to convert to sign")
        print("  'history' - Show recognition history")
        print("  'status' - Show personalization status")
        print("  'quit'  - Exit session")
        print("="*60)
        
        while True:
            try:
                command = input("\nEnter command: ").strip().lower()
                
                if command == "sign":
                    result = self.recognize_sign_from_webcam(duration_seconds=5)
                    
                    if result['status'] == 'success':
                        print(f"\n[RECOGNIZED] {result['sign']}")
                        print(f"  Confidence: {result['confidence']:.4f}")
                        print(f"  Translation ({result['language']}): {result['translation']}")
                    else:
                        print(f"\n✗ Low confidence. Please try again.")
                
                elif command == "text":
                    text = input("Enter text: ")
                    language = input("Language (en/ml/hi/es/fr): ").lower() or "en"
                    
                    result = self.generate_sign_from_text(text, language)
                    print(f"\nWould generate signs: {result['signs_to_generate']}")
                
                elif command == "history":
                    print("\nRecognition History:")
                    for i, entry in enumerate(self.recognition_history[-5:]):
                        print(f"  {i+1}. {entry['sign']} (confidence: {entry['confidence']:.4f})")
                
                elif command == "status":
                    status = self.personalization.get_personalization_status()
                    print("\nPersonalization Status:")
                    for key, value in status.items():
                        print(f"  {key}: {value}")
                
                elif command == "quit":
                    print("\nSession ended. Thank you for using OmniSign!")
                    break
                
                else:
                    print("Unknown command. Try 'sign', 'text', 'history', 'status', or 'quit'")
            
            except KeyboardInterrupt:
                print("\n\nSession interrupted.")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def save_session(self, output_file: Optional[str] = None):
        """
        Save session data and user profile.
        
        Args:
            output_file (str): Output file path (optional)
        """
        
        if self.current_user:
            self.personalization.save_profile()
        
        # Save session history
        if output_file is None:
            output_file = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        session_data = {
            'session_start': self.session_start.isoformat(),
            'session_end': datetime.now().isoformat(),
            'user': {
                'id': self.current_user.user_id if self.current_user else None,
                'name': self.current_user.name if self.current_user else None,
            },
            'recognitions': self.recognition_history,
            'num_recognitions': len(self.recognition_history),
            'average_confidence': np.mean([r['confidence'] for r in self.recognition_history]) if self.recognition_history else 0
        }
        
        with open(output_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"\nSession saved to {output_file}")


def main():
    """Main entry point for OmniSign application."""
    
    print("\n" + "="*60)
    print("        OMNISIGN: Multilingual Bidirectional")
    print("        Sign Language Communication Framework")
    print("="*60)
    
    # Initialize app
    app = OmniSignApp(
        model_path=None,  # Set to actual model path when available
        actions=["Hello", "Goodbye", "Thank you", "How are you", "I need help"]
    )
    
    # Create user session
    user_id = input("\nEnter user ID: ").strip() or "user_001"
    user_name = input("Enter your name: ").strip() or "User"
    language = input("Preferred language (en/ml/hi): ").strip().lower() or "en"
    
    profile = app.create_session(user_id, user_name, language)
    
    # Check if personalization needed
    if profile.calibration_samples < profile.calibration_target:
        response = input("\nWould you like to personalize the system? (y/n): ").lower()
        if response == 'y':
            app.personalization_wizard()
    
    # Start interactive session
    app.interactive_session(interactive=True)
    
    # Save session
    app.save_session()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
