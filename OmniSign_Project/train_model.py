"""
Comprehensive Training Script for OmniSign Model

Trains the dual-stream architecture with CTC loss on sign language data.
Includes data loading, validation, checkpointing, and visualization.
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from typing import Tuple

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.dual_stream_model import DualStreamSignRecognizer
from models.ctc_training import CTCTrainingPipeline
from data_pipeline.data_loader import DataLoader


class OmniSignTrainer:
    """
    Complete training pipeline for OmniSign model.
    """
    
    def __init__(self, data_path: str = "Sign_Language_Data",
                actions: list = None,
                checkpoint_dir: str = "checkpoints"):
        """
        Initialize trainer.
        
        Args:
            data_path (str): Path to training data
            actions (list): List of actions to recognize
            checkpoint_dir (str): Directory for checkpoints
        """
        
        if actions is None:
            self.actions = ["Hello", "How are you", "I need help", "Thank you", "Goodbye"]
        else:
            self.actions = actions
        
        self.data_path = data_path
        self.checkpoint_dir = checkpoint_dir
        self.num_classes = len(self.actions)
        
        # Model parameters
        self.sequence_length = 30
        self.manual_features = 168
        self.non_manual_features = 332
        
        # Training parameters
        self.batch_size = 32
        self.epochs = 50
        self.learning_rate = 1e-3
        
        print("OmniSign Trainer Initialized")
        print(f"  Actions: {self.actions}")
        print(f"  Classes: {self.num_classes}")
    
    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray]:
        """
        Load data, preprocess, and split.
        
        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        
        print("\n" + "="*60)
        print("LOADING AND PREPARING DATA")
        print("="*60)
        
        # Initialize loader
        loader = DataLoader(
            data_path=self.data_path,
            actions=self.actions,
            no_sequences=30,
            sequence_length=self.sequence_length
        )
        
        # Load data
        print("\n1. Loading data...")
        X, y = loader.load_data()
        
        if X.shape[0] == 0:
            print("No data found. Creating dummy data for demonstration...")
            X = np.random.randn(150, self.sequence_length, 500)
            y = np.repeat(np.arange(self.num_classes), 30)
        
        # Normalize
        print("\n2. Normalizing data...")
        X_normalized = loader.normalize(X, fit=True)
        print(f"   Range: [{X_normalized.min():.4f}, {X_normalized.max():.4f}]")
        
        # Augment
        print("\n3. Augmenting data (2x)...")
        X_augmented, y_augmented = loader.augment_data(X_normalized, y, augmentation_factor=2)
        
        # Split
        print("\n4. Splitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(
            X_augmented, y_augmented, train_ratio=0.7, val_ratio=0.15
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def build_model(self) -> Tuple:
        """
        Build dual-stream model.
        
        Returns:
            Tuple: (model, training_model)
        """
        
        print("\n" + "="*60)
        print("BUILDING MODEL")
        print("="*60)
        
        recognizer = DualStreamSignRecognizer(
            num_classes=self.num_classes,
            manual_features=self.manual_features,
            non_manual_features=self.non_manual_features,
            sequence_length=self.sequence_length
        )
        
        model, training_model = recognizer.build_model()
        recognizer.compile(learning_rate=self.learning_rate)
        
        print("\nModel Summary:")
        recognizer.summary()
        
        return recognizer, training_model
    
    def prepare_dual_stream_inputs(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into manual and non-manual streams.
        
        Args:
            X (np.ndarray): Combined features (N, sequence_length, 500)
            
        Returns:
            Tuple: (manual_stream, non_manual_stream)
        """
        
        # Manual stream: hand landmarks (168 dims)
        manual_stream = X[:, :, :self.manual_features]
        
        # Non-manual stream: facial + body landmarks
        non_manual_stream = X[:, :, self.manual_features:self.manual_features + self.non_manual_features]
        
        return manual_stream, non_manual_stream
    
    def train(self, X_train: np.ndarray, X_val: np.ndarray,
             y_train: np.ndarray, y_val: np.ndarray) -> keras.Model:
        """
        Train the model.
        
        Args:
            X_train, X_val: Training and validation data
            y_train, y_val: Training and validation labels
            
        Returns:
            keras.Model: Trained model
        """
        
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        
        # Build model
        recognizer, training_model = self.build_model()
        
        # Prepare dual-stream inputs
        print("\nPreparing dual-stream inputs...")
        X_train_manual, X_train_non_manual = self.prepare_dual_stream_inputs(X_train)
        X_val_manual, X_val_non_manual = self.prepare_dual_stream_inputs(X_val)
        
        print(f"  Manual stream: {X_train_manual.shape}")
        print(f"  Non-manual stream: {X_train_non_manual.shape}")
        
        # Create training pipeline
        pipeline = CTCTrainingPipeline(training_model, self.num_classes, self.checkpoint_dir)
        
        # Train
        print("\nStarting training...")
        history = pipeline.train(
            [X_train_manual, X_train_non_manual], y_train,
            [X_val_manual, X_val_non_manual], y_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            early_stopping_patience=10
        )
        
        # Save history
        pipeline.save_history()
        
        return recognizer.model, history
    
    def evaluate(self, model: keras.Model, X_test: np.ndarray, 
                y_test: np.ndarray) -> float:
        """
        Evaluate model on test set.
        
        Args:
            model: Trained model
            X_test: Test data
            y_test: Test labels
            
        Returns:
            float: Test accuracy
        """
        
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)
        
        # Prepare inputs
        X_test_manual, X_test_non_manual = self.prepare_dual_stream_inputs(X_test)
        
        # Predict
        predictions = model.predict([X_test_manual, X_test_non_manual])
        
        # Calculate accuracy
        predicted_classes = np.argmax(predictions, axis=-1)
        
        # predicted_classes is already (N,) so we can use it directly
        accuracy = np.mean(predicted_classes == y_test)
        
        print(f"\nTest Set Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Correct: {np.sum(predicted_classes == y_test)}/{len(y_test)}")
        
        # Per-class accuracy
        print(f"\nPer-class accuracy:")
        for class_idx, action in enumerate(self.actions):
            mask = y_test == class_idx
            if np.sum(mask) > 0:
                class_acc = np.mean(predicted_classes[mask] == class_idx)
                print(f"  {action}: {class_acc:.4f}")
        
        return accuracy
    
    def visualize_training(self, history: dict, output_file: str = "training_history.png"):
        """
        Visualize training history.
        
        Args:
            history (dict): Training history from pipeline
            output_file (str): Output file path
        """
        
        print(f"\nVisualizing training history...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(history['val_accuracy'], label='Val Accuracy', marker='s', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved to {output_file}")
        plt.close()
    
    def save_model(self, model: keras.Model, filename: str = "omnisign_model.h5"):
        """
        Save trained model.
        
        Args:
            model: Trained model
            filename (str): Output filename
        """
        
        model.save(filename)
        print(f"\nModel saved to {filename}")


def main():
    """Main training script."""
    
    print("\n" + "="*70)
    print("          OMNISIGN: COMPREHENSIVE MODEL TRAINING SCRIPT")
    print("="*70)
    
    # Initialize trainer
    trainer = OmniSignTrainer(
        data_path="Sign_Language_Data",
        actions=["Hello", "How are you", "I need help", "Thank you", "Goodbye"]
    )
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.load_and_prepare_data()
    
    # Train
    model, history = trainer.train(X_train, X_val, y_train, y_val)
    
    # Evaluate
    test_accuracy = trainer.evaluate(model, X_test, y_test)
    
    # Visualize
    trainer.visualize_training(history)
    
    # Save
    trainer.save_model(model, "sign_language_model.h5")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"\nModel saved as: sign_language_model.h5")
    print(f"Checkpoints saved in: {trainer.checkpoint_dir}/")
    print(f"History saved as: {trainer.checkpoint_dir}/training_history.json")
    print("\nNext steps:")
    print("  1. Run 'python main_app.py' for interactive communication")
    print("  2. Run 'python predict_sign.py' for batch predictions")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
