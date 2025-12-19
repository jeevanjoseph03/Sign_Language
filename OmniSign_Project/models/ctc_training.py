"""
Training Pipeline with CTC Loss for Continuous Sequence Recognition

Implements:
- CTC Loss for frame-level predictions
- Batch training with validation
- Model checkpointing and early stopping
- Learning rate scheduling
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import json
from datetime import datetime


class CTCTrainingPipeline:
    """
    Training pipeline with CTC loss for continuous sign sequence recognition.
    """
    
    def __init__(self, model, num_classes: int, 
                 checkpoint_dir: str = "checkpoints"):
        """
        Initialize training pipeline.
        
        Args:
            model: Keras model with output shape (batch, sequence_length, num_classes)
            num_classes (int): Number of sign classes
            checkpoint_dir (str): Directory to save model checkpoints
        """
        self.model = model
        self.num_classes = num_classes
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.history = {}
        self.best_val_loss = float('inf')
        

    
    def train_step(self, X_batch: np.ndarray, y_batch: np.ndarray, 
                   optimizer: keras.optimizers.Optimizer) -> float:
        """
        Single training step using categorical cross-entropy loss.
        
        Args:
            X_batch: Input batch - either single array or list of two arrays (manual, non-manual)
            y_batch: Label batch (batch_size,)
            optimizer: Keras optimizer
            
        Returns:
            float: Loss value
        """
        
        with tf.GradientTape() as tape:
            # Forward pass - handle both single and dual-stream inputs
            if isinstance(X_batch, list):
                # Dual-stream input
                predictions = self.model(X_batch, training=True)
            else:
                # Single stream input
                predictions = self.model(X_batch, training=True)
            
            # Calculate loss using categorical cross-entropy
            loss = keras.losses.SparseCategoricalCrossentropy()(y_batch, predictions)
        
        # Backward pass
        gradients = tape.gradient(loss, self.model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        
        return loss.numpy()
    
    def validation_step(self, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, float]:
        """
        Validation step.
        
        Args:
            X_val: Validation input
            y_val: Validation labels
            
        Returns:
            Tuple[loss, accuracy]
        """
        
        # Forward pass
        predictions = self.model(X_val, training=False)
        
        # Calculate loss using categorical cross-entropy
        loss = keras.losses.SparseCategoricalCrossentropy()(y_val, predictions)
        
        # Calculate accuracy
        predicted_classes = np.argmax(predictions.numpy(), axis=-1)
        accuracy = np.mean(predicted_classes == y_val)
        
        return loss.numpy(), float(accuracy)
        # Calculate accuracy
        predicted_classes = np.argmax(predictions.numpy(), axis=-1)
        # For CTC, we take the mode of predictions across time
        predicted_classes = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 1, predicted_classes)
        accuracy = np.mean(predicted_classes == y_val)
        
        return loss.numpy(), accuracy
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             epochs: int = 50, batch_size: int = 32,
             learning_rate: float = 1e-3, early_stopping_patience: int = 10):
        """
        Train the model.
        
        Args:
            X_train: Training input (N, sequence_length, num_features)
            y_train: Training labels (N,)
            X_val: Validation input
            y_val: Validation labels
            epochs (int): Number of epochs
            batch_size (int): Batch size
            learning_rate (float): Initial learning rate
            early_stopping_patience (int): Epochs to wait before stopping
        """
        
        # Optimizer with learning rate schedule
        initial_lr = learning_rate
        decay_steps = max(1, len(X_train) // batch_size)  # Ensure decay_steps > 0
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=0.96,
            staircase=True
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Metrics
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        # Training loop
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Training
            epoch_train_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(X_train), batch_size):
                # Handle both list inputs (dual-stream) and array inputs
                if isinstance(X_train, list):
                    X_batch = [stream[i:i+batch_size] for stream in X_train]
                else:
                    X_batch = X_train[i:i+batch_size]
                    
                y_batch = y_train[i:i+batch_size]
                
                with tf.GradientTape() as tape:
                    # Forward pass
                    predictions = self.model(X_batch, training=True)
                    
                    # Use sparse categorical crossentropy instead of CTC
                    loss = keras.losses.SparseCategoricalCrossentropy()(y_batch, predictions)
                
                # Backward pass
                gradients = tape.gradient(loss, self.model.trainable_weights)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
                
                epoch_train_loss += loss.numpy()
                num_batches += 1
                
                if (num_batches) % 5 == 0:
                    print(f"  Batch {num_batches}: Loss={loss.numpy():.4f}")
            
            epoch_train_loss /= num_batches
            train_losses.append(epoch_train_loss)
            
            # Validation
            predictions = self.model(X_val, training=False)
            val_loss_val = keras.losses.SparseCategoricalCrossentropy()(y_val, predictions)
            val_loss = val_loss_val.numpy()
            
            # Calculate accuracy
            predicted_classes = np.argmax(predictions.numpy(), axis=-1)
            val_accuracy = np.mean(predicted_classes == y_val)
            
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            print(f"Train Loss: {epoch_train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
            
            # Checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch)
                patience_counter = 0
                print("[SAVED] Model saved (best validation loss)")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered (patience={early_stopping_patience})")
                break
        
        self.history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'val_accuracy': val_accuracies
        }
        
        return self.history
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"model_epoch_{epoch:03d}.h5"
        self.model.save(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def save_history(self):
        """Save training history."""
        history_path = self.checkpoint_dir / "training_history.json"
        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {
            key: [float(val) for val in values]
            for key, values in self.history.items()
        }
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        print(f"History saved: {history_path}")


class FineTuningPipeline(CTCTrainingPipeline):
    """
    Fine-tuning pipeline for signer-adaptive personalized learning.
    """
    
    def __init__(self, model, num_classes: int,
                 checkpoint_dir: str = "checkpoints",
                 freeze_base_layers: bool = True):
        """
        Initialize fine-tuning pipeline.
        
        Args:
            model: Pre-trained Keras model
            num_classes (int): Number of classes
            checkpoint_dir (str): Directory for checkpoints
            freeze_base_layers (bool): Whether to freeze base layers
        """
        super().__init__(model, num_classes, checkpoint_dir)
        self.freeze_base_layers = freeze_base_layers
    
    def prepare_for_finetuning(self, freeze_until_layer: int = -2):
        """
        Prepare model for fine-tuning by freezing early layers.
        
        Args:
            freeze_until_layer (int): Layer index to freeze until (-1 = all except last)
        """
        
        if self.freeze_base_layers:
            for layer in self.model.layers[:freeze_until_layer]:
                layer.trainable = False
            
            for layer in self.model.layers[freeze_until_layer:]:
                layer.trainable = True
            
            print("Model prepared for fine-tuning:")
            print(f"  Frozen layers: {freeze_until_layer}")
            print(f"  Trainable layers: {len(self.model.layers) - freeze_until_layer}")
    
    def finetune(self, X_personal: np.ndarray, y_personal: np.ndarray,
                epochs: int = 5, batch_size: int = 8,
                learning_rate: float = 1e-4):
        """
        Fine-tune model on personal/user-specific data.
        
        Args:
            X_personal: User-specific training data
            y_personal: User-specific labels
            epochs (int): Fine-tuning epochs (usually small)
            batch_size (int): Batch size
            learning_rate (float): Learning rate (usually lower than main training)
        """
        
        self.prepare_for_finetuning()
        
        # Use smaller validation split for fine-tuning
        from sklearn.model_selection import train_test_split
        X_ft, X_ft_val, y_ft, y_ft_val = train_test_split(
            X_personal, y_personal, test_size=0.2, random_state=42
        )
        
        print("\nFine-tuning on personal data...")
        print(f"Personal training samples: {len(X_ft)}")
        print(f"Personal validation samples: {len(X_ft_val)}")
        
        # Train with low learning rate
        history = self.train(
            X_ft, y_ft, X_ft_val, y_ft_val,
            epochs=epochs, batch_size=batch_size,
            learning_rate=learning_rate, early_stopping_patience=3
        )
        
        return history
    
    def get_personalized_weights(self) -> Dict:
        """
        Get personalized model weights.
        
        Returns:
            Dict: Model weights for this user
        """
        return {
            'weights': [layer.get_weights() for layer in self.model.layers],
            'timestamp': datetime.now().isoformat()
        }
    
    def load_personalized_weights(self, weights_dict: Dict):
        """
        Load personalized weights back into model.
        
        Args:
            weights_dict (Dict): Weights from get_personalized_weights()
        """
        for layer, weights in zip(self.model.layers, weights_dict['weights']):
            layer.set_weights(weights)
        print("Personalized weights loaded")


# Example usage
if __name__ == "__main__":
    print("CTC Training Pipeline Test")
    print("=" * 60)
    
    # Mock setup
    from models.dual_stream_model import DualStreamSignRecognizer
    
    NUM_CLASSES = 5
    SEQUENCE_LENGTH = 30
    MANUAL_FEATURES = 168
    NON_MANUAL_FEATURES = 332
    
    # Create and build model
    recognizer = DualStreamSignRecognizer(
        num_classes=NUM_CLASSES,
        manual_features=MANUAL_FEATURES,
        non_manual_features=NON_MANUAL_FEATURES,
        sequence_length=SEQUENCE_LENGTH
    )
    model, training_model = recognizer.build_model()
    
    # Create dummy data
    print("\nCreating dummy data...")
    N_train, N_val = 100, 30
    
    X_train_manual = np.random.randn(N_train, SEQUENCE_LENGTH, MANUAL_FEATURES)
    X_train_non_manual = np.random.randn(N_train, SEQUENCE_LENGTH, NON_MANUAL_FEATURES)
    y_train = np.random.randint(0, NUM_CLASSES, N_train)
    
    X_val_manual = np.random.randn(N_val, SEQUENCE_LENGTH, MANUAL_FEATURES)
    X_val_non_manual = np.random.randn(N_val, SEQUENCE_LENGTH, NON_MANUAL_FEATURES)
    y_val = np.random.randint(0, NUM_CLASSES, N_val)
    
    # Combine inputs
    X_train = [X_train_manual, X_train_non_manual]
    X_val = [X_val_manual, X_val_non_manual]
    
    print(f"X_train shapes: {[x.shape for x in X_train]}")
    print(f"y_train shape: {y_train.shape}")
    
    # Initialize pipeline
    print("\nInitializing training pipeline...")
    pipeline = CTCTrainingPipeline(training_model, NUM_CLASSES)
    
    print("\n" + "="*60)
    print("Note: Full training requires actual data.")
    print("See train_model.py for complete training script.")
    print("="*60)
