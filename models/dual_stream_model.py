"""
Dual-Stream Neural Network Architecture for Sign Language Recognition

This module implements the core OmniSign architecture with:
- Manual Stream: LSTM for hand landmark sequences
- Non-Manual Stream: CNN for facial expressions and body posture
- Fusion Layer: Combines both streams for final prediction
- CTC Loss: Handles continuous sequence recognition
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Conv1D, MaxPooling1D, Dense, 
    GlobalAveragePooling1D, Dropout, Concatenate, Lambda
)
import numpy as np


class DualStreamSignRecognizer:
    """
    Dual-Stream Neural Network for Sign Language Recognition.
    
    Combines:
    - Manual Stream (LSTM): Processes hand landmarks
    - Non-Manual Stream (CNN): Processes facial and body landmarks
    """
    
    def __init__(self, num_classes, manual_features=168, non_manual_features=332, 
                 sequence_length=30):
        """
        Initialize the Dual-Stream model.
        
        Args:
            num_classes (int): Number of sign classes
            manual_features (int): Hand landmark features (21 pts × 2 hands × 4 values = 168)
            non_manual_features (int): Facial + body features (~100 facial + 33 body × 4)
            sequence_length (int): Number of frames per sequence
        """
        self.num_classes = num_classes
        self.manual_features = manual_features
        self.non_manual_features = non_manual_features
        self.sequence_length = sequence_length
        self.model = None
        self.training_model = None
        
    def build_model(self):
        """
        Build the complete dual-stream architecture.
        
        Returns:
            Model: Keras model for inference (with CTC decoding)
            Model: Keras model for training (outputs raw logits)
        """
        
        # ==================== MANUAL STREAM ====================
        # Input: (batch_size, sequence_length, manual_features)
        manual_input = Input(shape=(self.sequence_length, self.manual_features), 
                            name='manual_input')
        
        # Bidirectional LSTM for temporal hand movement modeling
        manual_lstm = Bidirectional(LSTM(256, return_sequences=True))(manual_input)
        manual_lstm = Bidirectional(LSTM(128, return_sequences=False))(manual_lstm)
        # Output shape: (batch_size, 256)
        manual_output = manual_lstm
        
        
        # ==================== NON-MANUAL STREAM ====================
        # Input: (batch_size, sequence_length, non_manual_features)
        non_manual_input = Input(shape=(self.sequence_length, self.non_manual_features), 
                                name='non_manual_input')
        
        # 1D CNN for spatial feature extraction
        non_manual_conv = Conv1D(filters=64, kernel_size=3, activation='relu', 
                                padding='same')(non_manual_input)
        non_manual_conv = MaxPooling1D(pool_size=2)(non_manual_conv)
        
        non_manual_conv = Conv1D(filters=128, kernel_size=3, activation='relu', 
                                padding='same')(non_manual_conv)
        non_manual_conv = MaxPooling1D(pool_size=2)(non_manual_conv)
        
        non_manual_conv = Conv1D(filters=128, kernel_size=3, activation='relu', 
                                padding='same')(non_manual_conv)
        
        # Global average pooling
        non_manual_output = GlobalAveragePooling1D()(non_manual_conv)
        non_manual_output = Dense(128, activation='relu')(non_manual_output)
        # Output shape: (batch_size, 128)
        
        
        # ==================== FUSION LAYER ====================
        # Concatenate both streams
        fusion = Concatenate()([manual_output, non_manual_output])
        # Shape: (batch_size, 256 + 128 = 384)
        
        # Dense layers with dropout for regularization
        fusion = Dense(256, activation='relu')(fusion)
        fusion = Dropout(0.3)(fusion)
        
        fusion = Dense(128, activation='relu')(fusion)
        fusion = Dropout(0.2)(fusion)
        
        # Output logits (before softmax/CTC)
        logits = Dense(self.num_classes, activation='softmax')(fusion)
        # Output shape: (batch_size, num_classes)
        
        
        # ==================== BUILD MODELS ====================
        # Inference model (direct output)
        self.model = Model(inputs=[manual_input, non_manual_input], 
                          outputs=logits, 
                          name='DualStreamSignRecognizer')
        
        # Training model (for CTC loss)
        self.training_model = Model(inputs=[manual_input, non_manual_input], 
                                   outputs=logits, 
                                   name='DualStreamSignRecognizer_Training')
        
        return self.model, self.training_model
    
    def get_model(self):
        """Get the built model."""
        if self.model is None:
            self.build_model()
        return self.model
    
    def get_training_model(self):
        """Get the training model."""
        if self.training_model is None:
            self.build_model()
        return self.training_model
    
    def compile(self, learning_rate=1e-3):
        """
        Compile the model for training.
        
        Args:
            learning_rate (float): Learning rate for optimizer
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Use sparse categorical crossentropy for multi-class classification
        loss = keras.losses.SparseCategoricalCrossentropy()
        
        metrics = [
            keras.metrics.SparseCategoricalAccuracy(),
            keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
        ]
        
        self.training_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
    def summary(self):
        """Print model summary."""
        if self.model is None:
            self.build_model()
        self.model.summary()
    
    def predict(self, manual_input, non_manual_input, batch_size=32):
        """
        Make predictions on input data.
        
        Args:
            manual_input (np.array): Hand landmark sequences (N, sequence_length, manual_features)
            non_manual_input (np.array): Facial/body landmark sequences (N, sequence_length, non_manual_features)
            batch_size (int): Batch size for prediction
            
        Returns:
            np.array: Predicted class probabilities (N, num_classes)
        """
        return self.model.predict([manual_input, non_manual_input], 
                                 batch_size=batch_size)
    
    def get_confidence_scores(self, manual_input, non_manual_input):
        """
        Get prediction confidence scores.
        
        Args:
            manual_input (np.array): Hand landmark sequences
            non_manual_input (np.array): Facial/body landmark sequences
            
        Returns:
            tuple: (predicted_classes, confidence_scores)
        """
        predictions = self.predict(manual_input, non_manual_input)
        predicted_classes = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
        
        return predicted_classes, confidence_scores


class CTCLoss(keras.layers.Layer):
    """
    Custom CTC Loss Layer for continuous sequence recognition.
    """
    
    def __init__(self, **kwargs):
        super(CTCLoss, self).__init__(**kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost
    
    def call(self, y_true, y_pred, input_length, label_length):
        """
        Calculate CTC loss.
        
        Args:
            y_true: Ground truth labels (batch_size, max_label_length)
            y_pred: Predictions (batch_size, sequence_length, num_classes)
            input_length: Length of input sequences
            label_length: Length of label sequences
            
        Returns:
            CTC loss value
        """
        loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        return loss


class DualStreamModelWithCTC:
    """
    Complete model with CTC loss for continuous sequence recognition.
    """
    
    def __init__(self, num_classes, manual_features=168, non_manual_features=332, 
                 sequence_length=30):
        """Initialize model with CTC capabilities."""
        self.num_classes = num_classes
        self.manual_features = manual_features
        self.non_manual_features = non_manual_features
        self.sequence_length = sequence_length
        self.recognizer = DualStreamSignRecognizer(num_classes, manual_features, 
                                                   non_manual_features, sequence_length)
        self.model = None
    
    def build_model_with_ctc(self):
        """Build model with CTC loss."""
        
        # Build base model
        manual_input = Input(shape=(self.sequence_length, self.manual_features), 
                            name='manual_input')
        non_manual_input = Input(shape=(self.sequence_length, self.non_manual_features), 
                                name='non_manual_input')
        
        # Get predictions from dual-stream model
        _, training_model = self.recognizer.build_model()
        predictions = training_model([manual_input, non_manual_input])
        
        # Reshape for CTC: (batch, time_steps, num_classes)
        # Add temporal dimension if needed
        ctc_input = Lambda(lambda x: keras.backend.expand_dims(x, 1))(predictions)
        
        self.model = Model(inputs=[manual_input, non_manual_input], 
                          outputs=ctc_input,
                          name='DualStream_CTC')
        
        return self.model


# Example usage and model instantiation
if __name__ == "__main__":
    # Create model
    NUM_CLASSES = 5  # Hello, Goodbye, Thank you, How are you, I need help
    
    recognizer = DualStreamSignRecognizer(
        num_classes=NUM_CLASSES,
        manual_features=168,      # 21 pts × 2 hands × 4 values
        non_manual_features=332,  # ~100 facial + 33 body × 4
        sequence_length=30        # 30 frames per sequence
    )
    
    # Build model
    model, training_model = recognizer.build_model()
    recognizer.compile(learning_rate=1e-3)
    
    # Print summary
    print("\n" + "="*80)
    print("DUAL-STREAM SIGN LANGUAGE RECOGNITION MODEL")
    print("="*80)
    recognizer.summary()
    
    # Test with dummy data
    print("\n" + "="*80)
    print("TESTING WITH DUMMY DATA")
    print("="*80)
    
    batch_size = 4
    manual_dummy = np.random.randn(batch_size, 30, 168)
    non_manual_dummy = np.random.randn(batch_size, 30, 332)
    
    predictions = recognizer.predict(manual_dummy, non_manual_dummy)
    predicted_classes, confidence_scores = recognizer.get_confidence_scores(
        manual_dummy, non_manual_dummy
    )
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predicted classes: {predicted_classes}")
    print(f"Confidence scores: {confidence_scores}")
    print(f"Mean confidence: {np.mean(confidence_scores):.4f}")
