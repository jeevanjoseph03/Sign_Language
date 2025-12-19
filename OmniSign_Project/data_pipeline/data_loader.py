"""
Data Loader and Preprocessing Pipeline

Loads collected data and prepares it for model training with:
- Data augmentation techniques
- Normalization
- Train/validation/test splitting
- Batch generation
"""

import numpy as np
import os
from pathlib import Path
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """
    Load and preprocess sign language data.
    """
    
    def __init__(self, data_path: str, actions: List[str], 
                 no_sequences: int = 30, sequence_length: int = 30):
        """
        Initialize data loader.
        
        Args:
            data_path (str): Path to data directory
            actions (List[str]): List of action classes
            no_sequences (int): Number of sequences per action
            sequence_length (int): Number of frames per sequence
        """
        self.data_path = Path(data_path)
        self.actions = actions
        self.no_sequences = no_sequences
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all data from directory structure.
        
        Expected structure:
            data_path/
            ├── Action1/
            │   ├── 0/
            │   │   ├── 0.npy
            │   │   └── ...
            │   └── 1/
            ├── Action2/
            └── ...
        
        Returns:
            Tuple[X, y]: 
                - X: Feature sequences (N, sequence_length, 500)
                - y: Labels (N,) - class indices
        """
        
        X = []
        y = []
        
        for action_idx, action in enumerate(self.actions):
            action_path = self.data_path / action
            
            if not action_path.exists():
                print(f"Warning: {action_path} not found")
                continue
            
            print(f"Loading {action}...")
            
            for sequence_idx in range(self.no_sequences):
                seq_path = action_path / str(sequence_idx)
                
                if not seq_path.exists():
                    print(f"  Warning: Sequence {sequence_idx} not found")
                    continue
                
                sequence_data = []
                
                # Load all .npy files in this sequence
                for frame_idx in range(self.sequence_length):
                    frame_file = seq_path / f"{frame_idx}.npy"
                    
                    if frame_file.exists():
                        frame_data = np.load(frame_file)
                        # Ensure frame data is 1D and reshape to 500 features
                        frame_data = np.array(frame_data).flatten()
                        if len(frame_data) < 500:
                            # Pad with zeros if too short
                            frame_data = np.pad(frame_data, (0, 500 - len(frame_data)))
                        elif len(frame_data) > 500:
                            # Truncate if too long
                            frame_data = frame_data[:500]
                        sequence_data.append(frame_data)
                    else:
                        # Pad with zeros if frame missing
                        sequence_data.append(np.zeros(500))
                
                # Convert to array - ensure consistent shape
                sequence_array = np.array(sequence_data, dtype=np.float32)
                
                # Verify and fix shape
                if sequence_array.shape[0] == self.sequence_length:
                    if sequence_array.shape[1] != 500:
                        sequence_array = sequence_array[:, :500]  # Ensure 500 features
                        if sequence_array.shape[1] < 500:
                            sequence_array = np.pad(sequence_array, 
                                                   ((0, 0), (0, 500 - sequence_array.shape[1])))
                    X.append(sequence_array)
                    y.append(action_idx)
        
        if len(X) == 0:
            print("\nNo valid sequences found. Creating dummy data for demonstration...")
            # Create dummy data if no real data found
            X = np.random.randn(150, self.sequence_length, 500).astype(np.float32)
            y = np.repeat(np.arange(len(self.actions)), 30)
        else:
            X = np.array(X, dtype=np.float32)
        
        y = np.array(y)
        
        print(f"\nLoaded data shape: X={X.shape}, y={y.shape}")
        if len(y) > 0:
            print(f"Classes: {dict(zip(self.actions, np.bincount(y)))}")
        
        return X, y
    
    def normalize(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize features to zero mean and unit variance.
        
        Args:
            X (np.ndarray): Input data (N, sequence_length, num_features)
            fit (bool): Whether to fit the scaler or use existing fit
            
        Returns:
            np.ndarray: Normalized data
        """
        
        # Reshape for scaling
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        if fit:
            X_normalized = self.scaler.fit_transform(X_reshaped)
        else:
            X_normalized = self.scaler.transform(X_reshaped)
        
        # Reshape back
        X_normalized = X_normalized.reshape(original_shape)
        
        return X_normalized
    
    def augment_data(self, X: np.ndarray, y: np.ndarray, 
                     augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation techniques.
        
        Techniques:
        - Random flipping (left-right mirror)
        - Temporal shift (forward-backward)
        - Gaussian noise
        - Random scaling
        
        Args:
            X (np.ndarray): Input data (N, sequence_length, 500)
            y (np.ndarray): Labels (N,)
            augmentation_factor (int): How many augmented copies per sample
            
        Returns:
            Tuple[X_aug, y_aug]: Augmented data
        """
        
        X_augmented = [X]
        y_augmented = [y]
        
        for _ in range(augmentation_factor - 1):
            X_aug = X.copy()
            
            # Random flipping (mirror on x-axis for hand landmarks)
            if np.random.rand() > 0.5:
                # Mirror x-coordinates of hands (indices 0-167)
                X_aug[:, :, :84] = 1 - X_aug[:, :, :84]  # Mirror hand x-coords
            
            # Temporal shift
            if np.random.rand() > 0.5:
                shift = np.random.randint(-2, 3)  # Shift by -2 to 2 frames
                X_aug = np.roll(X_aug, shift, axis=1)
            
            # Gaussian noise
            if np.random.rand() > 0.5:
                noise = np.random.normal(0, 0.01, X_aug.shape)
                X_aug = np.clip(X_aug + noise, 0, 1)
            
            # Random scaling
            if np.random.rand() > 0.5:
                scale_factor = np.random.uniform(0.95, 1.05)
                X_aug = np.clip(X_aug * scale_factor, 0, 1)
            
            X_augmented.append(X_aug)
            y_augmented.append(y)
        
        X_final = np.concatenate(X_augmented, axis=0)
        y_final = np.concatenate(y_augmented, axis=0)
        
        print(f"Augmented data shape: X={X_final.shape}, y={y_final.shape}")
        
        return X_final, y_final
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                  train_ratio: float = 0.7, val_ratio: float = 0.15,
                  random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, 
                                                    np.ndarray, np.ndarray,
                                                    np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X (np.ndarray): Input data
            y (np.ndarray): Labels
            train_ratio (float): Proportion for training
            val_ratio (float): Proportion for validation
            random_state (int): Random seed
            
        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=1-train_ratio-val_ratio, 
            random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        val_size = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size,
            random_state=random_state, stratify=y_temp
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Val set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_batches(self, X: np.ndarray, y: np.ndarray, 
                      batch_size: int = 32, shuffle: bool = True):
        """
        Create data batches for training.
        
        Args:
            X (np.ndarray): Input data
            y (np.ndarray): Labels
            batch_size (int): Batch size
            shuffle (bool): Whether to shuffle data
            
        Yields:
            Tuple: (X_batch, y_batch)
        """
        
        indices = np.arange(len(X))
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, len(X), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            yield X[batch_indices], y[batch_indices]


# Example usage
if __name__ == "__main__":
    print("Data Loader Test")
    print("=" * 60)
    
    # Initialize loader
    loader = DataLoader(
        data_path="Sign_Language_Data",
        actions=["Hello", "How are you", "I need help", "Thank you", "Goodbye"],
        no_sequences=30,
        sequence_length=30
    )
    
    # Load data
    X, y = loader.load_data()
    
    # Normalize
    print("\nNormalizing data...")
    X_normalized = loader.normalize(X, fit=True)
    print(f"Normalized X shape: {X_normalized.shape}")
    print(f"Normalized X range: [{X_normalized.min():.4f}, {X_normalized.max():.4f}]")
    
    # Augment
    print("\nAugmenting data (2x)...")
    X_aug, y_aug = loader.augment_data(X_normalized, y, augmentation_factor=2)
    
    # Split
    print("\nSplitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X_aug, y_aug)
    
    # Create batches
    print("\nCreating batches (batch_size=32)...")
    batch_count = 0
    for X_batch, y_batch in loader.create_batches(X_train, y_train, batch_size=32):
        batch_count += 1
        if batch_count <= 3:
            print(f"Batch {batch_count}: X={X_batch.shape}, y={y_batch.shape}")
    
    print(f"Total batches: {batch_count}")
