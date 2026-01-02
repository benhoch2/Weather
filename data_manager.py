import os
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
import json

class RadarDataManager:
    """
    Manages radar image data for training and prediction.
    """
    
    def __init__(self, data_dir="data/radar_images", sequence_length=12, image_size=(512, 512)):
        """
        Initialize the data manager.
        
        Args:
            data_dir: Directory containing radar images
            sequence_length: Number of images in a sequence (default 12 for 1 hour)
            image_size: Target size for images (width, height)
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.image_size = image_size
        
    def get_all_radar_images(self):
        """
        Get all radar images sorted by timestamp.
        
        Returns:
            List of tuples (timestamp, filepath)
        """
        images = []
        for file in self.data_dir.glob("radar_*.png"):
            try:
                # Extract timestamp from filename: radar_1767120804.png
                timestamp = int(file.stem.split('_')[1])
                images.append((timestamp, file))
            except (ValueError, IndexError):
                continue
        
        # Sort by timestamp
        images.sort(key=lambda x: x[0])
        return images
    
    def load_image(self, filepath, normalize=True):
        """
        Load and preprocess a single radar image.
        
        Args:
            filepath: Path to the image
            normalize: Whether to normalize pixel values to [0, 1]
        
        Returns:
            numpy array of shape (height, width, channels)
        """
        img = Image.open(filepath)
        
        # Resize if needed
        if img.size != self.image_size:
            img = img.resize(self.image_size, Image.LANCZOS)
        
        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize to [0, 1]
        if normalize:
            img_array = img_array / 255.0
        
        return img_array
    
    def create_sequences(self, min_sequences=1):
        """
        Create training sequences from available images.
        Each sequence contains sequence_length images as input and 1 image as target.
        
        Args:
            min_sequences: Minimum number of sequences required
        
        Returns:
            Tuple of (X, y) where:
                X: array of shape (num_sequences, sequence_length, height, width, channels)
                y: array of shape (num_sequences, height, width, channels)
        """
        images = self.get_all_radar_images()
        
        if len(images) < self.sequence_length + 1:
            raise ValueError(f"Not enough images. Need at least {self.sequence_length + 1}, found {len(images)}")
        
        X_sequences = []
        y_targets = []
        
        # Create overlapping sequences
        for i in range(len(images) - self.sequence_length):
            # Load sequence of images
            sequence = []
            for j in range(self.sequence_length):
                img = self.load_image(images[i + j][1])
                sequence.append(img)
            
            # Load target image (next frame)
            target = self.load_image(images[i + self.sequence_length][1])
            
            X_sequences.append(sequence)
            y_targets.append(target)
        
        # Convert to numpy arrays
        X = np.array(X_sequences, dtype=np.float32)
        y = np.array(y_targets, dtype=np.float32)
        
        print(f"Created {len(X)} sequences from {len(images)} images")
        print(f"Input shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        return X, y
    
    def get_latest_sequence(self):
        """
        Get the most recent sequence of images for prediction.
        
        Returns:
            numpy array of shape (1, sequence_length, height, width, channels)
            or None if not enough images
        """
        images = self.get_all_radar_images()
        
        if len(images) < self.sequence_length:
            return None
        
        # Get the last sequence_length images
        sequence = []
        for i in range(self.sequence_length):
            img = self.load_image(images[-(self.sequence_length - i)][1])
            sequence.append(img)
        
        # Add batch dimension
        return np.array([sequence], dtype=np.float32)
    
    def save_metadata(self, metadata, filename="training_metadata.json"):
        """
        Save training metadata to a JSON file.
        """
        filepath = self.data_dir / filename
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {filepath}")
    
    def load_metadata(self, filename="training_metadata.json"):
        """
        Load training metadata from a JSON file.
        """
        filepath = self.data_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        return None
    
    def data_generator(self, batch_size=1, validation_split=0.2, training=True):
        """
        Generator that yields batches of sequences on-the-fly without loading all data into memory.
        
        Args:
            batch_size: Number of sequences per batch
            validation_split: Fraction of data to use for validation
            training: If True, yields training data; if False, yields validation data
        
        Yields:
            Tuple of (X_batch, y_batch) where:
                X_batch: array of shape (batch_size, sequence_length, height, width, channels)
                y_batch: array of shape (batch_size, height, width, channels)
        """
        images = self.get_all_radar_images()
        
        if len(images) < self.sequence_length + 1:
            raise ValueError(f"Not enough images. Need at least {self.sequence_length + 1}, found {len(images)}")
        
        # Calculate number of possible sequences
        num_sequences = len(images) - self.sequence_length
        split_idx = int((1 - validation_split) * num_sequences)
        
        # Determine which sequences to use
        if training:
            start_idx = 0
            end_idx = split_idx
        else:
            start_idx = split_idx
            end_idx = num_sequences
        
        while True:  # Infinite loop for keras fit
            # Shuffle indices for training
            indices = list(range(start_idx, end_idx))
            if training:
                np.random.shuffle(indices)
            
            # Generate batches
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                
                X_batch = []
                y_batch = []
                
                for i in batch_indices:
                    # Load sequence of images
                    sequence = []
                    for j in range(self.sequence_length):
                        img = self.load_image(images[i + j][1])
                        sequence.append(img)
                    
                    # Load target image
                    target = self.load_image(images[i + self.sequence_length][1])
                    
                    X_batch.append(sequence)
                    y_batch.append(target)
                
                # Convert to numpy arrays
                X_batch = np.array(X_batch, dtype=np.float32)
                y_batch = np.array(y_batch, dtype=np.float32)
                
                yield X_batch, y_batch
    
    def get_num_sequences(self):
        """
        Get the total number of sequences that can be created from available images.
        
        Returns:
            Number of sequences
        """
        images = self.get_all_radar_images()
        if len(images) < self.sequence_length + 1:
            return 0
        return len(images) - self.sequence_length


if __name__ == "__main__":
    # Test the data manager
    manager = RadarDataManager()
    
    images = manager.get_all_radar_images()
    print(f"Found {len(images)} radar images")
    
    if len(images) >= 13:
        print("\nCreating training sequences...")
        X, y = manager.create_sequences()
        print(f"\nDataset ready:")
        print(f"  Input sequences: {X.shape}")
        print(f"  Target images: {y.shape}")
    else:
        print(f"\nNeed at least 13 images to create sequences. Currently have {len(images)}.")
        print("Keep collecting images with fetch_radar_continuous.py")
