import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class RadarPredictionModel:
    """
    ConvLSTM-based model for predicting the next radar image.
    """
    
    def __init__(self, sequence_length=12, image_size=(256, 256), channels=3):
        """
        Initialize the model.
        
        Args:
            sequence_length: Number of input frames
            image_size: Size of images (height, width)
            channels: Number of color channels (3 for RGB)
        """
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.channels = channels
        self.model = None
        
    def build_model(self):
        """
        Build a ConvLSTM model for radar prediction.
        
        This model uses ConvLSTM layers which are designed for spatiotemporal data.
        They combine convolutional and LSTM operations to capture both spatial
        and temporal patterns in the sequence.
        """
        
        # Input shape: (sequence_length, height, width, channels)
        inputs = layers.Input(
            shape=(self.sequence_length, self.image_size[0], self.image_size[1], self.channels)
        )
        
        # ConvLSTM layers - these process the sequence while maintaining spatial structure
        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=True,
            activation='relu'
        )(inputs)
        x = layers.BatchNormalization()(x)
        
        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=True,
            activation='relu'
        )(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=False,  # Only return the last output
            activation='relu'
        )(x)
        x = layers.BatchNormalization()(x)
        
        # Decoder - convert the ConvLSTM output back to an image
        x = layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        
        x = layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        
        # Output layer - predict RGB image
        outputs = layers.Conv2D(
            filters=self.channels,
            kernel_size=(3, 3),
            padding='same',
            activation='sigmoid'  # Output values in [0, 1]
        )(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='radar_prediction')
        
        # Compile with appropriate loss and optimizer
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',  # Mean squared error for image prediction
            metrics=['mae']  # Mean absolute error as additional metric
        )
        
        return self.model
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def save(self, filepath='radar_model.keras'):
        """Save the model to disk."""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load(self, filepath='radar_model.keras'):
        """Load a model from disk."""
        try:
            self.model = keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Could not load model: {e}")
            return False
    
    def predict(self, sequence):
        """
        Predict the next frame given a sequence of frames.
        
        Args:
            sequence: numpy array of shape (1, sequence_length, height, width, channels)
        
        Returns:
            Predicted image as numpy array of shape (height, width, channels)
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        prediction = self.model.predict(sequence, verbose=0)
        return prediction[0]  # Remove batch dimension
    
    def train_on_batch(self, X_batch, y_batch):
        """
        Train the model on a single batch (for online learning).
        
        Args:
            X_batch: Input sequences
            y_batch: Target images
        
        Returns:
            Loss value
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        return self.model.train_on_batch(X_batch, y_batch)
    
    def evaluate(self, X, y):
        """
        Evaluate the model on a dataset.
        
        Args:
            X: Input sequences
            y: Target images
        
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")
        
        results = self.model.evaluate(X, y, verbose=0)
        return {
            'loss': results[0],
            'mae': results[1]
        }


if __name__ == "__main__":
    # Test the model architecture
    print("Building radar prediction model...")
    model = RadarPredictionModel()
    model.build_model()
    
    print("\nModel architecture:")
    model.summary()
    
    print(f"\nModel input shape: (batch, {model.sequence_length}, {model.image_size[0]}, {model.image_size[1]}, {model.channels})")
    print(f"Model output shape: (batch, {model.image_size[0]}, {model.image_size[1]}, {model.channels})")
