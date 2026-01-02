import numpy as np
from datetime import datetime
import sys
from data_manager import RadarDataManager
from radar_model import RadarPredictionModel
import tensorflow as tf

class SaveModelCallback(tf.keras.callbacks.Callback):
    """Callback to save model after each epoch."""
    def __init__(self, model_wrapper, model_path):
        super().__init__()
        self.model_wrapper = model_wrapper
        self.model_path = model_path
    
    def on_epoch_end(self, epoch, logs=None):
        self.model_wrapper.save(self.model_path)
        print(f"\n  Model saved to {self.model_path}")

def train_initial_model(epochs=50, batch_size=4):
    """
    Train the radar prediction model on collected data using a data generator.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    print("=" * 70)
    print("Radar Prediction Model - Initial Training")
    print("=" * 70)
    print()
    
    # Initialize data manager
    print("Initializing data manager...")
    data_manager = RadarDataManager()
    
    # Get number of sequences
    num_sequences = data_manager.get_num_sequences()
    
    if num_sequences < 1:
        print("Error: Not enough images to create sequences.")
        print("You need to collect more radar images first.")
        print("Run fetch_radar_continuous.py and wait for at least 13 images (65 minutes).")
        return
    
    print(f"Found {num_sequences} sequences")
    
    if num_sequences < 10:
        print("\nWarning: Very few sequences available.")
        print("The model will train but may not perform well.")
        print("Collect more images for better results.")
        print()
    
    # Calculate train/val split
    validation_split = 0.2
    num_train = int((1 - validation_split) * num_sequences)
    num_val = num_sequences - num_train
    
    print(f"Training sequences: {num_train}")
    print(f"Validation sequences: {num_val}")
    print()
    
    # Build or load the model
    model_path = 'radar_model.keras'
    model = RadarPredictionModel(
        sequence_length=12,
        image_size=(512, 512),
        channels=3
    )
    
    # Try to load existing model to continue training
    if model.load(model_path):
        print("Loaded existing model - continuing training from current weights")
    else:
        print("No existing model found - building new model with random weights")
        model.build_model()
    print()
    
    # Print model summary
    print("Model architecture:")
    model.summary()
    print()
    
    # Create data generators
    print("Creating data generators...")
    train_gen = data_manager.data_generator(
        batch_size=batch_size, 
        validation_split=validation_split, 
        training=True
    )
    
    val_gen = None
    if num_val > 0:
        val_gen = data_manager.data_generator(
            batch_size=batch_size,
            validation_split=validation_split,
            training=False
        )
    
    # Calculate steps per epoch
    steps_per_epoch = max(1, num_train // batch_size)
    validation_steps = max(1, num_val // batch_size) if num_val > 0 else None
    
    print(f"Steps per epoch: {steps_per_epoch}")
    if validation_steps:
        print(f"Validation steps: {validation_steps}")
    print()
    
    # Train the model
    model_path = 'radar_model.keras'
    save_callback = SaveModelCallback(model, model_path)
    
    print(f"Starting training for {epochs} epochs...")
    print("Note: Using data generator to avoid loading all data into memory")
    print("Model will be saved after each epoch")
    print("-" * 70)
    
    history = model.model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=[save_callback],
        verbose=1
    )
    
    print()
    print("-" * 70)
    print("Training completed!")
    print()
    
    # Save the model
    model_path = 'radar_model.keras'
    model.save(model_path)
    
    # Save training metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'num_sequences': num_sequences,
        'num_training': num_train,
        'num_validation': num_val,
        'epochs': epochs,
        'batch_size': batch_size,
        'final_loss': float(history.history['loss'][-1]),
        'final_mae': float(history.history['mae'][-1]),
    }
    
    if num_val > 0 and 'val_loss' in history.history:
        metadata['final_val_loss'] = float(history.history['val_loss'][-1])
        metadata['final_val_mae'] = float(history.history['val_mae'][-1])
    
    data_manager.save_metadata(metadata)
    
    print("=" * 70)
    print("Training Summary:")
    print(f"  Model saved: {model_path}")
    print(f"  Training sequences: {num_train}")
    print(f"  Final training loss: {history.history['loss'][-1]:.6f}")
    print(f"  Final training MAE: {history.history['mae'][-1]:.6f}")
    if num_val > 0 and 'val_loss' in history.history:
        print(f"  Final validation loss: {history.history['val_loss'][-1]:.6f}")
        print(f"  Final validation MAE: {history.history['val_mae'][-1]:.6f}")
    print("=" * 70)
    print()
    print("Next step: Run predict_continuous.py to start making predictions")
    print("and continuously improve the model!")


if __name__ == "__main__":
    # Check for command line arguments
    epochs = 50
    batch_size = 4
    
    if len(sys.argv) > 1:
        try:
            epochs = int(sys.argv[1])
        except ValueError:
            print(f"Invalid epochs value: {sys.argv[1]}")
            sys.exit(1)
    
    if len(sys.argv) > 2:
        try:
            batch_size = int(sys.argv[2])
        except ValueError:
            print(f"Invalid batch_size value: {sys.argv[2]}")
            sys.exit(1)
    
    train_initial_model(epochs=epochs, batch_size=batch_size)
