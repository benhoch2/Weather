import numpy as np
from datetime import datetime
import sys
from data_manager import RadarDataManager
from radar_model import RadarPredictionModel

def train_initial_model(epochs=50, batch_size=4):
    """
    Train the radar prediction model on collected data.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    print("=" * 70)
    print("Radar Prediction Model - Initial Training")
    print("=" * 70)
    print()
    
    # Initialize data manager
    print("Loading data...")
    data_manager = RadarDataManager()
    
    try:
        # Create sequences from collected images
        X, y = data_manager.create_sequences()
        
        num_sequences = len(X)
        print(f"\nTotal sequences: {num_sequences}")
        
        if num_sequences < 10:
            print("\nWarning: Very few sequences available.")
            print("The model will train but may not perform well.")
            print("Collect more images for better results.")
            print()
        
        # Split into train and validation sets (80/20 split)
        split_idx = int(0.8 * num_sequences)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"Training sequences: {len(X_train)}")
        print(f"Validation sequences: {len(X_val)}")
        print()
        
    except ValueError as e:
        print(f"Error: {e}")
        print("\nYou need to collect more radar images first.")
        print("Run fetch_radar_continuous.py and wait for at least 13 images (65 minutes).")
        return
    
    # Build the model
    print("Building model...")
    model = RadarPredictionModel(
        sequence_length=12,
        image_size=(256, 256),
        channels=3
    )
    model.build_model()
    print()
    
    # Print model summary
    print("Model architecture:")
    model.summary()
    print()
    
    # Train the model
    print(f"Starting training for {epochs} epochs...")
    print("-" * 70)
    
    history = model.model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val) if len(X_val) > 0 else None,
        verbose=1
    )
    
    print()
    print("-" * 70)
    print("Training completed!")
    print()
    
    # Evaluate on validation set
    if len(X_val) > 0:
        print("Evaluating on validation set...")
        results = model.evaluate(X_val, y_val)
        print(f"Validation Loss: {results['loss']:.6f}")
        print(f"Validation MAE: {results['mae']:.6f}")
        print()
    
    # Save the model
    model_path = 'radar_model.keras'
    model.save(model_path)
    
    # Save training metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'num_sequences': num_sequences,
        'num_training': len(X_train),
        'num_validation': len(X_val),
        'epochs': epochs,
        'batch_size': batch_size,
        'final_loss': float(history.history['loss'][-1]),
        'final_mae': float(history.history['mae'][-1]),
    }
    
    if len(X_val) > 0:
        metadata['final_val_loss'] = float(history.history['val_loss'][-1])
        metadata['final_val_mae'] = float(history.history['val_mae'][-1])
    
    data_manager.save_metadata(metadata)
    
    print("=" * 70)
    print("Training Summary:")
    print(f"  Model saved: {model_path}")
    print(f"  Training sequences: {len(X_train)}")
    print(f"  Final training loss: {history.history['loss'][-1]:.6f}")
    print(f"  Final training MAE: {history.history['mae'][-1]:.6f}")
    if len(X_val) > 0:
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
