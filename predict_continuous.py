import numpy as np
from PIL import Image
import time
from datetime import datetime
from pathlib import Path
from data_manager import RadarDataManager
from radar_model import RadarPredictionModel

def calculate_image_metrics(predicted, actual):
    """
    Calculate metrics to compare predicted and actual images.
    
    Args:
        predicted: Predicted image array (height, width, channels)
        actual: Actual image array (height, width, channels)
    
    Returns:
        Dictionary of metrics
    """
    mse = np.mean((predicted - actual) ** 2)
    mae = np.mean(np.abs(predicted - actual))
    
    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    if mse > 0:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    else:
        psnr = float('inf')
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'psnr': float(psnr)
    }

def save_prediction_comparison(predicted, actual, timestamp):
    """
    Save a comparison image showing predicted vs actual.
    
    Args:
        predicted: Predicted image array (height, width, channels)
        actual: Actual image array (height, width, channels)
        timestamp: Timestamp for filename
    """
    # Convert to uint8
    pred_img = (predicted * 255).astype(np.uint8)
    actual_img = (actual * 255).astype(np.uint8)
    
    # Create side-by-side comparison
    pred_pil = Image.fromarray(pred_img)
    actual_pil = Image.fromarray(actual_img)
    
    # Create a wider image to hold both
    comparison = Image.new('RGB', (pred_pil.width * 2, pred_pil.height))
    comparison.paste(pred_pil, (0, 0))
    comparison.paste(actual_pil, (pred_pil.width, 0))
    
    # Save
    filename = f"prediction_comparison_{timestamp}.png"
    comparison.save(filename)
    return filename

def continuous_learning():
    """
    Continuously predict the next radar image and update the model.
    
    This script:
    1. Waits for enough images to make a prediction
    2. Predicts the next image
    3. Waits 5 minutes for the actual image
    4. Compares prediction with actual
    5. Updates the model weights
    6. Repeats
    """
    print("=" * 70)
    print("Radar Prediction - Continuous Learning")
    print("=" * 70)
    print()
    
    # Initialize
    data_manager = RadarDataManager()
    model = RadarPredictionModel()
    
    # Try to load existing model
    if not model.load('radar_model.keras'):
        print("No trained model found!")
        print("Please run train_model.py first to train an initial model.")
        print()
        response = input("Do you want to build a new untrained model? (y/n): ")
        if response.lower() != 'y':
            return
        print("\nBuilding new model...")
        model.build_model()
        model.save('radar_model.keras')
    
    print()
    print("Model loaded and ready.")
    print("Waiting for radar images to make predictions...")
    print("Press Ctrl+C to stop")
    print()
    print("-" * 70)
    
    prediction_count = 0
    last_prediction_time = None
    
    try:
        while True:
            # Get the latest sequence
            sequence = data_manager.get_latest_sequence()
            
            if sequence is None:
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp_str}] Waiting for enough images (need 12)...")
                time.sleep(60)  # Check every minute
                continue
            
            # Make prediction
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp_str}] Making prediction #{prediction_count + 1}...")
            
            prediction = model.predict(sequence)
            prediction_timestamp = int(time.time())
            
            # Save the prediction
            pred_filename = f"predicted_{prediction_timestamp}.png"
            pred_img = (prediction * 255).astype(np.uint8)
            Image.fromarray(pred_img).save(pred_filename)
            print(f"  Prediction saved: {pred_filename}")
            
            # Wait 5 minutes for the actual image to be collected
            print("  Waiting 5 minutes for actual image...")
            time.sleep(5 * 60)
            
            # Get the actual image that should have been collected
            images = data_manager.get_all_radar_images()
            if len(images) == 0:
                print("  Warning: No actual image found!")
                continue
            
            # Find the actual image closest to our prediction time + 5 minutes
            target_time = prediction_timestamp + 5 * 60
            actual_image = min(images, key=lambda x: abs(x[0] - target_time))
            
            # Load the actual image
            actual = data_manager.load_image(actual_image[1])
            
            # Calculate metrics
            metrics = calculate_image_metrics(prediction, actual)
            
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp_str}] Prediction evaluation:")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  PSNR: {metrics['psnr']:.2f} dB")
            
            # Save metrics to JSON file for web viewer
            import json
            metrics_file = f"metrics_{prediction_timestamp}.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save comparison image
            comparison_file = save_prediction_comparison(
                prediction, actual, prediction_timestamp
            )
            print(f"  Comparison saved: {comparison_file}")
            
            # Update the model with this new data
            print("  Updating model weights...")
            
            # Get a fresh sequence including the new actual image
            new_sequence = data_manager.get_latest_sequence()
            if new_sequence is not None:
                # The target is the image right after this sequence
                target = np.expand_dims(actual, axis=0)  # Add batch dimension
                
                # Train on this single example (online learning)
                loss = model.train_on_batch(new_sequence, target)
                print(f"  Model updated - Loss: {loss:.6f}")
                
                # Save updated model periodically
                if (prediction_count + 1) % 10 == 0:
                    model.save('radar_model.keras')
                    print("  Model saved to disk")
            
            prediction_count += 1
            print(f"\n  Total predictions made: {prediction_count}")
            print("-" * 70)
            
    except KeyboardInterrupt:
        print("\n")
        print("=" * 70)
        print("Stopping continuous learning...")
        
        # Save the final model
        model.save('radar_model.keras')
        print(f"Model saved with {prediction_count} predictions")
        print("=" * 70)


if __name__ == "__main__":
    continuous_learning()
