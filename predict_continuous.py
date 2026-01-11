import numpy as np
from PIL import Image
import time
from datetime import datetime
from pathlib import Path
import json
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
    filename = f"data/predictions/prediction_comparison_{timestamp}.png"
    comparison.save(filename)
    return filename

def create_animated_comparison(predicted_frames, actual_frames, timestamp):
    """
    Create an animated GIF showing predicted vs actual frames.
    
    Args:
        predicted_frames: List of predicted image arrays
        actual_frames: List of actual image arrays
        timestamp: Timestamp for filename
    """
    comparison_frames = []
    
    for pred, actual in zip(predicted_frames, actual_frames):
        # Convert to uint8
        pred_img = (pred * 255).astype(np.uint8)
        actual_img = (actual * 255).astype(np.uint8)
        
        # Create side-by-side comparison
        pred_pil = Image.fromarray(pred_img)
        actual_pil = Image.fromarray(actual_img)
        
        # Create a wider image to hold both
        comparison = Image.new('RGB', (pred_pil.width * 2, pred_pil.height))
        comparison.paste(pred_pil, (0, 0))
        comparison.paste(actual_pil, (pred_pil.width, 0))
        
        comparison_frames.append(comparison)
    
    # Save as animated GIF
    filename = f"data/predictions/prediction_animation_{timestamp}.gif"
    comparison_frames[0].save(
        filename,
        save_all=True,
        append_images=comparison_frames[1:],
        duration=500,  # 500ms per frame
        loop=0  # Loop forever
    )
    return filename

def cleanup_old_predictions():
    """
    Keep only the last 10 predictions, delete older ones.
    This includes metrics JSON files, comparison GIFs, prediction-only GIFs, and all frame images.
    """
    try:
        predictions_dir = Path("data/predictions")
        if not predictions_dir.exists():
            return
        
        # Get all prediction animations sorted by timestamp (newest first)
        animations = sorted(
            predictions_dir.glob("prediction_animation_*.gif"),
            key=lambda p: int(p.stem.split('_')[-1]),
            reverse=True
        )
        
        # Keep only the 10 most recent timestamps
        if len(animations) > 10:
            keep_timestamps = set()
            for anim in animations[:10]:
                timestamp = anim.stem.split('_')[-1]
                keep_timestamps.add(timestamp)
            
            # Delete ALL files not associated with the kept timestamps
            deleted_count = 0
            
            # Delete old prediction animations
            for anim in animations[10:]:
                anim.unlink()
                deleted_count += 1
            
            # Delete all prediction_only GIFs (not needed for web viewer)
            for pred_only in predictions_dir.glob("prediction_only_*.gif"):
                timestamp = pred_only.stem.split('_')[-1]
                if timestamp not in keep_timestamps:
                    pred_only.unlink()
                    deleted_count += 1
            
            # Delete old comparison PNGs
            for comp_file in predictions_dir.glob("prediction_comparison_*.png"):
                timestamp = comp_file.stem.split('_')[-1]
                if timestamp not in keep_timestamps:
                    comp_file.unlink()
                    deleted_count += 1
            
            # Delete old metrics JSON
            for metrics_file in predictions_dir.glob("metrics_*.json"):
                timestamp = metrics_file.stem.split('_')[1].replace('.json', '')
                if timestamp not in keep_timestamps:
                    metrics_file.unlink()
                    deleted_count += 1
            
            # Delete old individual prediction frames
            for pred_file in predictions_dir.glob("predicted_*.png"):
                # Extract timestamp from predicted_TIMESTAMP_frame1.png
                parts = pred_file.stem.split('_')
                if len(parts) >= 2:
                    timestamp = parts[1]
                    if timestamp not in keep_timestamps:
                        pred_file.unlink()
                        deleted_count += 1
            
            if deleted_count > 0:
                print(f"  [CLEANUP] Cleaned up {deleted_count} old file(s)")
    except Exception as e:
        print(f"  [WARNING] Error during predictions cleanup: {e}")

def continuous_learning():
    """
    Continuously predict the next 5 radar images every 5 minutes.
    
    This creates a rolling prediction window:
    - Every 5 minutes: Make new 5-frame prediction
    - Track multiple predictions in flight
    - When actual data arrives, compare and update model
    
    At any time, there are 5 live predictions being tracked.
    """
    print("=" * 70)
    print("Radar Prediction - Continuous Learning (Rolling 5-Frame)")
    print("=" * 70)
    print()
    
    # Initialize
    data_manager = RadarDataManager(data_dir="data/radar_images")
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
    print("Making predictions every 5 minutes...")
    print("Each prediction covers next 25 minutes (5 frames)")
    print("Press Ctrl+C to stop")
    print()
    print("-" * 70)
    
    prediction_count = 0
    pending_predictions = []  # List of (timestamp, predicted_frames) tuples
    
    try:
        while True:
            try:
                # Get the latest sequence
                sequence = data_manager.get_latest_sequence()
                
                if sequence is None:
                    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp_str}] Waiting for enough images (need 12)...")
                    time.sleep(60)  # Check every minute
                    continue
                
                # Make 5 predictions recursively
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{timestamp_str}] Making 5-frame prediction #{prediction_count + 1}...")
                
                prediction_timestamp = int(time.time())
                predicted_frames = []
                current_sequence = sequence.copy()
                
                for frame_num in range(5):
                    # Predict next frame
                    prediction = model.predict(current_sequence)
                    predicted_frames.append(prediction)
                    
                    # Update sequence for next prediction: remove oldest, add prediction
                    new_frame = np.expand_dims(prediction, axis=0)  # (1, 512, 512, 3)
                    new_frame = np.expand_dims(new_frame, axis=1)   # (1, 1, 512, 512, 3)
                    current_sequence = np.concatenate([current_sequence[:, 1:, :, :, :], new_frame], axis=1)
                
                # Save individual frames
                for frame_num, pred_frame in enumerate(predicted_frames):
                    pred_filename = f"data/predictions/predicted_{prediction_timestamp}_frame{frame_num+1}.png"
                    pred_img = (pred_frame * 255).astype(np.uint8)
                    Image.fromarray(pred_img).save(pred_filename)
                
                # Create prediction-only animated GIF
                prediction_frames_pil = []
                for pred_frame in predicted_frames:
                    pred_img = (pred_frame * 255).astype(np.uint8)
                    prediction_frames_pil.append(Image.fromarray(pred_img))
                
                pred_only_filename = f"data/predictions/prediction_only_{prediction_timestamp}.gif"
                prediction_frames_pil[0].save(
                    pred_only_filename,
                    save_all=True,
                    append_images=prediction_frames_pil[1:],
                    duration=500,
                    loop=0
                )
                
                # Create immediate "preview" comparison GIF (prediction-only, will be replaced later)
                comparison_preview_frames = []
                for pred_frame in predicted_frames:
                    pred_img = (pred_frame * 255).astype(np.uint8)
                    pred_pil = Image.fromarray(pred_img)
                    
                    # Create placeholder - just show prediction on left, gray placeholder on right
                    placeholder = Image.new('RGB', (512, 512), color=(100, 100, 100))
                    comparison = Image.new('RGB', (pred_pil.width * 2, pred_pil.height))
                    comparison.paste(pred_pil, (0, 0))
                    comparison.paste(placeholder, (pred_pil.width, 0))
                    comparison_preview_frames.append(comparison)
                
                preview_filename = f"data/predictions/prediction_animation_{prediction_timestamp}.gif"
                comparison_preview_frames[0].save(
                    preview_filename,
                    save_all=True,
                    append_images=comparison_preview_frames[1:],
                    duration=500,
                    loop=0,
                    optimize=False
                )
                
                # Create placeholder metrics JSON
                placeholder_metrics = {
                    'frames': [{'mse': 0, 'mae': 0, 'psnr': 0} for _ in range(5)],
                    'average': {'mse': 0, 'mae': 0, 'psnr': 0},
                    'pending': True
                }
                metrics_file = f"data/predictions/metrics_{prediction_timestamp}.json"
                with open(metrics_file, 'w') as f:
                    json.dump(placeholder_metrics, f, indent=2)
                
                print(f"  All 5 frames predicted and saved")
                print(f"  Prediction animation: {pred_only_filename}")
                print(f"  Preview comparison: {preview_filename} (will update with actual in 25 min)")
                
                # Add to pending predictions
                pending_predictions.append((prediction_timestamp, predicted_frames))
                
                # Clean up old files - run every cycle to keep disk usage minimal
                cleanup_old_predictions()
                
                # Check if any predictions are ready to be evaluated (25 minutes old)
                current_time = int(time.time())
                ready_predictions = [(ts, frames) for ts, frames in pending_predictions 
                                    if current_time >= ts + 25 * 60]
                
                # Evaluate ready predictions
                for pred_timestamp, pred_frames in ready_predictions:
                    print(f"\n  Evaluating prediction from {datetime.fromtimestamp(pred_timestamp).strftime('%H:%M:%S')}...")
                    
                    # Get actual images
                    images = data_manager.get_all_radar_images()
                    if len(images) == 0:
                        continue
                    
                    actual_frames = []
                    frame_metrics = []
                    
                    for frame_num in range(5):
                        # Find actual image for this frame
                        target_time = pred_timestamp + (frame_num + 1) * 5 * 60
                        actual_image = min(images, key=lambda x: abs(x[0] - target_time))
                        
                        # Load the actual image
                        actual = data_manager.load_image(actual_image[1])
                        actual_frames.append(actual)
                        
                        # Calculate metrics
                        metrics = calculate_image_metrics(pred_frames[frame_num], actual)
                        frame_metrics.append(metrics)
                    
                    # Calculate averages
                    avg_mse = sum(m['mse'] for m in frame_metrics) / len(frame_metrics)
                    avg_mae = sum(m['mae'] for m in frame_metrics) / len(frame_metrics)
                    avg_psnr = sum(m['psnr'] for m in frame_metrics) / len(frame_metrics)
                    
                    print(f"  Avg MSE: {avg_mse:.6f}, MAE: {avg_mae:.6f}, PSNR: {avg_psnr:.2f} dB")
                    
                    # Save metrics
                    all_metrics = {
                        'frames': frame_metrics,
                        'average': {
                            'mse': avg_mse,
                            'mae': avg_mae,
                            'psnr': avg_psnr
                        }
                    }
                    metrics_file = f"data/predictions/metrics_{pred_timestamp}.json"
                    with open(metrics_file, 'w') as f:
                        json.dump(all_metrics, f, indent=2)
                    
                    # Update comparison animation with actual data
                    animation_file = create_animated_comparison(
                        pred_frames, actual_frames, pred_timestamp
                    )
                    
                    # Create static comparison for backward compatibility
                    comparison_file = save_prediction_comparison(
                        pred_frames[0], actual_frames[0], pred_timestamp
                    )
                    
                    print(f"  ✓ Comparison updated with actual data: {animation_file}")
                    
                    # Update model with first frame
                    new_sequence = data_manager.get_latest_sequence()
                    if new_sequence is not None:
                        target = np.expand_dims(actual_frames[0], axis=0)
                        result = model.train_on_batch(new_sequence, target)
                        loss_value = result[0] if isinstance(result, list) else result
                        print(f"  Model updated - Loss: {loss_value:.6f}")
                    
                    # Remove from pending
                    pending_predictions.remove((pred_timestamp, pred_frames))
                
                prediction_count += 1
                print(f"\n  Total predictions: {prediction_count}")
                print(f"  Pending evaluations: {len(pending_predictions)}")
                print("-" * 70)
                
                # Wait 5 minutes before next prediction
                time.sleep(5 * 60)
                
            except Exception as e:
                # Don't crash on individual prediction errors
                print(f"\n  [ERROR] during prediction cycle: {e}")
                print(f"  Traceback: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                print(f"  Waiting 60 seconds before retry...")
                print("-" * 70)
                time.sleep(60)  # Wait a minute before trying again
            
    except KeyboardInterrupt:
        print("\n")
        print("=" * 70)
        print("Stopping continuous learning...")
        
        # Save the final model
        model.save('radar_model.keras')
        print(f"Model saved with {prediction_count} prediction cycles")
        print("=" * 70)


if __name__ == "__main__":
    continuous_learning()
