import os
# Disable oneDNN/MKL optimizations: with them enabled, TF uses the mklcpu
# allocator which can OOM mid-operation and then hit a C++ dtype CHECK failure,
# crashing the process with an unrecoverable abort. Without oneDNN, TF's own
# BFC allocator handles OOM by raising a Python ResourceExhaustedError instead.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import gc
import re
import numpy as np
from PIL import Image
import time
from datetime import datetime
from pathlib import Path
import json
from data_manager import RadarDataManager
from radar_model import RadarPredictionModel

_STATUS_FILE = Path("data/predictor_status.json")

def _write_predictor_status(status: str, **extra):
    """Atomically write predictor status JSON so downstream tools (web UI) can read it."""
    payload = {"status": status, "updated_at": int(time.time())}
    payload.update(extra)
    try:
        _STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATUS_FILE.write_text(json.dumps(payload))
    except Exception:
        pass

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

def cleanup_old_predictions(pending_timestamps=None):
    """
    Keep only the last 10 predictions, delete older ones.
    This includes metrics JSON files, comparison GIFs, prediction-only GIFs, and all frame images.
    Predictions still pending evaluation are always protected from deletion.
    """
    try:
        predictions_dir = Path("data/predictions")
        if not predictions_dir.exists():
            return
        
        # Timestamps that must never be deleted (pending evaluation)
        protected = set(str(ts) for ts in (pending_timestamps or []))
        
        # Get all prediction animations sorted by timestamp (newest first)
        animations = sorted(
            predictions_dir.glob("prediction_animation_*.gif"),
            key=lambda p: int(p.stem.split('_')[-1]),
            reverse=True
        )
        
        # Keep only the 20 most recent timestamps
        max_keep = 20
        if len(animations) > max_keep:
            keep_timestamps = set()
            for anim in animations[:max_keep]:
                timestamp = anim.stem.split('_')[-1]
                keep_timestamps.add(timestamp)
            
            # Always protect pending predictions
            keep_timestamps |= protected
            
            # Delete ALL files not associated with the kept timestamps
            deleted_count = 0
            
            # Delete old prediction animations
            for anim in animations[max_keep:]:
                ts = anim.stem.split('_')[-1]
                if ts not in keep_timestamps:
                    anim.unlink()
                    deleted_count += 1
            
            # Delete prediction_only GIFs not in the keep set
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

    run_start_timestamp = int(time.time())
    # Allow frames captured up to 1 hour before this process started.
    # This lets the fetcher warm up before the predictor is launched without
    # treating those recent frames as stale. Months-old frames are still excluded.
    fresh_cutoff = run_start_timestamp - 3600
    run_start_str = datetime.fromtimestamp(run_start_timestamp).strftime("%Y-%m-%d %H:%M:%S")
    
    # Initialize
    print("Initializing data manager...", flush=True)
    data_manager = RadarDataManager(data_dir="data/radar_images")
    model = RadarPredictionModel()
    
    print("Loading model...", flush=True)
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
    
    # Lower the learning rate for online single-sample updates
    model.model.optimizer.learning_rate.assign(0.0001)
    print("  Online learning rate set to 0.0001")

    print()
    print("Model loaded and ready.", flush=True)
    print("Making predictions every 5 minutes...", flush=True)
    print("Each prediction covers next 25 minutes (5 frames)")
    print(f"Waiting for 12 new frames captured after: {run_start_str}")
    print("Press Ctrl+C to stop")
    print()
    print("-" * 70)
    
    prediction_count = 0
    train_count = 0  # Number of online learning updates since launch

    # Recover pending predictions from disk that weren't evaluated before a restart.
    # Any metrics file with "pending": true whose predicted frames still exist on disk
    # is re-queued so evaluation resumes without losing the prediction from history.
    # Exception: if the prediction is already past its evaluation window AND actual
    # frames are missing (stuck-radar gap), mark it skipped immediately rather than
    # re-queuing it to fail again on every future restart.
    pending_predictions = []
    _predictions_dir = Path("data/predictions")
    if _predictions_dir.exists():
        # Pre-load available radar timestamps once for fast lookup
        _available_ts = sorted(
            int(re.search(r'radar_(\d+)\.png', p.name).group(1))
            for p in Path("data/radar_images").glob("radar_*.png")
            if re.search(r'radar_(\d+)\.png', p.name)
        ) if Path("data/radar_images").exists() else []

        for _mf in sorted(_predictions_dir.glob("metrics_*.json")):
            try:
                _data = json.loads(_mf.read_text())
                if _data.get('pending'):
                    _m = re.search(r'metrics_(\d+)\.json', _mf.name)
                    if not _m:
                        continue
                    _ts = int(_m.group(1))
                    # Skip if predicted frames no longer exist
                    if not (_predictions_dir / f"predicted_{_ts}_frame1.png").exists():
                        continue
                    # If window has passed, check whether actual frames exist.
                    # If every one of the 5 target slots is more than 3 min from any
                    # stored image, the data is permanently missing — mark skipped now.
                    _current = int(time.time())
                    if _current >= _ts + 25 * 60 and _available_ts:
                        _unevaluable = False
                        for _fn in range(1, 6):
                            _target = _ts + _fn * 5 * 60
                            _nearest = min(_available_ts, key=lambda x: abs(x - _target))
                            if abs(_nearest - _target) > 3 * 60:
                                _unevaluable = True
                                break
                        if _unevaluable:
                            print(f"  [RECOVERY] Skipping unevaluable prediction from "
                                  f"{datetime.fromtimestamp(_ts).strftime('%Y-%m-%d %H:%M:%S')} "
                                  f"(actual frames missing — stuck radar gap)",
                                  flush=True)
                            with open(_mf, 'w') as _f:
                                json.dump({'skipped': True, 'reason': 'missing_actual_frames'}, _f)
                            continue
                    pending_predictions.append(_ts)
                    print(f"  [RECOVERY] Recovered pending prediction from "
                          f"{datetime.fromtimestamp(_ts).strftime('%Y-%m-%d %H:%M:%S')}",
                          flush=True)
            except Exception:
                pass
    if pending_predictions:
        print(f"  [RECOVERY] {len(pending_predictions)} pending prediction(s) re-queued for evaluation",
              flush=True)
    del _predictions_dir
    
    try:
        while True:
            try:
                cycle_start = time.time()

                # Validate the 12-frame input using 5-minute slot analysis.
                # Slots are aligned to wall-clock boundaries (XX:00, XX:05, …)
                # matching the fetcher's schedule. Need at least 11/12 filled.
                now = int(time.time())
                slot_size = 5 * 60
                latest_slot = (now // slot_size) * slot_size  # e.g. 22:00 when now=22:01:42
                all_images = data_manager.get_all_radar_images()

                filled_slots = 0
                missing_slot_times = []  # human-readable times for empty slots
                slot_images = []  # best image per slot (for building sequence)
                for slot_idx in range(12):
                    slot_end = latest_slot - slot_idx * slot_size
                    slot_start = slot_end - slot_size
                    candidates = [img for img in all_images
                                  if slot_start < img[0] <= slot_end]
                    if candidates:
                        filled_slots += 1
                        # Pick the one closest to the slot centre
                        slot_centre = (slot_start + slot_end) / 2
                        best = min(candidates, key=lambda x: abs(x[0] - slot_centre))
                        slot_images.append(best)
                    else:
                        slot_images.append(None)
                        # Label with the slot's expected time (its end boundary)
                        missing_slot_times.append(
                            datetime.fromtimestamp(slot_end).strftime("%H:%M"))

                # slot_images[0] is the newest slot, [11] is the oldest.
                # Reverse so index 0 = oldest (chronological order for the model).
                slot_images.reverse()
                empty_slots = 12 - filled_slots

                if empty_slots > 1:
                    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\n[{timestamp_str}] [WARNING] Only {filled_slots}/12 time slots "
                          f"filled in the last hour ({empty_slots} empty) — need at least 11. "
                          f"Missing: {', '.join(missing_slot_times)}. "
                          f"Skipping prediction.", flush=True)
                    # Compute ETA: the oldest missing slot will slide out of the
                    # 12-slot window as newer wall-clock images accumulate.
                    # Parse the oldest missing slot to determine when it exits.
                    today = datetime.now().date()
                    oldest_missing_dt = None
                    for ms in missing_slot_times:
                        h, m = map(int, ms.split(':'))
                        dt = datetime.combine(today, datetime.min.time()).replace(hour=h, minute=m)
                        if oldest_missing_dt is None or dt < oldest_missing_dt:
                            oldest_missing_dt = dt
                    if oldest_missing_dt:
                        # The slot exits the window 60 min after its boundary
                        recovery_time = oldest_missing_dt.timestamp() + 3600
                        eta_min = max(5, int((recovery_time - time.time()) / 60) + 1)
                    else:
                        eta_min = (empty_slots - 1) * 5
                    eta_wall = datetime.fromtimestamp(
                        ((time.time() + eta_min * 60) // 300 + 1) * 300
                    ).strftime("%H:%M")
                    _write_predictor_status(
                        "stuck",
                        frames_available=filled_slots,
                        frames_needed=empty_slots - 1,
                        eta_minutes=eta_min,
                        eta_wall=eta_wall,
                        missing_slots=missing_slot_times,
                    )
                    made_prediction = False
                else:
                    # Fill any single gap by duplicating the nearest neighbour
                    for i in range(12):
                        if slot_images[i] is None:
                            # Find nearest filled slot
                            neighbour = None
                            for offset in range(1, 12):
                                if i + offset < 12 and slot_images[i + offset] is not None:
                                    neighbour = slot_images[i + offset]
                                    break
                                if i - offset >= 0 and slot_images[i - offset] is not None:
                                    neighbour = slot_images[i - offset]
                                    break
                            slot_images[i] = neighbour

                    seq_images = [img for img in slot_images if img is not None]

                    if len(seq_images) < 12:
                        # Shouldn't happen, but guard against it
                        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"\n[{timestamp_str}] [WARNING] Could not build 12-frame sequence "
                              f"after gap-filling. Skipping.", flush=True)
                        made_prediction = False
                    else:
                        newest_ts = seq_images[-1][0]
                        newest_age = int((time.time() - newest_ts) / 60)
                        gap_note = " (1 slot gap-filled)" if empty_slots == 1 else ""
                        print(f"  [DATA] Loaded 12-frame sequence from last hour{gap_note}  "
                              f"(newest frame: {datetime.fromtimestamp(newest_ts).strftime('%H:%M:%S')}, "
                              f"{newest_age}m ago)", flush=True)

                        input_sequence = []
                        for img_info in seq_images:
                            input_sequence.append(data_manager.load_image(img_info[1]))
                        sequence = np.array([input_sequence], dtype=np.float32)
                        del input_sequence
                        made_prediction = None  # will be set below

                if made_prediction is False:
                    pass  # skip to evaluation
                elif made_prediction is None:
                    # Make 5 predictions recursively
                    _write_predictor_status("predicting")
                    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\n[{timestamp_str}] Making 5-frame prediction #{prediction_count + 1}...")
                    
                    prediction_timestamp = int(time.time())
                    predicted_frames_pil = []
                    current_sequence = sequence.copy()
                    
                    for frame_num in range(5):
                        # Predict next frame
                        _frame_t = time.time()
                        print(f"  [PREDICT] Frame {frame_num+1}/5...", end=" ", flush=True)
                        prediction = model.predict(current_sequence)
                        print(f"done ({time.time()-_frame_t:.1f}s)", flush=True)

                        # Save individual frame to disk immediately
                        pred_filename = f"data/predictions/predicted_{prediction_timestamp}_frame{frame_num+1}.png"
                        pred_img = (prediction * 255).astype(np.uint8)
                        pred_pil = Image.fromarray(pred_img)
                        pred_pil.save(pred_filename)
                        predicted_frames_pil.append(pred_pil)

                        # Update sequence for next prediction: remove oldest, add prediction
                        new_frame = np.expand_dims(prediction, axis=0)  # (1, 512, 512, 3)
                        new_frame = np.expand_dims(new_frame, axis=1)   # (1, 1, 512, 512, 3)
                        current_sequence = np.concatenate([current_sequence[:, 1:, :, :, :], new_frame], axis=1)
                        del prediction, pred_img, new_frame
                    
                    # Free the large sequence arrays
                    del current_sequence, sequence
                    
                    # Create prediction-only animated GIF
                    prediction_frames_pil = predicted_frames_pil
                    
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
                    for pred_pil in prediction_frames_pil:
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
                    
                    # Add to pending predictions (only timestamp, frames are on disk)
                    pending_predictions.append(prediction_timestamp)
                    
                    # Free PIL frames
                    del prediction_frames_pil, predicted_frames_pil, comparison_preview_frames
                    gc.collect()
                    
                    # Clean up old files - run every cycle to keep disk usage minimal
                    cleanup_old_predictions(pending_timestamps=pending_predictions)

                    prediction_count += 1
                    made_prediction = True

                # ── Evaluation runs every cycle, whether or not a prediction was made ──
                # This ensures pending predictions are evaluated on time even when
                # the prediction step was skipped (e.g. stuck radar).
                current_time = int(time.time())
                ready_predictions = [ts for ts in pending_predictions
                                    if current_time >= ts + 25 * 60]

                if len(ready_predictions) > 1:
                    print(f"  [EVAL] {len(ready_predictions)} evaluations ready — draining skipped, then 1 successful per cycle", flush=True)

                # Skipped evaluations (missing actual frames) cost nothing — drain them all.
                # Stop after 1 successful evaluation that triggers model training.
                successful_evals = 0
                for pred_timestamp in ready_predictions:
                    if successful_evals >= 1:
                        break
                    print(f"\n  Evaluating prediction from {datetime.fromtimestamp(pred_timestamp).strftime('%H:%M:%S')}...")
                    
                    # Reload predicted frames from disk
                    print(f"  [EVAL] Loading 5 predicted frames from disk...", flush=True)
                    pred_frames = []
                    frames_found = True
                    for frame_num in range(5):
                        pred_path = Path(f"data/predictions/predicted_{pred_timestamp}_frame{frame_num+1}.png")
                        if not pred_path.exists():
                            print(f"  [WARNING] Missing predicted frame: {pred_path.name}, skipping evaluation")
                            frames_found = False
                            break
                        pred_frames.append(data_manager.load_image(pred_path))
                    if frames_found:
                        print(f"  [EVAL] Frames loaded OK", flush=True)
                    
                    if not frames_found:
                        pending_predictions.remove(pred_timestamp)
                        continue
                    
                    # Get actual images
                    images = data_manager.get_all_radar_images()
                    if len(images) == 0:
                        continue
                    
                    actual_frames = []
                    frame_metrics = []
                    skipped_frames = []
                    
                    for frame_num in range(5):
                        # Find actual image for this frame
                        target_time = pred_timestamp + (frame_num + 1) * 5 * 60
                        actual_image = min(images, key=lambda x: abs(x[0] - target_time))
                        
                        time_off = abs(actual_image[0] - target_time)
                        if time_off > 5 * 60:
                            # No frame within 5 minutes of target — skip this individual frame
                            print(f"  [EVAL] Frame {frame_num+1}: nearest actual is {time_off}s off target "
                                  f"(>{5*60}s) — skipping this frame", flush=True)
                            skipped_frames.append(frame_num)
                            # Use grey placeholder for the comparison GIF
                            actual_frames.append(None)
                            frame_metrics.append(None)
                        else:
                            if time_off > 3 * 60:
                                print(f"  [EVAL] Frame {frame_num+1}: nearest actual is {time_off}s off target "
                                      f"(using best available)", flush=True)
                            actual = data_manager.load_image(actual_image[1])
                            actual_frames.append(actual)
                            metrics = calculate_image_metrics(pred_frames[frame_num], actual)
                            frame_metrics.append(metrics)
                    
                    # Count how many frames we got
                    valid_metrics = [m for m in frame_metrics if m is not None]
                    
                    if len(valid_metrics) == 0:
                        # No usable frames at all
                        print(f"  [WARNING] All 5 actual frames missing — skipping evaluation")
                        pending_predictions.remove(pred_timestamp)
                        try:
                            metrics_file = f"data/predictions/metrics_{pred_timestamp}.json"
                            with open(metrics_file, 'w') as f:
                                json.dump({'skipped': True, 'reason': 'missing_actual_frames'}, f)
                        except Exception:
                            pass
                        del pred_frames
                        gc.collect()
                        continue
                    
                    if skipped_frames:
                        print(f"  [EVAL] {len(skipped_frames)} frame(s) skipped, "
                              f"evaluating with {len(valid_metrics)}/5 frames", flush=True)
                    
                    # Calculate averages (only over valid frames)
                    avg_mse = sum(m['mse'] for m in valid_metrics) / len(valid_metrics)
                    avg_mae = sum(m['mae'] for m in valid_metrics) / len(valid_metrics)
                    avg_psnr = sum(m['psnr'] for m in valid_metrics) / len(valid_metrics)
                    
                    print(f"  Avg MSE: {avg_mse:.6f}, MAE: {avg_mae:.6f}, PSNR: {avg_psnr:.2f} dB")
                    
                    # Save metrics (None entries for skipped frames become null in JSON)
                    all_metrics = {
                        'frames': [m if m is not None else {'skipped': True} for m in frame_metrics],
                        'average': {
                            'mse': avg_mse,
                            'mae': avg_mae,
                            'psnr': avg_psnr
                        },
                        'evaluated_frames': len(valid_metrics),
                        'total_frames': 5
                    }
                    metrics_file = f"data/predictions/metrics_{pred_timestamp}.json"
                    with open(metrics_file, 'w') as f:
                        json.dump(all_metrics, f, indent=2)
                    
                    # Build actual frames list for comparison GIF, using grey
                    # placeholders for any skipped frames.
                    actual_for_gif = []
                    for af in actual_frames:
                        if af is not None:
                            actual_for_gif.append(af)
                        else:
                            actual_for_gif.append(np.full((512, 512, 3), 100.0/255.0, dtype=np.float32))

                    # Update comparison animation with actual data
                    animation_file = create_animated_comparison(
                        pred_frames, actual_for_gif, pred_timestamp
                    )
                    
                    # Create static comparison for backward compatibility
                    # Use the first valid actual frame
                    first_actual = next((af for af in actual_frames if af is not None), actual_for_gif[0])
                    comparison_file = save_prediction_comparison(
                        pred_frames[0], first_actual, pred_timestamp
                    )
                    
                    print(f"  [SUCCESS] Comparison updated with actual data: {animation_file}")
                    successful_evals += 1
                    
                    # Update model with first available actual frame
                    first_valid_actual = next((af for af in actual_frames if af is not None), None)
                    new_sequence = data_manager.get_sequence_before_timestamp(pred_timestamp, min_timestamp=fresh_cutoff)
                    if new_sequence is not None and first_valid_actual is not None:
                        target = np.expand_dims(first_valid_actual, axis=0)
                        try:
                            result = model.train_on_batch(new_sequence, target)
                            loss_value = result[0] if isinstance(result, list) else result
                            print(f"  Model updated - Loss: {loss_value:.6f}")
                            train_count += 1
                            if train_count % 5 == 0:
                                model.save('radar_model.keras')
                                print(f"  [SAVE] Model checkpoint (after {train_count} updates)")
                        except Exception as train_err:
                            print(f"  [WARNING] Skipping model update (training failed: {train_err})")
                        del new_sequence, target
                    
                    # Free evaluation arrays
                    del pred_frames, actual_frames
                    gc.collect()
                    
                    # Remove from pending
                    pending_predictions.remove(pred_timestamp)

                # ── Stats + sleep — always runs ──
                print(f"\n  Total predictions: {prediction_count}", flush=True)
                print(f"  Pending evaluations: {len(pending_predictions)}", flush=True)
                print("-" * 70, flush=True)
                
                # Sleep only the remaining time to keep a strict 5-minute cycle
                elapsed = time.time() - cycle_start
                sleep_for = max(0, 5 * 60 - elapsed)
                if sleep_for > 0:
                    sleep_mins = int(sleep_for // 60)
                    sleep_secs = int(sleep_for % 60)
                    print(f"  [SLEEP] Cycle took {elapsed:.0f}s  —  sleeping {sleep_mins}m {sleep_secs:02d}s until next cycle", flush=True)
                    time.sleep(sleep_for)
                else:
                    print(f"  [SLEEP] Cycle took {elapsed:.0f}s  —  no sleep needed (running behind)", flush=True)
                
            except Exception as e:
                # Don't crash on individual prediction errors
                print(f"\n  [ERROR] during prediction cycle: {e}")
                print(f"  Traceback: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                gc.collect()
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
