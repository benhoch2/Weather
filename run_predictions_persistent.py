"""
Persistent wrapper for predict_continuous.py that automatically restarts on crashes.
Logs all errors to help debug issues.
"""
import subprocess
import sys
import os
import time
from datetime import datetime
from pathlib import Path
from process_utils import acquire_lock, release_lock

# ── Duplicate-process guard ───────────────────────────────────────────────────
# If another instance of this script is already running, refuse to start.
# This prevents the "two predictors eating each other's history" bug.

PREDICTOR_LOCK = Path(__file__).parent / "predictor.lock"

log_file = Path("prediction_errors.log")

def log_message(message):
    """Log message to both console and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}\n"
    print(log_line, end='')
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_line)

def run_predictions():
    """Run the prediction script and restart on failure."""
    restart_count = 0

    log_message("=" * 70)
    log_message("Prediction Persistent Runner Started")
    log_message("Will automatically restart on crashes")
    log_message("=" * 70)
    
    while True:
        try:
            restart_count += 1
            log_message(f"\nStarting predict_continuous.py (attempt #{restart_count})")
            
            # Run the prediction script - binary pipe (bufsize=0) bypasses all
            # Python text-layer buffering so every line reaches us immediately.
            process = subprocess.Popen(
                [sys.executable, "-u", "predict_continuous.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                env={**__import__('os').environ, "PYTHONUNBUFFERED": "1", "TF_ENABLE_ONEDNN_OPTS": "0"}
            )
            
            # Stream output in real-time — write raw bytes straight to our
            # stdout buffer so there are zero extra buffering layers.
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
                
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code != 0:
                log_message(f"[WARNING] Process exited with code {return_code}")
            else:
                log_message("Process exited normally")
                
        except KeyboardInterrupt:
            log_message("\n" + "=" * 70)
            log_message("Stopped by user (Ctrl+C)")
            log_message("=" * 70)
            break
            
        except Exception as e:
            log_message(f"[ERROR] {type(e).__name__}: {e}")
            import traceback
            log_message(traceback.format_exc())
        
        # Wait before restart
        wait_time = 10
        log_message(f"Waiting {wait_time} seconds before restart...")
        time.sleep(wait_time)

if __name__ == "__main__":
    acquire_lock(PREDICTOR_LOCK, "PERSISTENT")
    try:
        run_predictions()
    finally:
        release_lock(PREDICTOR_LOCK)
