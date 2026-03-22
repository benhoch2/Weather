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

# ── Duplicate-process guard ───────────────────────────────────────────────────
# If another instance of this script is already running, refuse to start.
# This prevents the "two predictors eating each other's history" bug.

PREDICTOR_LOCK = Path(__file__).parent / "predictor.lock"


def _is_pid_running(pid: int) -> bool:
    """Return True when a **Python** process with this PID is alive."""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if str(pid) in line and "python" in line.lower():
                return True
        return False
    except Exception:
        return False


def acquire_predictor_lock() -> None:
    if PREDICTOR_LOCK.exists():
        try:
            pid = int(PREDICTOR_LOCK.read_text().strip())
            if _is_pid_running(pid):
                print(f"[PERSISTENT] ERROR: A prediction engine is already running (PID {pid}).")
                print(f"[PERSISTENT]   Stop it first, or delete {PREDICTOR_LOCK} if it is stale.")
                sys.exit(1)
            else:
                print(f"[PERSISTENT] Cleaning stale lock (PID {pid} is gone)")
                PREDICTOR_LOCK.unlink(missing_ok=True)
        except (ValueError, SystemExit):
            raise
        except Exception:
            PREDICTOR_LOCK.unlink(missing_ok=True)  # Unreadable — remove
    PREDICTOR_LOCK.write_text(str(os.getpid()))
    print(f"[PERSISTENT] Lock acquired (PID {os.getpid()})")


def release_predictor_lock() -> None:
    try:
        PREDICTOR_LOCK.unlink(missing_ok=True)
    except Exception:
        pass

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
            
            # Run the prediction script with -u for unbuffered output
            process = subprocess.Popen(
                [sys.executable, "-u", "predict_continuous.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**__import__('os').environ, "PYTHONUNBUFFERED": "1", "TF_ENABLE_ONEDNN_OPTS": "0"}
            )
            
            # Stream output in real-time
            for line in process.stdout:
                print(line, end='')
                
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
    acquire_predictor_lock()
    try:
        run_predictions()
    finally:
        release_predictor_lock()
