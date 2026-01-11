"""
Persistent wrapper for predict_continuous.py that automatically restarts on crashes.
Logs all errors to help debug issues.
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

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
            
            # Run the prediction script
            process = subprocess.Popen(
                [sys.executable, "predict_continuous.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Stream output in real-time
            for line in process.stdout:
                print(line, end='')
                
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code != 0:
                log_message(f"⚠️  Process exited with code {return_code}")
            else:
                log_message("Process exited normally")
                
        except KeyboardInterrupt:
            log_message("\n" + "=" * 70)
            log_message("Stopped by user (Ctrl+C)")
            log_message("=" * 70)
            break
            
        except Exception as e:
            log_message(f"⚠️  ERROR: {type(e).__name__}: {e}")
            import traceback
            log_message(traceback.format_exc())
        
        # Wait before restart
        wait_time = 10
        log_message(f"Waiting {wait_time} seconds before restart...")
        time.sleep(wait_time)

if __name__ == "__main__":
    run_predictions()
