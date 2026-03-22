from flask import Flask, render_template, jsonify, send_file, request
from pathlib import Path
import json
from datetime import datetime
import re
import subprocess
import os
import signal
import sys
import time
from process_utils import is_pid_running

app = Flask(__name__)

DEFAULT_PORT = 5050

# Global process tracking
prediction_process = None
prediction_running = False
prediction_start_time = None

# ── Lock-file helpers (mirrors run_predictions_persistent.py) ─────────────────
# When the app is launched via start.py the predictor is NOT a child of this
# web_viewer process, so the globals above never get set.  We check the lock
# file so all status/start/stop/readiness endpoints work correctly regardless
# of how the predictor was started.

PREDICTOR_LOCK = Path(__file__).parent / "predictor.lock"


def _lock_info():
    """
    Return (pid, started_at) from predictor.lock, or (None, None) if absent/stale.
    started_at is the lock file's modification time (unix timestamp).
    """
    try:
        if not PREDICTOR_LOCK.exists():
            return None, None
        pid = int(PREDICTOR_LOCK.read_text().strip())
        if not is_pid_running(pid):
            return None, None  # Stale lock
        started_at = int(PREDICTOR_LOCK.stat().st_mtime)
        return pid, started_at
    except Exception:
        return None, None

class PredictionViewer:
    """
    Manages viewing of radar predictions and comparisons.
    """
    
    def __init__(self, data_dir="data/predictions"):
        self.data_dir = Path(data_dir)
    
    def get_all_predictions(self):
        """
        Get all predictions with their metadata.
        Returns both pending (prediction-only) and completed (with actuals).
        Backward compatible with old prediction format.
        
        Returns:
            List of dictionaries containing prediction info
        """
        predictions = []
        
        # Find all prediction-only GIFs (new format - active predictions)
        for file in sorted(self.data_dir.glob("prediction_only_*.gif"), reverse=True):
            try:
                match = re.search(r'prediction_only_(\d+)\.gif', file.name)
                if not match:
                    continue
                
                timestamp = int(match.group(1))
                current_time = int(datetime.now().timestamp())
                
                # Check if evaluation is complete (comparison file exists AND metrics don't have pending flag)
                comparison_file = self.data_dir / f"prediction_animation_{timestamp}.gif"
                metrics = self.load_metrics(timestamp)
                
                # Only include if comparison is ready AND not pending (no grey placeholders)
                is_evaluated = comparison_file.exists() and metrics and not metrics.get('pending', False)
                
                # Calculate time remaining until evaluation (25 minutes from prediction)
                eval_time = timestamp + 25 * 60
                time_remaining = max(0, eval_time - current_time)
                minutes_remaining = time_remaining // 60
                
                # Only include predictions where comparison is ready (no grey placeholders)
                if is_evaluated:
                    prediction_info = {
                        'timestamp': timestamp,
                        'datetime': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                        'prediction_file': file.name,
                        'is_active': False,
                        'minutes_remaining': 0,
                        'comparison_file': comparison_file.name,
                        'metrics': metrics
                    }
                    predictions.append(prediction_info)
            except (ValueError, IndexError):
                continue
        
        # Also find old format predictions (comparison animations)
        for file in sorted(self.data_dir.glob("prediction_animation_*.gif"), reverse=True):
            try:
                match = re.search(r'prediction_animation_(\d+)\.gif', file.name)
                if not match:
                    continue
                
                timestamp = int(match.group(1))
                
                # Skip if already added from prediction_only
                if any(p['timestamp'] == timestamp for p in predictions):
                    continue
                
                # Check metrics to ensure it's not pending
                metrics = self.load_metrics(timestamp)
                if metrics and metrics.get('pending', False):
                    continue  # Skip pending predictions with grey placeholders
                
                prediction_info = {
                    'timestamp': timestamp,
                    'datetime': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                    'prediction_file': None,
                    'is_active': False,
                    'minutes_remaining': 0,
                    'comparison_file': file.name,
                    'metrics': metrics
                }
                
                predictions.append(prediction_info)
            except (ValueError, IndexError):
                continue
        
        # Fallback: old static comparison images
        for file in sorted(self.data_dir.glob("prediction_comparison_*.png"), reverse=True):
            try:
                match = re.search(r'prediction_comparison_(\d+)\.png', file.name)
                if not match:
                    continue
                
                timestamp = int(match.group(1))
                
                # Skip if already added
                if any(p['timestamp'] == timestamp for p in predictions):
                    continue
                
                prediction_info = {
                    'timestamp': timestamp,
                    'datetime': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                    'prediction_file': None,
                    'is_active': False,
                    'minutes_remaining': 0,
                    'comparison_file': file.name,
                    'metrics': self.load_metrics(timestamp)
                }
                
                predictions.append(prediction_info)
            except (ValueError, IndexError):
                continue
        
        # Sort by timestamp descending
        predictions.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return predictions
    
    def get_latest_prediction_only(self):
        """
        Get the most recent prediction (prediction-only GIF, not comparison).
        This shows what the model just predicted, without waiting for actual data.
        """
        # Find the most recent prediction_only file
        prediction_files = sorted(self.data_dir.glob("prediction_only_*.gif"), reverse=True)
        if not prediction_files:
            return None
        
        latest = prediction_files[0]
        match = re.search(r'prediction_only_(\d+)\.gif', latest.name)
        if not match:
            return None
        
        timestamp = int(match.group(1))
        return {
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            'prediction_file': latest.name,
            'minutes_ago': int((datetime.now().timestamp() - timestamp) / 60)
        }
    
    def load_metrics(self, timestamp):
        """
        Load metrics for a specific prediction if available.
        """
        # Try to load from a metrics file if it exists
        metrics_file = self.data_dir / f"metrics_{timestamp}.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                return json.load(f)
        return None
    
    def get_statistics(self):
        """
        Get overall statistics about predictions.
        """
        predictions = self.get_all_predictions()
        
        if not predictions:
            return {
                'total_predictions': 0,
                'avg_mse': None,
                'avg_mae': None,
                'avg_psnr': None
            }
        
        # Calculate averages for metrics
        metrics_list = [p['metrics'] for p in predictions if p['metrics']]
        
        if not metrics_list:
            return {
                'total_predictions': len(predictions),
                'avg_mse': None,
                'avg_mae': None,
                'avg_psnr': None
            }
        
        # Handle both old (flat) and new (nested 'average') metric formats
        def _avg(m, key):
            if 'average' in m and key in m['average']:
                return m['average'][key]
            return m.get(key)

        mse_values = [v for m in metrics_list if (v := _avg(m, 'mse')) is not None]
        mae_values = [v for m in metrics_list if (v := _avg(m, 'mae')) is not None]
        psnr_values = [v for m in metrics_list if (v := _avg(m, 'psnr')) is not None]
        
        avg_mse = sum(mse_values) / len(mse_values) if mse_values else None
        avg_mae = sum(mae_values) / len(mae_values) if mae_values else None
        avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else None
        
        return {
            'total_predictions': len(predictions),
            'predictions_with_metrics': len(metrics_list),
            'avg_mse': avg_mse,
            'avg_mae': avg_mae,
            'avg_psnr': avg_psnr
        }

viewer = PredictionViewer()

@app.route('/')
def index():
    """Main page showing all predictions."""
    return render_template('index.html')

@app.route('/api/predictions')
def get_predictions():
    """API endpoint to get predictions with pagination."""
    limit = int(request.args.get('limit', 20))  # Default: last 20 predictions
    predictions = viewer.get_all_predictions()
    
    # Return only the most recent predictions
    return jsonify(predictions[:limit])

@app.route('/api/current_prediction')
def get_current_prediction():
    """API endpoint to get the latest prediction-only (most recent prediction without comparison)."""
    current = viewer.get_latest_prediction_only()
    return jsonify(current if current else {})

@app.route('/api/statistics')
def get_statistics():
    """API endpoint to get prediction statistics."""
    stats = viewer.get_statistics()
    return jsonify(stats)

@app.route('/images/<filename>')
def serve_image(filename):
    """Serve image files."""
    file_path = viewer.data_dir / filename
    if file_path.exists():
        return send_file(file_path, mimetype='image/png')
    return "Image not found", 404

@app.route('/api/prediction_control/start', methods=['POST'])
def start_predictions():
    """Start the prediction process."""
    global prediction_process, prediction_running, prediction_start_time

    # Already running via this web viewer
    if prediction_running and prediction_process and prediction_process.poll() is None:
        return jsonify({'status': 'already_running', 'message': 'Predictions are already running'})

    # Already running externally (e.g. launched by start.py)
    ext_pid, ext_started = _lock_info()
    if ext_pid:
        return jsonify({'status': 'already_running',
                        'message': f'Prediction engine is already running (PID {ext_pid})',
                        'pid': ext_pid})
    
    try:
        # Start persistent runner with auto-restart capability.
        # Use sys.executable so the venv Python is always used, regardless of PATH.
        # Do NOT pipe stdout/stderr: piping without a reader causes a pipe-buffer
        # deadlock (run_predictions_persistent streams child output via print(),
        # which would block once the ~64 KB Windows pipe buffer fills up).
        # Output flows to the web_viewer.py terminal instead, which is useful
        # for debugging and keeps the child alive.
        prediction_process = subprocess.Popen(
            [sys.executable, 'run_predictions_persistent.py'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        prediction_running = True
        prediction_start_time = int(time.time())
        return jsonify({'status': 'started', 'message': 'Prediction process started with auto-restart', 'pid': prediction_process.pid})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/prediction_control/stop', methods=['POST'])
def stop_predictions():
    """Stop the prediction process."""
    global prediction_process, prediction_running, prediction_start_time

    # If started by this web viewer, stop it directly
    if prediction_running and prediction_process:
        try:
            if os.name == 'nt':
                os.kill(prediction_process.pid, signal.CTRL_BREAK_EVENT)
            else:
                prediction_process.terminate()
            prediction_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            prediction_process.kill()
        except Exception:
            pass
        prediction_running = False
        prediction_process = None
        prediction_start_time = None
        return jsonify({'status': 'stopped', 'message': 'Prediction process stopped'})

    # If started externally (start.py), stop via the lock-file PID
    ext_pid, _ = _lock_info()
    if ext_pid:
        try:
            if os.name == 'nt':
                subprocess.run(["taskkill", "/PID", str(ext_pid), "/T", "/F"],
                               capture_output=True, timeout=10)
            else:
                os.kill(ext_pid, signal.SIGTERM)
            # Give it a moment to clean up its own lock file
            time.sleep(2)
            # Force-remove stale lock if the process left it behind
            try:
                PREDICTOR_LOCK.unlink(missing_ok=True)
            except Exception:
                pass
            return jsonify({'status': 'stopped',
                            'message': f'External prediction process (PID {ext_pid}) stopped'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    return jsonify({'status': 'not_running', 'message': 'Predictions are not running'})

@app.route('/api/prediction_control/status')
def prediction_status():
    """Get the status of the prediction process."""
    global prediction_process, prediction_running, prediction_start_time

    # Sync state for process started by this web viewer
    if prediction_process and prediction_process.poll() is not None:
        prediction_running = False
        prediction_process = None
        prediction_start_time = None

    if prediction_running and prediction_process:
        return jsonify({'running': True, 'pid': prediction_process.pid})

    # Check for predictor started externally (e.g. via start.py)
    ext_pid, _ = _lock_info()
    if ext_pid:
        return jsonify({'running': True, 'pid': ext_pid, 'external': True})

    return jsonify({'running': False, 'pid': None})

@app.route('/api/radar_health')
def radar_health():
    """
    Return predictor health from the status file written by predict_continuous.py,
    plus fetcher duplicate info from fetcher_status.json.
    Falls back to OK if the status file is absent or stale (>15 min old).
    """
    result = {'stuck': False, 'duplicate_count': 0, 'frames_available': None,
              'frames_needed': 0, 'eta_minutes': 0, 'message': 'OK',
              'last_duplicate_slot': None, 'missing_slots': []}

    # ── Fetcher duplicate info ──
    fetcher_file = Path("data/fetcher_status.json")
    if fetcher_file.exists():
        try:
            fdata = json.loads(fetcher_file.read_text())
            age_s = int(time.time()) - fdata.get("updated_at", 0)
            if age_s <= 600 and fdata.get("duplicate"):
                result['last_duplicate_slot'] = fdata.get("last_slot_str")
        except Exception:
            pass

    # ── Predictor stuck info ──
    status_file = Path("data/predictor_status.json")
    if status_file.exists():
        try:
            data = json.loads(status_file.read_text())
            age_s = int(time.time()) - data.get("updated_at", 0)
            # Treat status as expired after 15 min (3 prediction cycles missed)
            if age_s <= 900 and data.get("status") == "stuck":
                dup = data.get("duplicate_count", 0)
                available = data.get("frames_available")
                needed = data.get("frames_needed", max(0, dup - 1))
                eta = data.get("eta_minutes", needed * 5)
                if available is not None:
                    msg = (
                        f'Only {available} frame(s) in the last hour — '
                        f'{needed} more needed (~{eta} min)'
                    )
                else:
                    msg = (
                        f'Radar source appears stuck — '
                        f'{needed} more unique frame(s) needed (~{eta} min)'
                    )
                result.update({
                    'stuck': True,
                    'duplicate_count': dup,
                    'frames_available': available,
                    'frames_needed': needed,
                    'eta_minutes': eta,
                    'message': msg,
                    'missing_slots': data.get('missing_slots', []),
                })
        except Exception:
            pass

    return jsonify(result)

@app.route('/api/readiness')
def get_readiness():
    """Return how many fresh radar frames have arrived since the predictor was started."""
    global prediction_start_time, prediction_running, prediction_process

    # Sync process state first
    if prediction_process and prediction_process.poll() is not None:
        prediction_running = False
        prediction_process = None
        prediction_start_time = None

    # Determine effective start time: prefer the in-process value, fall back to
    # the lock file mtime for predictors launched externally by start.py.
    effective_start = prediction_start_time
    is_running = prediction_running

    if not is_running:
        ext_pid, ext_started = _lock_info()
        if ext_pid:
            is_running = True
            effective_start = ext_started

    if not is_running or effective_start is None:
        return jsonify({
            'running': False,
            'frames_since_start': 0,
            'frames_needed': 12,
            'ready': False,
            'eta_minutes': None,
            'started_at': None
        })

    radar_dir = Path('data/radar_images')
    frames_since_start = 0
    # Allow a 1-hour grace window before the predictor started so frames
    # captured by the fetcher just before clicking Start Predictions also count.
    # This still excludes frames that are hours/days/months old.
    fresh_cutoff = effective_start - 3600
    if radar_dir.exists():
        for f in radar_dir.glob('radar_*.png'):
            try:
                ts = int(f.stem.split('_')[1])
                if ts >= fresh_cutoff:
                    frames_since_start += 1
            except (ValueError, IndexError):
                continue

    frames_needed = 12
    ready = frames_since_start >= frames_needed
    eta_minutes = 0 if ready else (frames_needed - frames_since_start) * 5

    return jsonify({
        'running': True,
        'frames_since_start': frames_since_start,
        'frames_needed': frames_needed,
        'ready': ready,
        'eta_minutes': eta_minutes,
        'started_at': datetime.fromtimestamp(effective_start).strftime('%Y-%m-%d %H:%M:%S')
    })

if __name__ == '__main__':
    import logging
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--web-debug', action='store_true',
                        help='Show HTTP access log lines from the web server')
    args = parser.parse_args()

    if not args.web_debug:
        # Suppress werkzeug's per-request access log (the [WEB] GET /api/... lines)
        logging.getLogger('werkzeug').setLevel(logging.ERROR)

    port = int(os.environ.get('WEATHER_RADAR_VIEWER_PORT', DEFAULT_PORT))
    print("=" * 70)
    print("Weather Radar Prediction Viewer")
    print("=" * 70)
    print()
    print("Starting web interface...")
    print(f"Open your browser and go to: http://localhost:{port}")
    if not args.web_debug:
        print("(HTTP access logs hidden — run with --web-debug to show them)")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 70)

    # use_reloader=False prevents the werkzeug file-watcher from killing the
    # prediction subprocess reference stored in global variables on file changes.
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=port)
