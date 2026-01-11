from flask import Flask, render_template, jsonify, send_file, request
from pathlib import Path
import json
from datetime import datetime
import re
import subprocess
import os
import signal

app = Flask(__name__)

# Global process tracking
prediction_process = None
prediction_running = False

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
        
        # Handle missing keys gracefully
        mse_values = [m['mse'] for m in metrics_list if 'mse' in m]
        mae_values = [m['mae'] for m in metrics_list if 'mae' in m]
        psnr_values = [m['psnr'] for m in metrics_list if 'psnr' in m]
        
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
    global prediction_process, prediction_running
    
    if prediction_running and prediction_process and prediction_process.poll() is None:
        return jsonify({'status': 'already_running', 'message': 'Predictions are already running'})
    
    try:
        # Start persistent runner with auto-restart capability
        prediction_process = subprocess.Popen(
            ['python', 'run_predictions_persistent.py'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        prediction_running = True
        return jsonify({'status': 'started', 'message': 'Prediction process started with auto-restart', 'pid': prediction_process.pid})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/prediction_control/stop', methods=['POST'])
def stop_predictions():
    """Stop the prediction process."""
    global prediction_process, prediction_running
    
    if not prediction_running or not prediction_process:
        return jsonify({'status': 'not_running', 'message': 'Predictions are not running'})
    
    try:
        if os.name == 'nt':
            # Windows: Send CTRL_BREAK_EVENT
            os.kill(prediction_process.pid, signal.CTRL_BREAK_EVENT)
        else:
            # Unix: Send SIGTERM
            prediction_process.terminate()
        
        prediction_process.wait(timeout=5)
        prediction_running = False
        prediction_process = None
        return jsonify({'status': 'stopped', 'message': 'Prediction process stopped'})
    except subprocess.TimeoutExpired:
        # Force kill if it doesn't stop gracefully
        prediction_process.kill()
        prediction_running = False
        prediction_process = None
        return jsonify({'status': 'force_stopped', 'message': 'Prediction process force stopped'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/prediction_control/status')
def prediction_status():
    """Get the status of the prediction process."""
    global prediction_process, prediction_running
    
    # Check if process is actually running
    if prediction_process and prediction_process.poll() is not None:
        prediction_running = False
        prediction_process = None
    
    return jsonify({
        'running': prediction_running,
        'pid': prediction_process.pid if prediction_process else None
    })

if __name__ == '__main__':
    print("=" * 70)
    print("Weather Radar Prediction Viewer")
    print("=" * 70)
    print()
    print("Starting web interface...")
    print("Open your browser and go to: http://localhost:5000")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
