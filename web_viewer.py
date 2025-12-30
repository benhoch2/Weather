from flask import Flask, render_template, jsonify, send_file
from pathlib import Path
import json
from datetime import datetime
import re

app = Flask(__name__)

class PredictionViewer:
    """
    Manages viewing of radar predictions and comparisons.
    """
    
    def __init__(self, data_dir="."):
        self.data_dir = Path(data_dir)
    
    def get_all_predictions(self):
        """
        Get all predictions with their metadata.
        
        Returns:
            List of dictionaries containing prediction info
        """
        predictions = []
        
        # Find all comparison images
        for file in sorted(self.data_dir.glob("prediction_comparison_*.png"), reverse=True):
            try:
                # Extract timestamp from filename
                match = re.search(r'prediction_comparison_(\d+)\.png', file.name)
                if not match:
                    continue
                
                timestamp = int(match.group(1))
                
                # Look for corresponding predicted and actual images
                predicted_file = self.data_dir / f"predicted_{timestamp}.png"
                
                # Find the actual radar image (closest in time)
                target_time = timestamp + 5 * 60
                actual_files = list(self.data_dir.glob(f"radar_{target_time}*.png"))
                if not actual_files:
                    # Search for radar image within +/- 2 minutes
                    actual_files = []
                    for t in range(target_time - 120, target_time + 120):
                        files = list(self.data_dir.glob(f"radar_{t}.png"))
                        if files:
                            actual_files = files
                            break
                
                actual_file = actual_files[0] if actual_files else None
                
                prediction_info = {
                    'timestamp': timestamp,
                    'datetime': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                    'comparison_image': file.name,
                    'predicted_image': predicted_file.name if predicted_file.exists() else None,
                    'actual_image': actual_file.name if actual_file else None,
                    'metrics': self.load_metrics(timestamp)
                }
                
                predictions.append(prediction_info)
            except (ValueError, IndexError):
                continue
        
        return predictions
    
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
        
        avg_mse = sum(m['mse'] for m in metrics_list) / len(metrics_list)
        avg_mae = sum(m['mae'] for m in metrics_list) / len(metrics_list)
        avg_psnr = sum(m['psnr'] for m in metrics_list) / len(metrics_list)
        
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
    """API endpoint to get all predictions."""
    predictions = viewer.get_all_predictions()
    return jsonify(predictions)

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
