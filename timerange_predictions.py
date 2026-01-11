"""
Script to run predictions on specific time range and display them on web server.
Web server starts immediately and shows predictions as they're processed.
"""
from flask import Flask, render_template_string
import numpy as np
from PIL import Image
import io
import base64
from data_manager import RadarDataManager
from radar_model import RadarPredictionModel
from pathlib import Path
import threading
import time

app = Flask(__name__)

# Global variables
results = []
processing_complete = False
total_sequences = 0
current_sequence = 0

def calculate_metrics(predicted, actual):
    """Calculate comparison metrics."""
    mse = np.mean((predicted - actual) ** 2)
    mae = np.mean(np.abs(predicted - actual))
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'psnr': float(psnr)
    }

def create_animated_gif(frames, duration=500):
    """Create animated GIF from frames and return as base64."""
    pil_frames = []
    for frame in frames:
        img_uint8 = (frame * 255).astype(np.uint8)
        # Convert to RGB mode explicitly
        pil_img = Image.fromarray(img_uint8, mode='RGB')
        pil_frames.append(pil_img)
    
    # Save to bytes - use smaller optimization
    buffer = io.BytesIO()
    pil_frames[0].save(
        buffer,
        format='GIF',
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
        optimize=False  # Faster, slightly larger files
    )
    buffer.seek(0)
    
    img_str = base64.b64encode(buffer.read()).decode()
    return f"data:image/gif;base64,{img_str}"

def process_sequences():
    """Process all sequences in the time range."""
    global results, processing_complete, total_sequences, current_sequence
    
    print("=" * 70)
    print("Processing Predictions for Time Range")
    print("=" * 70)
    print()
    
    # Load model and data
    print("Loading model and data...")
    data_manager = RadarDataManager(data_dir="data/radar_images")
    model = RadarPredictionModel(
        sequence_length=12,
        image_size=(512, 512),
        channels=3
    )
    
    if not model.load('radar_model.keras'):
        print("Error: No trained model found!")
        return False
    
    # Get all images
    images = data_manager.get_all_radar_images()
    print(f"Found {len(images)} total images")
    
    # Filter images in time range
    start_time = 1767272839
    end_time = 1767295457
    
    # Find start and end indices
    start_idx = None
    end_idx = None
    
    for idx, (timestamp, path) in enumerate(images):
        if start_idx is None and timestamp >= start_time:
            start_idx = idx
        if timestamp <= end_time:
            end_idx = idx
    
    if start_idx is None or end_idx is None:
        print(f"Error: Could not find images in time range {start_time} - {end_time}")
        return False
    
    print(f"Time range covers images {start_idx} to {end_idx}")
    
    # Create sequences (need 12 input + 5 actual = 17 frames per sequence)
    sequences = []
    for i in range(start_idx, end_idx - 16):
        # Check if this sequence is fully within range
        seq_start_time = images[i][0]
        seq_end_time = images[i + 16][0]
        
        if seq_start_time >= start_time and seq_end_time <= end_time:
            sequences.append({
                'index': i,
                'start_time': seq_start_time,
                'end_time': seq_end_time,
                'input_images': images[i:i+12],
                'actual_images': images[i+12:i+17]
            })
    
    print(f"Found {len(sequences)} complete sequences in time range")
    print()
    
    if len(sequences) == 0:
        print("No sequences found in range!")
        processing_complete = True
        return False
    
    total_sequences = len(sequences)
    
    # Process each sequence
    for seq_idx, seq in enumerate(sequences):
        current_sequence = seq_idx + 1
        print(f"Processing sequence {current_sequence}/{total_sequences} (starting at {seq['start_time']})...")
        
        # Load input sequence
        input_frames = []
        for img_info in seq['input_images']:
            img = data_manager.load_image(img_info[1])
            input_frames.append(img)
        
        # Make 5 predictions recursively
        predictions = []
        current_sequence = input_frames.copy()
        
        for pred_num in range(5):
            input_seq = np.array([current_sequence[-12:]], dtype=np.float32)
            predicted = model.predict(input_seq)
            predictions.append(predicted)
            current_sequence.append(predicted)
        
        # Load actual frames
        actuals = []
        for img_info in seq['actual_images']:
            img = data_manager.load_image(img_info[1])
            actuals.append(img)
        
        # Calculate metrics
        frame_metrics = []
        for pred, actual in zip(predictions, actuals):
            metrics = calculate_metrics(pred, actual)
            frame_metrics.append(metrics)
        
        avg_metrics = {
            'mse': np.mean([m['mse'] for m in frame_metrics]),
            'mae': np.mean([m['mae'] for m in frame_metrics]),
            'psnr': np.mean([m['psnr'] for m in frame_metrics])
        }
        
        # Create animated GIFs
        pred_gif = create_animated_gif(predictions)
        actual_gif = create_animated_gif(actuals)
        
        results.append({
            'index': seq_idx,
            'start_time': seq['start_time'],
            'end_time': seq['end_time'],
            'pred_gif': pred_gif,
            'actual_gif': actual_gif,
            'avg_metrics': avg_metrics,
            'frame_metrics': frame_metrics
        })
    
    print()
    print(f"✓ Completed processing {len(results)} sequences")
    print()
    processing_complete = True
    return True

@app.route('/')
def index():
    """Display all predictions with auto-refresh."""
    status_msg = ""
    if processing_complete:
        status_msg = f"✓ Processing complete - Showing all {len(results)} predictions"
    else:
        status_msg = f"⏳ Processing... {current_sequence}/{total_sequences} sequences completed"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Predictions for Time Range</title>
        <meta http-equiv="refresh" content="5">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                min-height: 100vh;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
            }}
            header {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            }}
            h1 {{
                color: #667eea;
                margin-bottom: 10px;
            }}
            .info {{
                color: #666;
                font-size: 0.9em;
                margin-bottom: 5px;
            }}
            .status {{
                color: #667eea;
                font-weight: bold;
                font-size: 1.1em;
                margin-top: 10px;
            }}
            .predictions-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
                gap: 20px;
            }}
            .prediction-card {{
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            }}
            .prediction-header {{
                font-weight: bold;
                color: #667eea;
                margin-bottom: 15px;
                font-size: 1.1em;
            }}
            .comparison {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin-bottom: 15px;
            }}
            .comparison-item {{
                text-align: center;
            }}
            .comparison-item img {{
                width: 100%;
                border-radius: 5px;
                border: 2px solid #ddd;
            }}
            .comparison-item label {{
                display: block;
                margin-top: 8px;
                font-weight: bold;
                color: #333;
            }}
            .metrics {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 10px;
                padding: 10px;
                background: #f5f5f5;
                border-radius: 5px;
            }}
            .metric {{
                text-align: center;
            }}
            .metric-value {{
                font-size: 1.1em;
                font-weight: bold;
                color: #667eea;
            }}
            .metric-label {{
                font-size: 0.75em;
                color: #666;
                text-transform: uppercase;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🌦️ Weather Predictions - Time Range Analysis</h1>
                <div class="info">
                    Time range: 1767272839 to 1767295457 | Total sequences: {total_sequences}
                </div>
                <div class="status">{status_msg}</div>
                <div class="info" style="margin-top: 10px; font-size: 0.8em;">
                    Page auto-refreshes every 5 seconds to show new predictions
                </div>
            </header>
            
            <div class="predictions-grid">
    """
    
    for result in results:
        html += f"""
                <div class="prediction-card">
                    <div class="prediction-header">
                        Sequence #{result['index']} | Start: {result['start_time']} | 25 min ahead (5 frames)
                    </div>
                    
                    <div class="comparison">
                        <div class="comparison-item">
                            <img src="{result['pred_gif']}" alt="Predicted">
                            <label>🔮 Predicted (5 frames)</label>
                        </div>
                        <div class="comparison-item">
                            <img src="{result['actual_gif']}" alt="Actual">
                            <label>✅ Actual (5 frames)</label>
                        </div>
                    </div>
                    
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-value">{result['avg_metrics']['mse']:.6f}</div>
                            <div class="metric-label">Avg MSE</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{result['avg_metrics']['mae']:.6f}</div>
                            <div class="metric-label">Avg MAE</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{result['avg_metrics']['psnr']:.2f}</div>
                            <div class="metric-label">Avg PSNR (dB)</div>
                        </div>
                    </div>
                </div>
        """
    
    html += """
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

if __name__ == '__main__':
    # Start web server in background thread
    server_thread = threading.Thread(target=lambda: app.run(debug=False, host='0.0.0.0', port=5002, use_reloader=False))
    server_thread.daemon = True
    server_thread.start()
    
    print("=" * 70)
    print("Web server started at: http://localhost:5002")
    print("Open your browser to see predictions as they're processed")
    print("=" * 70)
    print()
    
    # Give server time to start
    time.sleep(2)
    
    # Process sequences in main thread
    try:
        process_sequences()
        print()
        print("=" * 70)
        print("Processing complete! Keep the browser open to view results.")
        print("Press Ctrl+C to stop the server")
        print("=" * 70)
        
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")

