"""
Temporary script to review all predictions with current model weights.
Shows predictions vs actual results with pagination (6 per page).
"""
from flask import Flask, render_template_string, request
from pathlib import Path
import numpy as np
from PIL import Image
import io
import base64
from data_manager import RadarDataManager
from radar_model import RadarPredictionModel

app = Flask(__name__)

# Global variables
data_manager = None
model = None
sequences = []
PREDICTIONS_PER_PAGE = 6

def initialize():
    """Initialize model and load sequences."""
    global data_manager, model, sequences
    
    print("Loading model and data...")
    data_manager = RadarDataManager(data_dir="data/radar_images")
    model = RadarPredictionModel(
        sequence_length=12,
        image_size=(512, 512),
        channels=3
    )
    
    if not model.load("radar_model.keras"):
        print("Error: No trained model found!")
        return False
    
    # Get all available images
    images = data_manager.get_all_radar_images()
    
    if len(images) < 17:  # Need 12 input + 5 actual = 17
        print(f"Error: Not enough images. Need at least 17, found {len(images)}")
        return False
    
    # Create all possible sequences
    print(f"Creating sequences from {len(images)} images...")
    for i in range(len(images) - 16):  # Need room for 12 input + 5 actual
        sequences.append({
            "index": i,
            "timestamp": images[i + 12][0],
            "sequence_images": images[i:i+12],
            "actual_images": images[i+12:i+17]  # 5 actual frames
        })
    
    print(f"Created {len(sequences)} sequences")
    return True

def calculate_metrics(predicted, actual):
    """Calculate comparison metrics."""
    mse = np.mean((predicted - actual) ** 2)
    mae = np.mean(np.abs(predicted - actual))
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float("inf")
    
    return {
        "mse": float(mse),
        "mae": float(mae),
        "psnr": float(psnr)
    }

def array_to_base64(img_array):
    """Convert numpy array to base64 string for HTML display."""
    img_uint8 = (img_array * 255).astype(np.uint8)
    img = Image.fromarray(img_uint8)
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    
    img_str = base64.b64encode(buffer.read()).decode()
    return f"data:image/png;base64,{img_str}"

def make_prediction(seq_data):
    """Make 5 predictions recursively (25 minutes)."""
    # Load sequence images
    sequence = []
    for img_info in seq_data["sequence_images"]:
        img = data_manager.load_image(img_info[1])
        sequence.append(img)
    
    # Make 5 predictions recursively
    predictions = []
    current_sequence = sequence.copy()
    
    for _ in range(5):
        # Create input batch
        input_seq = np.array([current_sequence[-12:]], dtype=np.float32)
        
        # Make prediction
        predicted = model.predict(input_seq)
        predictions.append(predicted)
        
        # Update sequence with prediction for next iteration
        current_sequence.append(predicted)
    
    # Load 5 actual images
    actuals = []
    for img_info in seq_data["actual_images"]:
        img = data_manager.load_image(img_info[1])
        actuals.append(img)
    
    # Calculate metrics for each frame
    frame_metrics = []
    for pred, actual in zip(predictions, actuals):
        metrics = calculate_metrics(pred, actual)
        frame_metrics.append(metrics)
    
    # Calculate average metrics
    avg_metrics = {
        "mse": np.mean([m["mse"] for m in frame_metrics]),
        "mae": np.mean([m["mae"] for m in frame_metrics]),
        "psnr": np.mean([m["psnr"] for m in frame_metrics])
    }
    
    return {
        "predictions": predictions,
        "actuals": actuals,
        "frame_metrics": frame_metrics,
        "avg_metrics": avg_metrics
    }

@app.route("/")
def index():
    """Main page with loading screen."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prediction Review - Loading...</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 20px;
                min-height: 100vh;
            }
            .loading {
                background: white;
                padding: 40px;
                border-radius: 10px;
                text-align: center;
                max-width: 600px;
                margin: 100px auto;
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 20px auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            #status {
                margin-top: 20px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="loading">
            <h1>Loading Predictions</h1>
            <div class="spinner"></div>
            <div id="status">Initializing...</div>
        </div>
        <script>
            const urlParams = new URLSearchParams(window.location.search);
            const page = urlParams.get("page") || 1;
            
            document.getElementById("status").textContent = `Loading page ${page}... This may take 1-2 minutes.`;
            
            // Load the actual predictions page
            fetch("/predictions?page=" + page)
                .then(response => response.text())
                .then(html => {
                    document.body.innerHTML = html;
                })
                .catch(error => {
                    document.getElementById("status").textContent = "Error loading predictions: " + error;
                });
        </script>
    </body>
    </html>
    """

@app.route("/predictions")
def predictions():
    """Generate predictions page."""
    print("Predictions endpoint called!")
    page = int(request.args.get("page", 1))
    print(f"Requested page: {page}")
    
    if not sequences:
        return "<h1>Error: No sequences loaded</h1>"
    
    # Calculate pagination
    total_pages = (len(sequences) + PREDICTIONS_PER_PAGE - 1) // PREDICTIONS_PER_PAGE
    page = max(1, min(page, total_pages))
    
    start_idx = (page - 1) * PREDICTIONS_PER_PAGE
    end_idx = min(start_idx + PREDICTIONS_PER_PAGE, len(sequences))
    
    page_sequences = sequences[start_idx:end_idx]
    
    # Make predictions for this page
    results = []
    for idx, seq_data in enumerate(page_sequences):
        print(f"Processing sequence {idx + 1}/{len(page_sequences)} (#{seq_data['index']})...")
        result = make_prediction(seq_data)
        
        # Convert all frames to base64
        predicted_imgs = [array_to_base64(pred) for pred in result["predictions"]]
        actual_imgs = [array_to_base64(act) for act in result["actuals"]]
        
        results.append({
            "index": seq_data["index"],
            "timestamp": seq_data["timestamp"],
            "predicted_imgs": predicted_imgs,
            "actual_imgs": actual_imgs,
            "frame_metrics": result["frame_metrics"],
            "avg_metrics": result["avg_metrics"]
        })
    
    print(f"Page {page} ready!")
    
    # Render HTML - using double braces for literal braces in template
    html = """<!DOCTYPE html><html><head><title>Prediction Review - Page """ + str(page) + """ / """ + str((len(sequences) + PREDICTIONS_PER_PAGE - 1) // PREDICTIONS_PER_PAGE) + """</title><style>* { margin: 0; padding: 0; box-sizing: border-box; } body { font-family: Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; min-height: 100vh; } .container { max-width: 1600px; margin: 0 auto; } header { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 5px 15px rgba(0,0,0,0.3); } h1 { color: #667eea; margin-bottom: 10px; } .info { color: #666; font-size: 0.9em; } .pagination { background: white; padding: 15px; border-radius: 10px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 5px 15px rgba(0,0,0,0.3); } .pagination a { background: #667eea; color: white; padding: 10px 20px; border-radius: 5px; text-decoration: none; font-weight: bold; } .pagination a:hover { background: #5568d3; } .pagination a.disabled { background: #ccc; pointer-events: none; } .predictions-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; margin-bottom: 20px; } .prediction-card { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 5px 15px rgba(0,0,0,0.3); } .prediction-header { font-weight: bold; color: #667eea; margin-bottom: 15px; font-size: 1.1em; } .frames-container { display: flex; gap: 10px; overflow-x: auto; margin-bottom: 15px; padding: 10px 0; } .frame-pair { flex: 0 0 auto; text-align: center; min-width: 200px; } .frame-pair img { width: 200px; height: 200px; object-fit: cover; border-radius: 5px; border: 2px solid #ddd; display: block; margin-bottom: 5px; } .frame-pair label { display: block; font-size: 0.8em; font-weight: bold; color: #333; margin-bottom: 3px; } .frame-time { font-size: 0.7em; color: #999; } .frame-metrics-small { font-size: 0.7em; color: #666; margin-top: 3px; } .avg-metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; padding: 10px; background: #f5f5f5; border-radius: 5px; margin-bottom: 10px; } .metric { text-align: center; } .metric-value { font-size: 1.2em; font-weight: bold; color: #667eea; } .metric-label { font-size: 0.8em; color: #666; text-transform: uppercase; }</style></head><body><div class="container"><header><h1>Prediction Review</h1><div class="info">Using current model weights - Showing """ + str(start_idx + 1) + """-""" + str(end_idx) + """ of """ + str(len(sequences)) + """ sequences</div></header><div class="pagination">"""
    
    total_pages = (len(sequences) + PREDICTIONS_PER_PAGE - 1) // PREDICTIONS_PER_PAGE
    if page <= 1:
        html += """<a href="/?page=1" class="disabled">Previous</a>"""
    else:
        html += f"""<a href="/?page={page - 1}">Previous</a>"""
    
    html += f"""<span style="font-weight: bold; color: #667eea;">Page {page} / {total_pages}</span>"""
    
    if page >= total_pages:
        html += """<a href="/?page=""" + str(total_pages) + """" class="disabled">Next</a>"""
    else:
        html += f"""<a href="/?page={page + 1}">Next</a>"""
    
    html += """</div><div class="predictions-grid">"""
    
    for result in results:
        html += f"""<div class="prediction-card"><div class="prediction-header">Sequence #{result["index"]} - Starting: {result["timestamp"]} - 25 minutes ahead (5 frames)</div><div class="avg-metrics"><div class="metric"><div class="metric-value">{result["avg_metrics"]["mse"]:.6f}</div><div class="metric-label">Avg MSE</div></div><div class="metric"><div class="metric-value">{result["avg_metrics"]["mae"]:.6f}</div><div class="metric-label">Avg MAE</div></div><div class="metric"><div class="metric-value">{result["avg_metrics"]["psnr"]:.2f}</div><div class="metric-label">Avg PSNR (dB)</div></div></div><div class="frames-container">"""
        
        for i in range(5):
            html += f"""<div class="frame-pair"><div class="frame-time">+{(i+1) * 5} min</div><img src="{result["predicted_imgs"][i]}" alt="Predicted Frame {i+1}"><label>Predicted</label><img src="{result["actual_imgs"][i]}" alt="Actual Frame {i+1}"><label>Actual</label><div class="frame-metrics-small">MSE: {result["frame_metrics"][i]["mse"]:.4f}<br>PSNR: {result["frame_metrics"][i]["psnr"]:.1f} dB</div></div>"""
        
        html += """</div></div>"""
    
    html += """</div><div class="pagination">"""
    
    if page <= 1:
        html += """<a href="/?page=1" class="disabled">Previous</a>"""
    else:
        html += f"""<a href="/?page={page - 1}">Previous</a>"""
    
    html += f"""<span style="font-weight: bold; color: white;">Page {page} / {total_pages}</span>"""
    
    if page >= total_pages:
        html += """<a href="/?page=""" + str(total_pages) + """" class="disabled">Next</a>"""
    else:
        html += f"""<a href="/?page={page + 1}">Next</a>"""
    
    html += """</div></div></body></html>"""
    
    return html

if __name__ == "__main__":
    print("=" * 70)
    print("Prediction Review Tool")
    print("=" * 70)
    print()
    
    if not initialize():
        print("Failed to initialize. Exiting.")
        exit(1)
    
    print()
    print("Starting web server...")
    print("Open your browser and go to: http://localhost:5001")
    print()
    print("This will recalculate predictions with current model weights")
    print("and show them 6 at a time with navigation.")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 70)
    
    app.run(debug=False, host="0.0.0.0", port=5001)
