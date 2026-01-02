# Weather Radar Prediction System

A machine learning system that collects radar images and learns to predict future weather patterns.

## Overview

This system continuously collects weather radar images from Weather2day.co.il and uses a deep learning model (ConvLSTM) to predict the next radar image based on the previous hour of data (12 images at 5-minute intervals).

## Files

1. **fetch_radar.py** - Single image fetcher (for testing)
2. **fetch_radar_continuous.py** - Continuously fetches radar images every 5 minutes
3. **data_manager.py** - Manages radar image data and creates training sequences
4. **radar_model.py** - ConvLSTM neural network model for radar prediction
5. **train_model.py** - Initial training script
6. **predict_continuous.py** - Continuous prediction and online learning
7. **web_viewer.py** - Web interface to view predictions and metrics

## Setup

1. Install required packages:
```bash
python -m pip install tensorflow numpy pillow requests flask
```

2. Start collecting radar images:
```bash
python fetch_radar_continuous.py
```
Let it run for at least 65 minutes to collect enough images for training (13 images minimum).

## Usage

### Step 1: Collect Images
Run the continuous fetcher to build your dataset:
```bash
python fetch_radar_continuous.py
```

Keep this running to continuously collect radar images. The more images you collect, the better your model will be.

### Step 2: Train Initial Model
Once you have at least 13 images (recommended: 50+ images for better results):
```bash
python train_model.py
```

Optional parameters:
```bash
python train_model.py [epochs] [batch_size]
# Example: python train_model.py 100 4
```

This will:
- Load all collected radar images
- Create training sequences (12 images → 1 target image)
- Train a ConvLSTM model
- Save the trained model as `radar_model.keras`

### Step 3: Continuous Prediction and Learning
Once you have a trained model, start the continuous prediction system:
```bash
python predict_continuous.py
```

This script will:
1. Wait for enough images (12) to make predictions
2. **Predict the next 5 radar images** (5, 10, 15, 20, 25 minutes ahead)
3. Save predictions as individual frames and animated GIF
4. Wait 25 minutes for the actual images to arrive
5. Compare predictions with actual images (calculate MSE, MAE, PSNR for each frame)
6. Create animated GIF showing predicted vs actual side-by-side
7. Update the model weights based on the error (online learning)
8. Repeat indefinitely

The model continuously improves as it sees more real-world data!

### Step 4: View Predictions in Browser
Start the web interface to see all predictions and metrics in your browser:
```bash
python web_viewer.py
```

Then open your browser and go to: **http://localhost:5000**

The web interface shows:
- All prediction comparisons (predicted vs actual side-by-side)
- **Animated GIFs showing 5-frame predictions**
- Metrics for each prediction (MSE, MAE, PSNR)
- Average metrics across all 5 frames
- Overall statistics
- Auto-refreshes every 30 seconds
- Click any image to view full-size

## How It Works

### Architecture
The system uses a **ConvLSTM (Convolutional LSTM)** architecture:
- Input: 12 consecutive radar images (256x256 or 512x512 RGB)
- ConvLSTM layers capture both spatial (weather patterns) and temporal (movement) features
- Output: Single predicted radar image (5 minutes into the future)
- **Multi-frame prediction:** Recursively predicts 5 frames (25 minutes ahead) by feeding predictions back as input

### Online Learning
After each prediction:
1. The system waits for the actual radar image
2. Calculates prediction error
3. Updates model weights using backpropagation
4. The model gets better over time with each prediction!

### Key Features
- **No powerful GPU needed**: Trains/predicts every 5 minutes, plenty of time even on CPU
- **Continuous improvement**: Model updates itself with each new prediction
- **Metrics tracking**: MSE, MAE, and PSNR to measure prediction quality
- **Comparison images**: Saves side-by-side comparisons of predicted vs actual

## File Outputs

- `radar_TIMESTAMP.png` - Original radar images
- `predicted_TIMESTAMP.png` / `predicted_TIMESTAMP_frameN.png` - Predicted images
- `prediction_comparison_TIMESTAMP.png` - Side-by-side comparison (predicted | actual)
- `prediction_animation_TIMESTAMP.gif` - Animated GIF showing 5-frame prediction vs actual
- `metrics_TIMESTAMP.json` - Prediction metrics (MSE, MAE, PSNR) for each frame
- `radar_model.keras` - Trained model
- `training_metadata.json` - Training statistics and metadata

## Tips

1. **Collect more data**: The more radar images you collect, the better the model performs
2. **Be patient**: Initial predictions may not be great, but the model improves over time
3. **Monitor metrics**: Watch the MSE/MAE values decrease as the model learns
4. **Retrain periodically**: After collecting a large dataset, you can retrain from scratch for better results

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pillow (PIL)
- Requests
- Flask

## Future Enhancements

- Add weather condition classification (rain/no rain)
- Predict multiple time steps (10, 15, 30 minutes ahead)
- Add attention mechanisms for better long-term predictions
- Ensemble multiple models for improved accuracy
