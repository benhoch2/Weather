"""
Test script to detect clouds in the first sequence.
Compares frames to a reference clear sky image.
"""
import numpy as np
from PIL import Image
from data_manager import RadarDataManager
from radar_model import RadarPredictionModel

def calculate_cloud_score(frame, reference):
    """
    Calculate how different a frame is from the reference (clear sky).
    Higher score = more clouds.
    """
    # Calculate absolute difference
    diff = np.abs(frame - reference)
    
    # Average difference across all pixels and channels
    score = np.mean(diff)
    
    # Also check standard deviation (clouds create more variation)
    std_score = np.std(frame)
    
    return score, std_score

def main():
    print("=" * 70)
    print("Cloud Detection Test - First Sequence")
    print("=" * 70)
    print()
    
    # Load data
    print("Loading radar images...")
    data_manager = RadarDataManager(data_dir="data/radar_images")
    images = data_manager.get_all_radar_images()
    
    if len(images) < 17:
        print(f"Error: Need at least 17 images, found {len(images)}")
        return
    
    print(f"Found {len(images)} images")
    print()
    
    # Use the first image as reference (or find the clearest one)
    # For now, let's find the image with lowest variance (likely clearest)
    print("Finding reference clear sky image...")
    min_std = float('inf')
    reference_img = None
    reference_idx = 0
    
    for idx in range(min(100, len(images))):  # Check first 100 images
        img = data_manager.load_image(images[idx][1])
        img_std = np.std(img)
        if img_std < min_std:
            min_std = img_std
            reference_img = img
            reference_idx = idx
    
    print(f"Using image #{reference_idx} as reference (std: {min_std:.6f})")
    print()
    
    # Skip full analysis for now - just analyze first sequence
    print("Analyzing first sequence only (frames 0-11)...")
    print("-" * 70)
    sequence_frames = []
    cloud_scores = []
    std_scores = []
    
    for i in range(12):
        img = data_manager.load_image(images[i][1])
        sequence_frames.append(img)
        diff_score, std_score = calculate_cloud_score(img, reference_img)
        cloud_scores.append(diff_score)
        std_scores.append(std_score)
        
        print(f"Frame {i:2d} | Difference: {diff_score:.6f} | Std Dev: {std_score:.6f} | Timestamp: {images[i][0]}")
    
    print("-" * 70)
    
    # Determine if sequence has clouds
    avg_cloud_score = np.mean(cloud_scores)
    avg_std_score = np.mean(std_scores)
    
    print()
    print(f"Average difference score: {avg_cloud_score:.6f}")
    print(f"Average std dev score: {avg_std_score:.6f}")
    print()
    
    # Simple threshold: if average difference > 0.2 or std > 0.25, consider it has clouds
    # Higher values = less sensitive, only detect significant clouds
    # Recommended: Use 75th-90th percentile from statistics above
    threshold_diff = 0.011620  # Very selective: top 5% cloudiest (95th percentile)
    threshold_std = 0.263
    
    has_clouds = avg_cloud_score > threshold_diff or avg_std_score > threshold_std
    
    print(f"Threshold for clouds: difference > {threshold_diff} OR std > {threshold_std}")
    print()
    
    if has_clouds:
        print("✅ VERDICT: This sequence HAS CLOUDS")
        print()
        print("All 12 frames appear to contain cloud activity.")
    else:
        print("❌ VERDICT: This sequence appears to be CLEAR/EMPTY")
        print()
        print("The frames look similar to the reference clear sky.")
    
    print()
    print("-" * 70)
    print("Visual Inspection:")
    print("Creating comparison image...")
    
    # Create a visual comparison using PIL
    # Layout: 3 rows x 4 columns = 12 frames only
    cell_width = 512
    cell_height = 512
    img_width = cell_width * 4
    img_height = cell_height * 3
    
    comparison = Image.new('RGB', (img_width, img_height), color='white')
    
    # Show all 12 frames in 3x4 grid
    for i in range(12):
        row = i // 4
        col = i % 4
        frame_pil = Image.fromarray((sequence_frames[i] * 255).astype(np.uint8))
        comparison.paste(frame_pil, (col * cell_width, row * cell_height))
    
    # Save
    comparison.save('cloud_detection_test.png')
    print("Saved visualization to: cloud_detection_test.png")
    print(f"Shows all 12 frames in a 3x4 grid")
    print()
    
    print("=" * 70)
    print("Review the image to verify if the cloud detection is correct.")
    print("If it looks good, we can process all sequences.")
    print("=" * 70)

if __name__ == "__main__":
    main()
