import requests
from PIL import Image
from io import BytesIO
import time
from datetime import datetime
from pathlib import Path

def fetch_radar_image():
    """
    Fetches the latest radar image from weather2day.co.il
    and saves it with an epoch timestamp filename.
    """
    # The radar.php endpoint returns the PNG image directly
    radar_image_url = "https://www.weather2day.co.il/radar.php"
    
    try:
        # Fetch the radar image
        response = requests.get(radar_image_url, timeout=10)
        response.raise_for_status()
        
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp_str}] Fetching radar image...")
        print(f"  Content-Type: {response.headers.get('Content-Type')}")
        print(f"  Content size: {len(response.content)} bytes")
        
        # Get current epoch timestamp for filename
        current_timestamp = int(time.time())
        filename = f"data/radar_images/radar_{current_timestamp}.png"
        
        # Open and process the image
        img = Image.open(BytesIO(response.content))
        
        # If it's an animated PNG (APNG), get the last frame
        if hasattr(img, 'n_frames') and img.n_frames > 1:
            # Seek to the last frame
            img.seek(img.n_frames - 1)
            print(f"  Animated PNG with {img.n_frames} frames. Extracting last frame.")
        
        # Save the image
        img.save(filename)
        print(f"  ✓ Saved as: {filename}")
        
        # Clean up old images - keep only last 12
        cleanup_old_images()
        
        return filename
            
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error fetching radar data: {e}")
        return None
    except Exception as e:
        print(f"  ✗ Error processing image: {e}")
        return None

def cleanup_old_images():
    """
    Keep only the last 12 radar images, delete older ones.
    This ensures we always have a complete sequence ready for predictions.
    """
    try:
        images_dir = Path("data/radar_images")
        if not images_dir.exists():
            return
        
        # Get all radar images sorted by timestamp (newest first)
        radar_images = sorted(
            images_dir.glob("radar_*.png"),
            key=lambda p: int(p.stem.split('_')[1]),
            reverse=True
        )
        
        # Keep only the 12 most recent, delete the rest
        if len(radar_images) > 12:
            images_to_delete = radar_images[12:]
            for img_path in images_to_delete:
                img_path.unlink()
            print(f"  [CLEANUP] Cleaned up {len(images_to_delete)} old image(s)")
    except Exception as e:
        print(f"  [WARNING] Error during cleanup: {e}")

def main():
    """
    Continuously fetch radar images every 5 minutes.
    """
    print("=" * 60)
    print("Weather Radar Image Fetcher")
    print("Fetching radar images every 5 minutes")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()
    
    interval_seconds = 5 * 60  # 5 minutes
    
    try:
        while True:
            # Fetch the image
            fetch_radar_image()
            
            # Wait for 5 minutes
            next_fetch = datetime.now().timestamp() + interval_seconds
            next_fetch_time = datetime.fromtimestamp(next_fetch).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  Next fetch at: {next_fetch_time}")
            print()
            
            time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        print("\n")
        print("=" * 60)
        print("Stopped by user. Goodbye!")
        print("=" * 60)

if __name__ == "__main__":
    main()
