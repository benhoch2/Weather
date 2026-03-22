import requests
from PIL import Image
from io import BytesIO
import hashlib
import time
from datetime import datetime
from pathlib import Path
import os
from process_utils import acquire_lock, release_lock

DEFAULT_MAX_STORED_IMAGES = 0

# Track the hash of the last successfully saved image to detect a stuck radar source
_last_content_hash: str | None = None

# ── Duplicate-process guard ──────────────────────────────────────────────────
FETCHER_LOCK = Path(__file__).parent / "fetcher.lock"


def fetch_radar_image():
    """
    Fetches the latest radar image from weather2day.co.il
    and saves it with an epoch timestamp filename.
    """
    global _last_content_hash
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

        # Detect stuck radar source: same bytes as last fetch → skip saving
        content_hash = hashlib.md5(response.content).hexdigest()
        if content_hash == _last_content_hash:
            print(f"  [WARNING] Radar source stuck — image unchanged from previous fetch, skipping save.")
            return None
        _last_content_hash = content_hash
        
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
        print(f"  [OK] Saved as: {filename}")
        
        # Clean up old images - keep only last 12
        cleanup_old_images()
        
        return filename
            
    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] Error fetching radar data: {e}")
        return None
    except Exception as e:
        print(f"  [ERROR] Error processing image: {e}")
        return None

def cleanup_old_images():
    """
    Optionally prune old radar images when a retention cap is configured.

    By default, no images are deleted so the fetcher can build a real
    training corpus over time. Set WEATHER_RADAR_MAX_IMAGES to a positive
    integer to enable automatic pruning.
    """
    try:
        images_dir = Path("data/radar_images")
        if not images_dir.exists():
            return

        max_images = int(os.environ.get("WEATHER_RADAR_MAX_IMAGES", DEFAULT_MAX_STORED_IMAGES))
        if max_images <= 0:
            return
        
        # Get all radar images sorted by timestamp (newest first)
        radar_images = sorted(
            images_dir.glob("radar_*.png"),
            key=lambda p: int(p.stem.split('_')[1]),
            reverse=True
        )
        
        # Keep only the configured number of recent images, delete the rest
        if len(radar_images) > max_images:
            images_to_delete = radar_images[max_images:]
            for img_path in images_to_delete:
                img_path.unlink()
            print(f"  [CLEANUP] Retained latest {max_images} images, removed {len(images_to_delete)} old image(s)")
    except Exception as e:
        print(f"  [WARNING] Error during cleanup: {e}")

def main():
    """
    Continuously fetch radar images every 5 minutes.
    """
    acquire_lock(FETCHER_LOCK, "FETCHER")

    print("=" * 60)
    print("Weather Radar Image Fetcher")
    print("Fetching radar images every 5 minutes")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()

    # Count how many images are already stored so the predictor can see the corpus size
    existing = sorted(Path("data/radar_images").glob("radar_*.png"))
    print(f"[FETCHER] Found {len(existing)} existing radar image(s) on disk")
    if existing:
        newest_ts = int(existing[-1].stem.split('_')[1])
        print(f"[FETCHER] Newest stored image: {datetime.fromtimestamp(newest_ts).strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    interval_seconds = 5 * 60  # 5 minutes

    try:
        while True:
            fetch_start = time.time()
            result = fetch_radar_image()
            fetch_elapsed = time.time() - fetch_start

            if result:
                total = len(list(Path("data/radar_images").glob("radar_*.png")))
                print(f"  [FETCHER] Stored {total} image(s) total  (fetch took {fetch_elapsed:.1f}s)")
            else:
                print(f"  [FETCHER] Fetch failed — will retry at next interval")

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
    finally:
        release_lock(FETCHER_LOCK)

if __name__ == "__main__":
    main()
