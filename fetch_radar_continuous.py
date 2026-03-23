import requests
from PIL import Image
from io import BytesIO
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
import os
from process_utils import acquire_lock, release_lock

DEFAULT_MAX_STORED_IMAGES = 500

# Track the hash of the last successfully saved image to detect a stuck radar source
_last_content_hash: str | None = None

_FETCHER_STATUS_FILE = Path("data/fetcher_status.json")

# ── Duplicate-process guard ──────────────────────────────────────────────────
FETCHER_LOCK = Path(__file__).parent / "fetcher.lock"


def _next_slot_time():
    """Return the epoch time of the next 5-minute wall-clock boundary."""
    now = time.time()
    slot = 5 * 60
    return int((now // slot) + 1) * slot


def _write_fetcher_status(slot_ts, *, duplicate=False):
    """Write a small JSON so the web UI can report duplicate/lost frames."""
    slot_str = datetime.fromtimestamp(slot_ts).strftime("%Y-%m-%d %H:%M")
    payload = {
        "updated_at": int(time.time()),
        "last_slot": slot_ts,
        "last_slot_str": slot_str,
        "duplicate": duplicate,
    }
    try:
        _FETCHER_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _FETCHER_STATUS_FILE.write_text(json.dumps(payload))
    except Exception:
        pass


def fetch_radar_image(slot_ts):
    """
    Fetches the latest radar image from weather2day.co.il
    and saves it with the scheduled slot epoch timestamp as filename.

    Returns the saved filename, or None on failure/duplicate.
    """
    global _last_content_hash
    radar_image_url = "https://www.weather2day.co.il/radar.php"

    slot_str = datetime.fromtimestamp(slot_ts).strftime("%Y-%m-%d %H:%M:%S")
    filename = f"data/radar_images/radar_{slot_ts}.png"

    # Skip if this slot was already fetched (e.g. server restarted)
    if Path(filename).exists():
        print(f"  [SKIP] Image for slot {slot_str} already exists on disk — skipping fetch.")
        return filename

    try:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp_str}] Fetching radar image for slot {slot_str}...")

        response = requests.get(radar_image_url, timeout=10)
        response.raise_for_status()

        print(f"  Content-Type: {response.headers.get('Content-Type')}")
        print(f"  Content size: {len(response.content)} bytes")

        # Duplicate detection disabled — always save so every 5-min slot
        # has a frame, letting the model learn from static radar too.
        content_hash = hashlib.md5(response.content).hexdigest()
        is_duplicate = (content_hash == _last_content_hash)
        _last_content_hash = content_hash
        if is_duplicate:
            print(f"  [INFO] Slot {slot_str} — image identical to previous fetch (saving anyway).")
            _write_fetcher_status(slot_ts, duplicate=True)
        else:
            _write_fetcher_status(slot_ts, duplicate=False)

        # Open and process the image
        img = Image.open(BytesIO(response.content))

        # If it's an animated PNG (APNG), get the last frame
        if hasattr(img, 'n_frames') and img.n_frames > 1:
            img.seek(img.n_frames - 1)
            print(f"  Animated PNG with {img.n_frames} frames. Extracting last frame.")

        img.save(filename)
        print(f"  [OK] Saved as: {filename}")
        _write_fetcher_status(slot_ts, duplicate=False)

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
    Continuously fetch radar images aligned to 5-minute wall-clock slots
    (XX:00, XX:05, XX:10, … XX:55).
    """
    acquire_lock(FETCHER_LOCK, "FETCHER")

    print("=" * 60)
    print("Weather Radar Image Fetcher")
    print("Fetching at 5-minute wall-clock intervals")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()

    # Count how many images are already stored
    existing = sorted(Path("data/radar_images").glob("radar_*.png"))
    print(f"[FETCHER] Found {len(existing)} existing radar image(s) on disk")
    if existing:
        newest_ts = int(existing[-1].stem.split('_')[1])
        print(f"[FETCHER] Newest stored image: {datetime.fromtimestamp(newest_ts).strftime('%Y-%m-%d %H:%M:%S')}")

    # Wait until the next 5-minute boundary before the first fetch
    next_slot = _next_slot_time()
    wait_sec = max(0, next_slot - time.time())
    next_slot_str = datetime.fromtimestamp(next_slot).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[FETCHER] Waiting for next 5-min slot: {next_slot_str} (in {int(wait_sec)}s)")
    print()
    time.sleep(wait_sec)

    try:
        while True:
            slot_ts = int((time.time() // (5 * 60)) * (5 * 60))
            slot_str = datetime.fromtimestamp(slot_ts).strftime("%Y-%m-%d %H:%M:%S")

            result = fetch_radar_image(slot_ts)

            if result:
                total = len(list(Path("data/radar_images").glob("radar_*.png")))
                print(f"  [FETCHER] Stored {total} image(s) total")
            else:
                print(f"  [FETCHER] No new image saved for slot {slot_str}")

            # Sleep until the next 5-minute boundary
            next_slot = _next_slot_time()
            wait_sec = max(0, next_slot - time.time())
            next_slot_str = datetime.fromtimestamp(next_slot).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  Next fetch at: {next_slot_str} (in {int(wait_sec)}s)")
            print()
            time.sleep(wait_sec)

    except KeyboardInterrupt:
        print("\n")
        print("=" * 60)
        print("Stopped by user. Goodbye!")
        print("=" * 60)
    finally:
        release_lock(FETCHER_LOCK)

if __name__ == "__main__":
    main()
