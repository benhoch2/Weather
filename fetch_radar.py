import requests
from PIL import Image
from io import BytesIO
import time

def fetch_radar_last_frame():
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
        
        print(f"Fetching radar image from: {radar_image_url}")
        print(f"Content-Type: {response.headers.get('Content-Type')}")
        print(f"Content size: {len(response.content)} bytes")
        
        # Get current epoch timestamp for filename
        current_timestamp = int(time.time())
        filename = f"data/radar_images/radar_{current_timestamp}.png"
        
        # Open and process the image
        img = Image.open(BytesIO(response.content))
        
        # If it's an animated PNG (APNG), get the last frame
        if hasattr(img, 'n_frames') and img.n_frames > 1:
            # Seek to the last frame
            img.seek(img.n_frames - 1)
            print(f"Animated PNG detected with {img.n_frames} frames. Extracting last frame.")
        else:
            print("Single frame image detected.")
        
        # Save the image
        img.save(filename)
        print(f"Radar image saved as: {filename}")
        print(f"Saved timestamp: {current_timestamp}")
        
        return filename
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching radar data: {e}")
        return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

if __name__ == "__main__":
    fetch_radar_last_frame()
