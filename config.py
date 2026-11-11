"""Central configuration: paths, timing, model shape, IMS endpoints and legend.

Every value can be overridden with an environment variable of the same name
prefixed by ``WEATHER_`` where noted.
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("WEATHER_DATA_DIR", PROJECT_DIR / "data"))
RAW_DIR = DATA_DIR / "ims" / "raw"            # raw IMS PNGs: YYYY/MM/DD/<utc_epoch>.png
MISSING_FILE = DATA_DIR / "ims" / "missing.txt"  # UTC epochs confirmed absent on IMS
CACHE_DIR = DATA_DIR / "ims" / "cache"        # per-day downsampled arrays
STATUS_DIR = DATA_DIR / "status"              # small JSON status files for the viewer
PRED_DIR = DATA_DIR / "predictions"
EVAL_DIR = DATA_DIR / "eval"
CONTROL_FILE = DATA_DIR / "control.json"      # viewer -> supervisor switches
MODELS_DIR = Path(os.environ.get("WEATHER_MODELS_DIR", PROJECT_DIR / "models"))
CURRENT_MODEL_FILE = MODELS_DIR / "current.json"

# ── Timing / shapes ───────────────────────────────────────────────────────────
STEP = 300                 # seconds between radar frames
N_IN = 12                  # input frames (60 min of history)
N_OUT = 12                 # forecast frames (60 min ahead)
NATIVE_SIZE = 940          # IMS PNG width/height
GRID = int(os.environ.get("WEATHER_GRID", 256))  # model resolution

# ── IMS endpoints ─────────────────────────────────────────────────────────────
IMS_BASE = "https://ims.gov.il"
IMS_LIST_URL = IMS_BASE + "/radar_satellite/0"
IMS_ARCHIVE_URL = (IMS_BASE + "/sites/default/files/ims_data/map_images/"
                   "IMSRadar4GIS/IMSRadar4GIS_{stamp}_0.png")
IMS_TZ = "Asia/Jerusalem"  # archive file names and JSON times are local time
USER_AGENT = "weather-radar-nowcast/2.0 (personal research project)"
REQUEST_INTERVAL = float(os.environ.get("WEATHER_REQUEST_INTERVAL", 0.5))  # seconds between requests

# Geographic bounds of the IMSRadar PNG, taken from IMS's own Leaflet code.
# ((south, west), (north, east)) in degrees.
RADAR_BOUNDS = ((29.4468656950938161, 31.7662931774178858),
                (34.5318966003582659, 37.864813247553009))

# IMS rain legend, ascending: (hex colour, mm/h at that colour).
# The PNGs use a continuous ramp blended between these colours.
LEGEND = [
    ("#0165ef", 0.1), ("#00c3c9", 0.2), ("#00a09b", 0.7), ("#008c4d", 1.2),
    ("#00b02c", 2.0), ("#01d31c", 4.0), ("#12f218", 6.0), ("#7cff21", 9.0),
    ("#fefd19", 13.0), ("#ffcf00", 18.0), ("#ffa800", 24.0), ("#ff7d01", 30.0),
    ("#fb3f00", 40.0), ("#e10a12", 50.0), ("#d00078", 100.0), ("#ff00fe", 200.0),
]

# ── Evaluation ────────────────────────────────────────────────────────────────
THRESHOLDS = (0.5, 2.0, 5.0)   # mm/h thresholds for CSI / POD / FAR
RAINY_PIXELS = int(os.environ.get("WEATHER_RAINY_PIXELS", 1000))  # px >= 0.5 mm/h at native res to call a frame "rainy"

# ── Services ──────────────────────────────────────────────────────────────────
VIEWER_HOST = os.environ.get("WEATHER_RADAR_VIEWER_HOST", "127.0.0.1")
VIEWER_PORT = int(os.environ.get("WEATHER_RADAR_VIEWER_PORT", 5050))
FETCH_INTERVAL = 120        # seconds between IMS list polls
GAP_FILL_HOURS = 48         # how far back the live fetcher repairs gaps
PREDICTION_RETENTION_DAYS = int(os.environ.get("WEATHER_PREDICTION_RETENTION_DAYS", 30))
EXIT_NO_MODEL = 3           # predictor exit code: no trained model -> don't restart
