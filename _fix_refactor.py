"""One-shot script to refactor fetch_radar_continuous.py and run_predictions_persistent.py
to use the shared process_utils module. Delete this file after running."""
import re

# === Fix fetch_radar_continuous.py ===
with open("fetch_radar_continuous.py", "r", encoding="utf-8") as f:
    content = f.read()

# Remove old subprocess/sys imports (no longer needed for lock logic)
content = content.replace("import subprocess\n", "")
content = content.replace("import sys\n", "")

# Add shared import after 'import os'
content = content.replace(
    "import os\n\nDEFAULT_MAX_STORED_IMAGES",
    "import os\nfrom process_utils import acquire_lock, release_lock\n\nDEFAULT_MAX_STORED_IMAGES",
)

# Remove the _is_pid_running, acquire_fetcher_lock, release_fetcher_lock functions
# They sit between FETCHER_LOCK line and fetch_radar_image()
pattern = r'(FETCHER_LOCK = Path\(__file__\)\.parent / "fetcher\.lock")\s*\n.*?(?=def fetch_radar_image)'
content = re.sub(pattern, r'\1\n\n\ndef fetch_radar_image', content, flags=re.DOTALL)

with open("fetch_radar_continuous.py", "w", encoding="utf-8") as f:
    f.write(content)
print("fetch_radar_continuous.py updated")


# === Fix run_predictions_persistent.py ===
with open("run_predictions_persistent.py", "r", encoding="utf-8") as f:
    content = f.read()

# Add shared import after 'from pathlib import Path'
content = content.replace(
    "from pathlib import Path\n\n# ",
    "from pathlib import Path\nfrom process_utils import acquire_lock, release_lock\n\n# ",
)

# Remove everything between PREDICTOR_LOCK definition and log_file definition
pattern = r'(PREDICTOR_LOCK = Path\(__file__\)\.parent / "predictor\.lock")\s*\n.*?(?=log_file = Path)'
content = re.sub(pattern, r'\1\n\n', content, flags=re.DOTALL)

with open("run_predictions_persistent.py", "w", encoding="utf-8") as f:
    f.write(content)
print("run_predictions_persistent.py updated")
