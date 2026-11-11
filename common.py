"""Small shared helpers: logging, atomic JSON files, time formatting."""

import json
import logging
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from config import IMS_TZ

LOCAL_TZ = ZoneInfo(IMS_TZ)


def setup_logging(name: str) -> logging.Logger:
    """Line-buffered stdout logging (the supervisor adds its own prefix)."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger(name)


def sanitize(obj):
    """Replace NaN/inf floats with None so the result is valid JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar
        return sanitize(obj.item())
    return obj


def write_json(path: Path, data) -> None:
    """Atomically write JSON (readers never see a half-written file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(sanitize(data), f, indent=2, allow_nan=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: Path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def local_str(t: int, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Format a UTC epoch in Israel local time."""
    return datetime.fromtimestamp(t, LOCAL_TZ).strftime(fmt)


def utc_str(t: int, fmt: str = "%Y-%m-%d %H:%M") -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime(fmt)
