"""Israel Meteorological Service (IMS) rain radar: listing, downloading, decoding.

IMS publishes a transparent 940x940 PNG every 5 minutes containing only the
rain layer (no basemap).  Colours follow a continuous ramp blended between the
16 legend colours, so each pixel colour is projected onto that ramp and
converted to a rain rate in mm/h (interpolated on a log scale).

All times in this project are UTC epoch seconds.  IMS names files by Israel
local time; conversion happens only in this module.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from io import BytesIO

import numpy as np
import requests
from PIL import Image

import config
from common import LOCAL_TZ

log = logging.getLogger(__name__)

# ── Time helpers ──────────────────────────────────────────────────────────────

def _offsets_differ(naive: datetime) -> bool:
    """True if a naive local time is ambiguous (autumn) or non-existent (spring)."""
    a = naive.replace(tzinfo=LOCAL_TZ, fold=0).utcoffset()
    b = naive.replace(tzinfo=LOCAL_TZ, fold=1).utcoffset()
    return a != b


def local_to_utc(naive: datetime) -> int | None:
    """Naive Israel-local datetime -> UTC epoch, or None if ambiguous/non-existent."""
    if _offsets_differ(naive):
        return None
    return int(naive.replace(tzinfo=LOCAL_TZ).timestamp())


def utc_to_stamp(t: int) -> str | None:
    """UTC epoch -> IMS archive stamp ``YYYYMMDDHHMM`` (local), or None if ambiguous."""
    local = datetime.fromtimestamp(t, LOCAL_TZ).replace(tzinfo=None)
    if _offsets_differ(local):
        return None
    return local.strftime("%Y%m%d%H%M")


def parse_ims_time(s: str) -> int | None:
    """Parse an IMS ``forecast_time`` such as ``2026-11-11 06:05:00`` (local)."""
    return local_to_utc(datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S"))


def archive_url(t: int) -> str | None:
    stamp = utc_to_stamp(t)
    return None if stamp is None else config.IMS_ARCHIVE_URL.format(stamp=stamp)


# ── Colour ramp ───────────────────────────────────────────────────────────────

def _hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


_LEGEND_RGB = np.array([_hex_rgb(h) for h, _ in config.LEGEND], dtype=np.float32)
_LEGEND_LOG = np.log10([v for _, v in config.LEGEND]).astype(np.float32)


def _dense_ramp(steps: int = 64):
    pts, vals = [], []
    for i in range(len(_LEGEND_RGB) - 1):
        s = np.linspace(0.0, 1.0, steps, endpoint=False, dtype=np.float32)[:, None]
        pts.append(_LEGEND_RGB[i] * (1 - s) + _LEGEND_RGB[i + 1] * s)
        vals.append(_LEGEND_LOG[i] * (1 - s[:, 0]) + _LEGEND_LOG[i + 1] * s[:, 0])
    pts.append(_LEGEND_RGB[-1:])
    vals.append(_LEGEND_LOG[-1:])
    return np.concatenate(pts), np.concatenate(vals)


_RAMP_RGB, _RAMP_LOG = _dense_ramp()
_colour_cache: dict[int, float] = {}
FAR_COLOUR_DIST = 60.0  # RGB distance beyond which a colour is "not on the ramp"


def _colours_to_mm(packed: np.ndarray) -> np.ndarray:
    """Map packed 0xRRGGBB colours to mm/h using the dense ramp (memoised)."""
    out = np.empty(len(packed), dtype=np.float32)
    todo = [i for i, c in enumerate(packed) if int(c) not in _colour_cache]
    if todo:
        cols = packed[todo]
        rgb = np.stack([(cols >> 16) & 255, (cols >> 8) & 255, cols & 255], axis=1).astype(np.float32)
        d2 = ((rgb[:, None, :] - _RAMP_RGB[None, :, :]) ** 2).sum(axis=2)
        nearest = d2.argmin(axis=1)
        dist = np.sqrt(d2[np.arange(len(todo)), nearest])
        if (dist > FAR_COLOUR_DIST).any():
            log.warning("IMS image has %d colour(s) far from the legend ramp (max dist %.0f)",
                        int((dist > FAR_COLOUR_DIST).sum()), float(dist.max()))
        for c, v in zip(cols, 10.0 ** _RAMP_LOG[nearest]):
            _colour_cache[int(c)] = float(v)
    for i, c in enumerate(packed):
        out[i] = _colour_cache[int(c)]
    return out


def decode_png(data: bytes | str) -> np.ndarray:
    """Decode an IMS radar PNG (bytes or path) to a float32 rain-rate field in mm/h."""
    img = Image.open(BytesIO(data) if isinstance(data, (bytes, bytearray)) else data)
    if img.size != (config.NATIVE_SIZE, config.NATIVE_SIZE):
        raise ValueError(f"unexpected IMS image size {img.size}")
    rgba = np.asarray(img.convert("RGBA"))
    rain = np.zeros(rgba.shape[:2], dtype=np.float32)
    mask = rgba[..., 3] >= 128
    if mask.any():
        px = rgba[mask][:, :3].astype(np.uint32)
        packed = (px[:, 0] << 16) | (px[:, 1] << 8) | px[:, 2]
        uniq, inv = np.unique(packed, return_inverse=True)
        rain[mask] = _colours_to_mm(uniq)[inv]
    return rain


def colorize(mm: np.ndarray, min_mm: float = 0.1) -> Image.Image:
    """Render a mm/h field as a transparent RGBA image in IMS legend colours."""
    mm = np.asarray(mm, dtype=np.float32)
    rgba = np.zeros(mm.shape + (4,), dtype=np.uint8)
    mask = mm >= min_mm
    if mask.any():
        pos = np.interp(np.log10(mm[mask]), _LEGEND_LOG, np.arange(len(_LEGEND_LOG)))
        lo = np.floor(pos).astype(int).clip(0, len(_LEGEND_RGB) - 2)
        frac = (pos - lo)[:, None]
        rgb = _LEGEND_RGB[lo] * (1 - frac) + _LEGEND_RGB[lo + 1] * frac
        rgba[mask, :3] = rgb.round().astype(np.uint8)
        rgba[mask, 3] = 255
    return Image.fromarray(rgba, "RGBA")


def legend_from_json(colorbar: dict) -> list[tuple[str, float]]:
    """Extract (hex, mm/h) ascending from the IMS ``radar_colorbar`` object."""
    items = colorbar.get("items", {})
    items = items.values() if isinstance(items, dict) else items
    rows = sorted(((str(i["color"]).lower(), float(i["mark"])) for i in items), key=lambda r: r[1])
    return rows


# ── HTTP client ───────────────────────────────────────────────────────────────

class ImsClient:
    """Polite IMS client: one session, a minimum gap between requests, retries."""

    def __init__(self, interval: float = config.REQUEST_INTERVAL):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = config.USER_AGENT
        self.interval = interval
        self._last = 0.0

    def _get(self, url: str, timeout: float = 20, retries: int = 3) -> requests.Response:
        for attempt in range(retries + 1):
            wait = self._last + self.interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.session.get(url, timeout=timeout)
            except requests.RequestException:
                if attempt == retries:
                    raise
                time.sleep(2 * 2 ** attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == retries:
                    r.raise_for_status()
                time.sleep(5 * 2 ** attempt)
                continue
            return r
        raise RuntimeError("unreachable")

    def list_latest(self) -> dict:
        """Return {'frames': [(utc, url), ...], 'active': bool, 'legend': [...]}."""
        r = self._get(config.IMS_LIST_URL)
        r.raise_for_status()
        data = r.json()["data"]
        items = data.get("types", {}).get("IMSRadar", [])
        items = list(items.values()) if isinstance(items, dict) else items
        frames = []
        for it in items:
            if not isinstance(it, dict) or it.get("type") != "IMSRadar4GIS":
                continue
            t = parse_ims_time(it["forecast_time"])
            if t is not None:
                frames.append((t, config.IMS_BASE + it["file_name"]))
        legend = None
        if isinstance(data.get("radar_colorbar"), dict):
            legend = legend_from_json(data["radar_colorbar"])
            expected = [(h.lower(), v) for h, v in config.LEGEND]
            if legend != expected:
                log.warning("IMS legend changed! Update config.LEGEND. Got: %s", legend)
        return {"frames": sorted(frames), "active": str(data.get("rain_radar_is_active", "1")) == "1",
                "legend": legend}

    def download(self, url: str) -> bytes | None:
        """PNG bytes, or None if IMS says the file does not exist."""
        r = self._get(url)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        if not r.content.startswith(b"\x89PNG"):
            return None
        return r.content

    def download_frame(self, t: int) -> bytes | None:
        url = archive_url(t)
        return None if url is None else self.download(url)
