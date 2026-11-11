"""On-disk archive of IMS radar frames plus a downsampled per-day cache.

Layout::

    data/ims/raw/YYYY/MM/DD/<utc_epoch>.png    raw IMS PNGs (UTC day folders)
    data/ims/missing.txt                       UTC epochs IMS confirmed absent
    data/ims/cache/<grid>/YYYY-MM-DD.npz       frames (288, grid, grid) uint8 + present mask

A frame that exists but has no rain is valid data (zero rain).  A frame that
is absent is a gap.  Sequences are only built from gap-free 5-minute slots.
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image

import config
from ims_source import decode_png

log = logging.getLogger(__name__)

SLOTS_PER_DAY = 86400 // config.STEP
CACHE_VERSION = 1

# ── Log-scale uint8 encoding of rain rate ─────────────────────────────────────
Q_MIN, Q_MAX = 0.01, 200.0
_LOG_MIN = np.log10(Q_MIN)
_LOG_SPAN = np.log10(Q_MAX) - _LOG_MIN


def encode_mm(mm: np.ndarray) -> np.ndarray:
    """mm/h float -> uint8 (0 = dry, 1..255 log-spaced 0.01..200 mm/h, ~4% steps)."""
    mm = np.asarray(mm, dtype=np.float32)
    q = 1 + np.round((np.log10(np.clip(mm, Q_MIN, Q_MAX)) - _LOG_MIN) / _LOG_SPAN * 254)
    return np.where(mm < Q_MIN, 0, q).astype(np.uint8)


DECODE_LUT = np.concatenate([[0.0], 10 ** (np.arange(254 + 1) / 254 * _LOG_SPAN + _LOG_MIN)]).astype(np.float32)[:256]


def decode_mm(q: np.ndarray) -> np.ndarray:
    return DECODE_LUT[np.asarray(q, dtype=np.uint8)]


def downsample(mm: np.ndarray, grid: int = config.GRID) -> np.ndarray:
    """Area-average a native mm/h field to grid x grid."""
    if mm.shape[0] == grid:
        return mm.astype(np.float32)
    img = Image.fromarray(mm.astype(np.float32), mode="F")
    return np.asarray(img.resize((grid, grid), Image.Resampling.BOX), dtype=np.float32)


def floor_slot(t: int) -> int:
    return int(t) // config.STEP * config.STEP


def utc_day(t: int) -> date:
    return datetime.fromtimestamp(t, timezone.utc).date()


def day_start(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


class RadarStore:
    def __init__(self, raw_dir: Path = config.RAW_DIR, cache_dir: Path = config.CACHE_DIR,
                 missing_file: Path = config.MISSING_FILE, grid: int = config.GRID):
        self.raw_dir = Path(raw_dir)
        self.cache_dir = Path(cache_dir) / str(grid)
        self.missing_file = Path(missing_file)
        self.grid = grid
        self._missing: set[int] | None = None

    # ── raw files ────────────────────────────────────────────────────────────
    def day_dir(self, d: date) -> Path:
        return self.raw_dir / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.day:02d}"

    def path_for(self, t: int) -> Path:
        return self.day_dir(utc_day(t)) / f"{int(t)}.png"

    def has(self, t: int) -> bool:
        return self.path_for(t).exists()

    def save(self, t: int, data: bytes) -> Path:
        path = self.path_for(t)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return path

    def times_on_day(self, d: date) -> list[int]:
        folder = self.day_dir(d)
        if not folder.is_dir():
            return []
        out = []
        for p in folder.glob("*.png"):
            try:
                out.append(int(p.stem))
            except ValueError:
                continue
        return sorted(out)

    def list_times(self, t_start: int, t_end: int) -> list[int]:
        """Sorted UTC epochs on disk with t_start <= t <= t_end."""
        out = []
        d, last = utc_day(t_start), utc_day(t_end)
        while d <= last:
            out.extend(t for t in self.times_on_day(d) if t_start <= t <= t_end)
            d += timedelta(days=1)
        return out

    def days_on_disk(self) -> list[date]:
        days = []
        for y in sorted(self.raw_dir.glob("[0-9][0-9][0-9][0-9]")):
            for m in sorted(y.glob("[0-9][0-9]")):
                for dd in sorted(m.glob("[0-9][0-9]")):
                    try:
                        days.append(date(int(y.name), int(m.name), int(dd.name)))
                    except ValueError:
                        continue
        return days

    def latest_time(self, lookback_days: int = 3) -> int | None:
        today = datetime.now(timezone.utc).date()
        for k in range(lookback_days + 1):
            times = self.times_on_day(today - timedelta(days=k))
            if times:
                return times[-1]
        return None

    # ── missing list ─────────────────────────────────────────────────────────
    @property
    def missing(self) -> set[int]:
        if self._missing is None:
            self._missing = set()
            if self.missing_file.exists():
                for line in self.missing_file.read_text().split():
                    try:
                        self._missing.add(int(line))
                    except ValueError:
                        pass
        return self._missing

    def mark_missing(self, t: int) -> None:
        if t in self.missing:
            return
        self.missing.add(t)
        self.missing_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.missing_file, "a") as f:
            f.write(f"{int(t)}\n")

    # ── decoded frames ───────────────────────────────────────────────────────
    def load_native(self, t: int) -> np.ndarray:
        return decode_png(str(self.path_for(t)))

    def load_grid(self, t: int) -> np.ndarray:
        """mm/h at model resolution, decoded straight from the raw PNG."""
        return downsample(self.load_native(t), self.grid)

    # ── per-day cache ────────────────────────────────────────────────────────
    def cache_path(self, d: date) -> Path:
        return self.cache_dir / f"{d.isoformat()}.npz"

    def load_day(self, d: date) -> tuple[np.ndarray, np.ndarray]:
        """(frames uint8 (288, G, G), present bool (288,)) for a UTC day, rebuilding if stale."""
        n_raw = len(self.times_on_day(d))
        path = self.cache_path(d)
        if path.exists():
            try:
                with np.load(path) as z:   # closes the file (Windows can't replace open files)
                    if int(z["version"]) == CACHE_VERSION and int(z["n_raw"]) == n_raw:
                        return z["frames"], z["present"]
            except Exception:
                pass
        return self._build_day(d)

    def _build_day(self, d: date) -> tuple[np.ndarray, np.ndarray]:
        start = day_start(d)
        frames = np.zeros((SLOTS_PER_DAY, self.grid, self.grid), dtype=np.uint8)
        present = np.zeros(SLOTS_PER_DAY, dtype=bool)
        times = self.times_on_day(d)
        for t in times:
            idx = (t - start) // config.STEP
            if 0 <= idx < SLOTS_PER_DAY and (t - start) % config.STEP == 0:
                try:
                    frames[idx] = encode_mm(self.load_grid(t))
                    present[idx] = True
                except Exception as e:
                    log.warning("Skipping unreadable frame %s: %s", self.path_for(t), e)
        path = self.cache_path(d)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, frames=frames, present=present, n_raw=len(times), version=CACHE_VERSION)
        os.replace(tmp, path)
        return frames, present

    # ── windows ──────────────────────────────────────────────────────────────
    def window_times(self, t0: int, n_in: int = config.N_IN, n_out: int = 0) -> list[int]:
        """Slot times t0-(n_in-1)*STEP ... t0+n_out*STEP."""
        return [t0 + k * config.STEP for k in range(-(n_in - 1), n_out + 1)]

    def missing_slots(self, t0: int, n_in: int = config.N_IN, n_out: int = 0) -> list[int]:
        return [t for t in self.window_times(t0, n_in, n_out) if not self.has(t)]


def _build_one(args):
    d, grid = args
    store = RadarStore(grid=grid)
    store.load_day(d)
    return d


def build_cache(days: list[date], workers: int = 4, grid: int = config.GRID) -> None:
    """Build/refresh the per-day cache for many days in parallel."""
    if workers <= 1:
        for d in days:
            _build_one((d, grid))
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, d in enumerate(ex.map(_build_one, [(d, grid) for d in days]), 1):
            if i % 10 == 0 or i == len(days):
                log.info("cache: %d/%d days (%s)", i, len(days), d)


if __name__ == "__main__":
    from common import setup_logging
    setup_logging("radar_store")
    ap = argparse.ArgumentParser(description="Build the downsampled per-day cache")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    store = RadarStore()
    days = store.days_on_disk()
    log.info("Building cache for %d day(s) with %d worker(s)", len(days), args.workers)
    build_cache(days, args.workers)
