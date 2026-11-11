"""Training/evaluation data: a time-ordered array of cached frames + valid windows.

Whole UTC days are assigned to train / val / test by a hash of the date, so
the split is stable as data grows and no window straddles two splits.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date

import numpy as np

import config
from radar_store import SLOTS_PER_DAY, RadarStore, day_start, decode_mm, encode_mm

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
Q_RAIN = int(encode_mm(np.array([0.5]))[0])        # uint8 code for 0.5 mm/h
RAINY_GRID_PX = int(0.002 * config.GRID * config.GRID)  # >= 0.2% of the grid raining


def split_of(d: date) -> str:
    h = int(hashlib.md5(d.isoformat().encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else "val" if h == 1 else "train"


class Timeline:
    """All cached frames for the given days, concatenated in time order."""

    def __init__(self, store: RadarStore, days: list[date]):
        days = sorted(set(days))
        frames, present, times, split = [], [], [], []
        for i, d in enumerate(days, 1):
            f, p = store.load_day(d)
            if not p.any():
                continue
            frames.append(f)
            present.append(p)
            times.append(day_start(d) + np.arange(SLOTS_PER_DAY, dtype=np.int64) * config.STEP)
            split.append(np.full(SLOTS_PER_DAY, SPLITS.index(split_of(d)), dtype=np.int8))
            if i % 25 == 0:
                log.info("loaded %d/%d days", i, len(days))
        if not frames:
            raise RuntimeError("No radar data on disk. Run backfill.py first.")
        self.frames = np.concatenate(frames)            # (N, G, G) uint8
        self.present = np.concatenate(present)
        self.times = np.concatenate(times)
        self.split = np.concatenate(split)
        self.rain_px = np.where(self.present, (self.frames >= Q_RAIN).sum(axis=(1, 2)), 0)
        log.info("Timeline: %d days, %d frames present", len(frames), int(self.present.sum()))

    def windows(self, split: str, n_in: int = config.N_IN, n_out: int = config.N_OUT,
                min_targets: int = 1) -> np.ndarray:
        """Start indices of usable windows within one split.

        All n_in input frames must be present (no gaps in the past hour).  The
        n_out target frames may have holes; at least `min_targets` must exist.
        Missing targets are masked out of the loss and the scores.
        """
        L = n_in + n_out
        n = len(self.times) - L + 1
        if n <= 0:
            return np.zeros(0, dtype=np.int64)
        cs = np.concatenate([[0], np.cumsum(self.present.astype(np.int64))])
        inputs_complete = (cs[n_in:n_in + n] - cs[:n]) == n_in
        targets_present = cs[L:L + n] - cs[n_in:n_in + n]
        contiguous = (self.times[L - 1:L - 1 + n] - self.times[:n]) == (L - 1) * config.STEP
        sid = SPLITS.index(split)
        same_split = (self.split[:n] == sid) & (self.split[L - 1:L - 1 + n] == sid)
        return np.flatnonzero(inputs_complete & (targets_present >= min_targets) & contiguous & same_split)

    def target_mask(self, i: int, n_in: int = config.N_IN, n_out: int = config.N_OUT) -> np.ndarray:
        """Bool (n_out,) - which target frames of window i were observed."""
        return self.present[i + n_in:i + n_in + n_out].copy()

    def rainy(self, starts: np.ndarray, n_in: int = config.N_IN, n_out: int = config.N_OUT) -> np.ndarray:
        """Bool per window: rain in the last input frame or any target frame."""
        idx = starts[:, None] + np.arange(n_in - 1, n_in + n_out)[None, :]
        return (self.rain_px[idx] >= RAINY_GRID_PX).any(axis=1)

    def window_mm(self, i: int, n_in: int = config.N_IN, n_out: int = config.N_OUT):
        """(inputs (n_in,G,G), targets (n_out,G,G), target mask (n_out,)) in mm/h for window start i."""
        block = decode_mm(self.frames[i:i + n_in + n_out])
        return block[:n_in], block[n_in:], self.target_mask(i, n_in, n_out)


def load_timeline(store: RadarStore | None = None, workers: int = 6) -> Timeline:
    """Refresh the per-day cache in parallel, then load everything on disk."""
    from radar_store import build_cache
    store = store or RadarStore()
    days = store.days_on_disk()
    build_cache(days, workers=workers, grid=store.grid)
    return Timeline(store, days)
