"""Download historical IMS radar frames for training.

Stage A ("scan") fetches one frame per hour across the date range and records
how much rain each shows.  Stage B ("fill") downloads every 5-minute frame
within +/- MARGIN hours of any rainy hour.  Use --all to download every frame
in the range regardless of rain (handy for small test ranges).

Requests are throttled (config.REQUEST_INTERVAL) and everything is resumable:
files already on disk and frames IMS confirmed missing are skipped.

Examples::

    python backfill.py                                  # 2025-01-01 .. now
    python backfill.py --from 2026-03-22 --to 2026-03-23 --all
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone

import requests

import config
from common import local_str, read_json, setup_logging, write_json
from ims_source import ImsClient, decode_png
from radar_store import RadarStore, floor_slot

log = logging.getLogger("backfill")
SCAN_FILE = config.DATA_DIR / "ims" / "scan.json"   # {utc_hour: px >= 0.5 mm/h}


def parse_day(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def rain_pixels(store: RadarStore, t: int) -> int:
    return int((store.load_native(t) >= 0.5).sum())


class Fetcher:
    """Download-if-needed with progress accounting."""

    def __init__(self, store: RadarStore, client: ImsClient):
        self.store, self.client = store, client
        self.requests = self.saved = self.absent = self.errors = 0

    def ensure(self, t: int) -> bool:
        """True if the frame is on disk afterwards."""
        if self.store.has(t):
            return True
        if t in self.store.missing:
            return False
        try:
            self.requests += 1
            data = self.client.download_frame(t)
        except requests.RequestException as e:
            self.errors += 1
            log.warning("download failed for %s: %s", local_str(t), e)
            return False
        if data is None:
            self.absent += 1
            if t < time.time() - 6 * 3600:   # recent frames may still appear
                self.store.mark_missing(t)
            return False
        try:
            rain = decode_png(data)           # validate before saving
        except Exception as e:
            self.errors += 1
            log.warning("invalid image for %s: %s", local_str(t), e)
            return False
        if self.store.is_stale_repeat(t, data, rain):
            self.absent += 1
            self.store.mark_missing(t)
            log.warning("stale repeat of the previous frame at %s - treated as missing", local_str(t))
            return False
        self.store.save(t, data)
        self.saved += 1
        return True


def run(t_from: int, t_to: int, margin_h: int, rain_px: int, all_frames: bool) -> None:
    store, client = RadarStore(), ImsClient()
    f = Fetcher(store, client)
    t_to = min(t_to, floor_slot(time.time()) - config.STEP)

    if all_frames:
        slots = list(range(floor_slot(t_from), t_to + 1, config.STEP))
        _download(f, slots, "all frames")
        return

    # ── Stage A: hourly scan ─────────────────────────────────────────────────
    scan: dict[str, int] = read_json(SCAN_FILE, {}) or {}
    hours = list(range(t_from // 3600 * 3600, t_to + 1, 3600))
    log.info("Stage A: scanning %d hourly frames %s .. %s", len(hours), local_str(hours[0]), local_str(hours[-1]))
    started = time.monotonic()
    for i, t in enumerate(hours, 1):
        key = str(t)
        if key not in scan and f.ensure(t):
            try:
                scan[key] = rain_pixels(store, t)
            except Exception as e:
                log.warning("cannot decode %s: %s", local_str(t), e)
        if i % 200 == 0 or i == len(hours):
            write_json(SCAN_FILE, scan)
            rainy = sum(1 for h in hours[:i] if scan.get(str(h), 0) >= rain_px)
            _progress("A", i, len(hours), started, f, f"rainy hours so far: {rainy}")
    write_json(SCAN_FILE, scan)

    # ── Stage B: fill around rainy hours ─────────────────────────────────────
    rainy = [t for t in hours if scan.get(str(t), 0) >= rain_px]
    wanted: set[int] = set()
    for t in rainy:
        lo, hi = t - margin_h * 3600, t + margin_h * 3600
        wanted.update(s for s in range(floor_slot(lo), hi + 1, config.STEP) if t_from <= s <= t_to)
    slots = sorted(wanted)
    log.info("Stage B: %d rainy hours -> %d five-minute slots (+/-%dh)", len(rainy), len(slots), margin_h)
    _download(f, slots, "B")


def _download(f: Fetcher, slots: list[int], stage: str) -> None:
    todo = [t for t in slots if not f.store.has(t) and t not in f.store.missing]
    log.info("Stage %s: %d slot(s), %d to download (~%.0f min)", stage, len(slots), len(todo),
             len(todo) * config.REQUEST_INTERVAL / 60)
    started = time.monotonic()
    for i, t in enumerate(todo, 1):
        f.ensure(t)
        if i % 500 == 0 or i == len(todo):
            _progress(stage, i, len(todo), started, f)


def _progress(stage, i, n, started, f: Fetcher, extra: str = "") -> None:
    rate = i / max(1e-6, time.monotonic() - started)
    eta = (n - i) / rate / 60 if rate > 0 else 0
    log.info("[%s] %d/%d  saved=%d absent=%d errors=%d  ETA %.0f min %s",
             stage, i, n, f.saved, f.absent, f.errors, eta, extra)


def main():
    setup_logging("backfill")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="t_from", default="2025-01-01", help="UTC date YYYY-MM-DD")
    ap.add_argument("--to", dest="t_to", default=None, help="UTC date YYYY-MM-DD (exclusive; default now)")
    ap.add_argument("--margin-hours", type=int, default=3)
    ap.add_argument("--rain-px", type=int, default=config.RAINY_PIXELS)
    ap.add_argument("--all", action="store_true", help="download every 5-min frame in range")
    a = ap.parse_args()
    t_from = parse_day(a.t_from)
    t_to = parse_day(a.t_to) - config.STEP if a.t_to else int(time.time())
    try:
        run(t_from, t_to, a.margin_hours, a.rain_px, a.all)
    except KeyboardInterrupt:
        log.info("Interrupted - progress is saved; rerun to resume.")


if __name__ == "__main__":
    main()
