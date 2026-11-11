"""Live IMS radar fetcher.

* Every FETCH_INTERVAL seconds: read the IMS frame list and download any new frames.
* At startup and hourly: repair gaps in the last GAP_FILL_HOURS from the IMS archive,
  so downtime or sleep never loses data.
* Never deletes anything (frames are 2-15 KB each).
* Writes data/status/fetcher.json for the web viewer.
"""

from __future__ import annotations

import logging
import time

import requests

import config
from backfill import Fetcher
from common import local_str, setup_logging, write_json
from ims_source import ImsClient
from radar_store import RadarStore, floor_slot

log = logging.getLogger("fetcher")
STATUS_FILE = config.STATUS_DIR / "fetcher.json"


def poll_latest(f: Fetcher, client: ImsClient) -> dict:
    listing = client.list_latest()
    new = 0
    for t, url in listing["frames"]:
        if f.store.has(t):
            continue
        try:
            data = client.download(url)
        except requests.RequestException as e:
            log.warning("download failed for %s: %s", local_str(t), e)
            continue
        if data and f.store.is_stale_repeat(t, data):
            log.warning("IMS frame %s repeats the previous one (stuck radar) - skipped", local_str(t))
            f.store.mark_missing(t)
        elif data:
            f.store.save(t, data)
            new += 1
    if new:
        log.info("Saved %d new frame(s); latest %s", new, local_str(listing["frames"][-1][0]))
    return listing


def fill_gaps(f: Fetcher, hours: int) -> int:
    now = floor_slot(time.time())
    slots = range(now - hours * 3600, now - config.STEP, config.STEP)
    todo = [t for t in slots if not f.store.has(t) and t not in f.store.missing]
    if not todo:
        return 0
    log.info("Gap fill: %d missing slot(s) in the last %dh", len(todo), hours)
    before = f.saved
    for t in todo:
        f.ensure(t)
    filled = f.saved - before
    log.info("Gap fill: recovered %d frame(s)", filled)
    return filled


def write_status(store: RadarStore, active: bool | None, error: str | None) -> None:
    latest = store.latest_time()
    now = time.time()
    recent = store.list_times(int(now) - 3600, int(now))
    write_json(STATUS_FILE, {
        "updated_at": int(now),
        "latest_frame": latest,
        "latest_frame_local": local_str(latest, "%Y-%m-%d %H:%M") if latest else None,
        "lag_minutes": round((now - latest) / 60, 1) if latest else None,
        "frames_last_hour": len(recent),
        "radar_active": active,
        "last_error": error,
    })


def main() -> None:
    setup_logging("fetcher")
    store, client = RadarStore(), ImsClient()
    f = Fetcher(store, client)
    log.info("IMS fetcher started (poll every %ds, gap fill %dh)", config.FETCH_INTERVAL, config.GAP_FILL_HOURS)
    last_gap_fill = 0.0
    while True:
        error, active = None, None
        try:
            listing = poll_latest(f, client)
            active = listing["active"]
            write_status(store, active, None)
            if time.monotonic() - last_gap_fill > 3600:
                fill_gaps(f, config.GAP_FILL_HOURS)
                last_gap_fill = time.monotonic()
        except Exception as e:  # keep running through network hiccups
            error = f"{type(e).__name__}: {e}"
            log.warning("Fetch cycle failed: %s", error)
        write_status(store, active, error)
        time.sleep(config.FETCH_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
