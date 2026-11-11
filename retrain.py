"""Scheduled, gated retraining (run nightly by start.py, or by hand).

Fine-tunes the live model on all data, but only if enough new rainy frames have
arrived since the last run (so it does nothing in the dry season).  The new
checkpoint replaces the live model only if it scores better on the same
validation windows (train.promote_if_better).

    python retrain.py            # respects the new-data threshold
    python retrain.py --force    # retrain regardless
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import timedelta

import config
from common import local_str, read_json, setup_logging, write_json
from dataset import Q_RAIN, RAINY_GRID_PX
from model import current_model_path
from radar_store import RadarStore, day_start, utc_day

log = logging.getLogger("retrain")
STATUS_FILE = config.STATUS_DIR / "retrain.json"
MIN_NEW_RAINY_FRAMES = 36      # ~3 hours of rain


def new_rainy_frames(store: RadarStore, since: int) -> int:
    count, d, today = 0, utc_day(since), utc_day(int(time.time()))
    while d <= today:
        frames, present = store.load_day(d)
        start = day_start(d)
        for idx in range(len(present)):
            if present[idx] and start + idx * config.STEP > since:
                count += int((frames[idx] >= Q_RAIN).sum() >= RAINY_GRID_PX)
        d += timedelta(days=1)
    return count


def main() -> None:
    setup_logging("retrain")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--no-sync", action="store_true", help="skip downloading recent data from IMS first")
    ap.add_argument("--no-push", action="store_true", help="don't copy a promoted model to the Pi")
    a = ap.parse_args()

    prev = read_json(STATUS_FILE, {}) or {}
    since = int(prev.get("data_until") or time.time() - 7 * 86400)
    now = int(time.time())
    status = {"last_run": now, "last_run_local": local_str(now), "data_until": prev.get("data_until")}

    init = current_model_path()
    if init is None:
        status.update(result="skipped", reason="no current model - run train.py first")
        write_json(STATUS_FILE, status)
        log.info(status["reason"])
        return

    if not a.no_sync:
        # The live fetcher runs on the Pi; bring this machine's archive up to date from IMS.
        from backfill import run as backfill_run
        log.info("Syncing radar data since %s from the IMS archive", local_str(since))
        backfill_run(min(since, now - 2 * 86400), now, margin_h=3, rain_px=config.RAINY_PIXELS, all_frames=False)

    n_new = new_rainy_frames(RadarStore(), since)
    status["new_rainy_frames"] = n_new
    if n_new < MIN_NEW_RAINY_FRAMES and not a.force:
        status.update(result="skipped", reason=f"only {n_new} new rainy frames since {local_str(since)}")
        write_json(STATUS_FILE, status)
        log.info(status["reason"])
        return

    from train import train
    log.info("Retraining from %s (%d new rainy frames)", init.name, n_new)
    status.update(result="running")
    write_json(STATUS_FILE, status)
    try:
        decision = train(epochs=a.epochs, steps=500, lr=2e-4, patience=3, init=init)
    except SystemExit as e:
        status.update(result="failed", reason=str(e))
        write_json(STATUS_FILE, status)
        return
    status.update(result="promoted" if decision.get("promoted") else "kept_current",
                  data_until=now, **decision)
    if decision.get("promoted") and decision.get("onnx") and not a.no_push:
        try:   # ship the better model to the Pi (needs SSH access; see push_model.py)
            subprocess.run([sys.executable, "push_model.py"], cwd=config.PROJECT_DIR, check=True, timeout=600)
            status["pushed_to_pi"] = True
        except Exception as e:
            status["pushed_to_pi"] = False
            log.warning("Promoted, but pushing to the Pi failed: %s", e)
    write_json(STATUS_FILE, status)
    log.info("Retrain finished: %s", status["result"])


if __name__ == "__main__":
    main()
