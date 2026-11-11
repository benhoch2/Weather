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
    write_json(STATUS_FILE, status)
    log.info("Retrain finished: %s", status["result"])


if __name__ == "__main__":
    main()
