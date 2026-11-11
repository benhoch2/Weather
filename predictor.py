"""Live nowcasting service.

* Watches the radar store; when a new frame arrives at time t0 it forecasts
  t0+5 ... t0+60 min from the 12 frames t0-55 ... t0 (one missing input may
  be filled from its neighbour).
* Saves data/predictions/<t0>/: pred.npz (mm/h), lead_XX.png overlays, meta.json.
* Once the observations exist (or 90 min have passed) it scores the forecast,
  together with persistence and optical-flow baselines for the same t0, and
  appends the raw counts to data/predictions/ledger.jsonl (kept forever).
* Never changes model weights (retraining is a separate, gated job).  Reloads
  the model when models/current.json changes.
* Exits with code 3 if there is no trained model, so the supervisor does not
  restart it in a loop.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from collections import OrderedDict

import numpy as np
from PIL import Image

import config
from baselines import optical_flow, persistence
from common import local_str, read_json, setup_logging, write_json
from ims_source import colorize
from metrics import Scores
from model import current_model_path, device, load_checkpoint, predict_mm
from radar_store import RadarStore

log = logging.getLogger("predictor")
STATUS_FILE = config.STATUS_DIR / "predictor.json"
LEDGER_FILE = config.PRED_DIR / "ledger.jsonl"
POLL_SECONDS = 30
OVERLAY_SIZE = 512          # forecast PNGs are upsampled for a smoother map overlay
EVAL_DEADLINE = config.N_OUT * config.STEP + 30 * 60


class FrameCache:
    """Small LRU cache of decoded grid frames (mm/h)."""

    def __init__(self, store: RadarStore, size: int = 64):
        self.store, self.size, self._d = store, size, OrderedDict()

    def get(self, t: int) -> np.ndarray | None:
        if t in self._d:
            self._d.move_to_end(t)
            return self._d[t]
        if not self.store.has(t):
            return None
        arr = self.store.load_grid(t)
        self._d[t] = arr
        if len(self._d) > self.size:
            self._d.popitem(last=False)
        return arr


def input_times(t0: int) -> list[int]:
    return [t0 - k * config.STEP for k in range(config.N_IN - 1, -1, -1)]


def target_times(t0: int) -> list[int]:
    """Forecast valid times: exactly t0 + 5 min ... t0 + 60 min."""
    return [t0 + k * config.STEP for k in range(1, config.N_OUT + 1)]


def gather_inputs(frames: FrameCache, t0: int) -> tuple[np.ndarray | None, list[int]]:
    """12 input frames ending at t0; one interior gap is filled from a neighbour."""
    times = input_times(t0)
    arrays = [frames.get(t) for t in times]
    missing = [t for t, a in zip(times, arrays) if a is None]
    if arrays[-1] is None or len(missing) > 1:
        return None, missing
    for i, a in enumerate(arrays):
        if a is None:
            arrays[i] = arrays[i - 1] if i > 0 else arrays[i + 1]
    return np.stack(arrays), missing


class Predictor:
    def __init__(self):
        self.store = RadarStore()
        self.frames = FrameCache(self.store)
        self.dev = device()
        self.model = None
        self.model_name = None
        self.model_mtime = None
        self.last_t0 = self._latest_prediction()
        self.state, self.detail = "starting", None

    # ── model ────────────────────────────────────────────────────────────────
    def ensure_model(self) -> bool:
        try:
            mtime = config.CURRENT_MODEL_FILE.stat().st_mtime
        except OSError:
            mtime = None
        if self.model is not None and mtime == self.model_mtime:
            return True
        path = current_model_path()
        if path is None:
            return self.model is not None
        model, meta = load_checkpoint(path, self.dev)
        if meta.get("grid") != config.GRID:
            log.error("Model %s was trained at grid %s, config is %s", path.name, meta.get("grid"), config.GRID)
            return self.model is not None
        self.model, self.model_name, self.model_mtime = model, path.stem, mtime
        log.info("Loaded model %s on %s", path.name, self.dev)
        return True

    # ── forecasting ──────────────────────────────────────────────────────────
    def _latest_prediction(self) -> int:
        best = 0
        if config.PRED_DIR.exists():
            for p in config.PRED_DIR.iterdir():
                if p.is_dir() and p.name.isdigit() and (p / "meta.json").exists():
                    best = max(best, int(p.name))
        return best

    def predict(self, t0: int) -> None:
        inputs, missing = gather_inputs(self.frames, t0)
        if inputs is None:
            self.state = "insufficient_data"
            self.detail = f"{len(missing)} of the last {config.N_IN} frames missing"
            log.warning("Cannot forecast from %s: %s", local_str(t0), self.detail)
            return
        started = time.perf_counter()
        pred = predict_mm(self.model, inputs, self.dev)
        out = config.PRED_DIR / str(t0)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "pred.npz", mm=pred.astype(np.float16))
        for k, field in enumerate(pred, 1):
            big = np.asarray(Image.fromarray(field.astype(np.float32), "F")
                             .resize((OVERLAY_SIZE, OVERLAY_SIZE), Image.Resampling.BILINEAR))
            colorize(big).save(out / f"lead_{k:02d}.png", optimize=True)
        write_json(out / "meta.json", {
            "t0": t0, "t0_local": local_str(t0), "model": self.model_name, "created": int(time.time()),
            "valid_times": target_times(t0), "inputs_filled": missing,
            "max_mm": float(pred.max()), "rain_px": int((pred >= 0.5).sum(axis=(1, 2)).max())})
        self.state, self.detail = "ok", None
        log.info("Forecast from %s done in %.2fs (max %.1f mm/h)%s", local_str(t0, "%H:%M"),
                 time.perf_counter() - started, float(pred.max()),
                 f" [filled {len(missing)} gap]" if missing else "")

    # ── evaluation ───────────────────────────────────────────────────────────
    def evaluate_matured(self) -> int:
        pending = 0
        now = time.time()
        for folder in sorted(config.PRED_DIR.glob("[0-9]*")):
            if not folder.is_dir() or (folder / "metrics.json").exists() or not (folder / "meta.json").exists():
                continue
            t0 = int(folder.name)
            targets = target_times(t0)
            have = [self.store.has(t) for t in targets]
            if not all(have) and now < t0 + EVAL_DEADLINE:
                pending += 1
                continue
            self._evaluate(folder, t0, targets, np.array(have))
        return pending

    def _evaluate(self, folder, t0: int, targets: list[int], valid: np.ndarray) -> None:
        if not valid.any():
            write_json(folder / "metrics.json", {"skipped": True, "reason": "no observations"})
            return
        pred = np.load(folder / "pred.npz")["mm"].astype(np.float32)
        obs = np.stack([self.frames.get(t) if v else np.zeros_like(pred[0]) for t, v in zip(targets, valid)])
        inputs, _ = gather_inputs(self.frames, t0)
        scores = {"model": Scores()}
        scores["model"].update(pred, obs, valid)
        if inputs is not None:
            for name, fn in (("persistence", persistence), ("optflow", optical_flow)):
                scores[name] = Scores()
                scores[name].update(fn(inputs), obs, valid)
        meta = read_json(folder / "meta.json", {})
        entry = {"t0": t0, "model": meta.get("model"), "valid_leads": int(valid.sum()),
                 "obs_rain_px": int((obs >= 0.5).sum(axis=(1, 2)).max()),
                 "counts": {k: s.to_counts() for k, s in scores.items()}}
        with open(LEDGER_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        write_json(folder / "metrics.json", {"t0": t0, "valid_leads": int(valid.sum()),
                                             "obs_rain_px": entry["obs_rain_px"],
                                             "scores": {k: s.summary() for k, s in scores.items()}})
        log.info("Evaluated forecast from %s: mean CSI model %.3f vs optflow %s", local_str(t0, "%H:%M"),
                 scores["model"].mean_csi(),
                 f"{scores['optflow'].mean_csi():.3f}" if "optflow" in scores else "n/a")

    def prune(self) -> None:
        cutoff = time.time() - config.PREDICTION_RETENTION_DAYS * 86400
        for folder in config.PRED_DIR.glob("[0-9]*"):
            if folder.is_dir() and folder.name.isdigit() and int(folder.name) < cutoff:
                shutil.rmtree(folder, ignore_errors=True)

    def write_status(self, pending: int) -> None:
        cur = read_json(config.CURRENT_MODEL_FILE, {}) or {}
        write_json(STATUS_FILE, {
            "updated_at": int(time.time()), "state": self.state, "detail": self.detail,
            "model": self.model_name, "model_promoted_at": cur.get("promoted_at"),
            "model_val_mean_csi": cur.get("val_mean_csi"),
            "last_t0": self.last_t0 or None, "last_t0_local": local_str(self.last_t0) if self.last_t0 else None,
            "pending_evaluations": pending, "device": str(self.dev)})

    # ── main loop ────────────────────────────────────────────────────────────
    def run(self) -> None:
        config.PRED_DIR.mkdir(parents=True, exist_ok=True)
        last_prune = 0.0
        while True:
            if not self.ensure_model():
                self.state, self.detail = "no_model", "No trained model - run train.py"
                self.write_status(0)
                log.error("No trained model found (%s). Run train.py first.", config.CURRENT_MODEL_FILE)
                sys.exit(config.EXIT_NO_MODEL)
            latest = self.store.latest_time()
            if latest and latest > self.last_t0:
                try:
                    self.predict(latest)
                except Exception as e:
                    self.state, self.detail = "error", f"{type(e).__name__}: {e}"
                    log.exception("Forecast failed")
                self.last_t0 = latest
            elif latest is None:
                self.state, self.detail = "waiting", "No radar frames yet"
            try:
                pending = self.evaluate_matured()
            except Exception:
                log.exception("Evaluation failed")
                pending = 0
            if time.monotonic() - last_prune > 3600:
                self.prune()
                last_prune = time.monotonic()
            self.write_status(pending)
            time.sleep(POLL_SECONDS)


def main() -> None:
    setup_logging("predictor")
    try:
        Predictor().run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
