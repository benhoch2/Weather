"""Nowcast verification scores, accumulated as exact counts.

Per lead time and threshold we accumulate hits / misses / false alarms, so
scores can be aggregated over any set of forecasts exactly (the live ledger
stores the raw counts).  Derived scores:

* CSI (critical success index) = hits / (hits + misses + false alarms)
* POD (probability of detection) = hits / (hits + misses)
* FAR (false alarm ratio)        = false alarms / (hits + false alarms)
* MAE in mm/h over all pixels
"""

from __future__ import annotations

import numpy as np

import config


class Scores:
    def __init__(self, n_out: int = config.N_OUT, thresholds=config.THRESHOLDS):
        self.n_out = n_out
        self.thresholds = tuple(float(t) for t in thresholds)
        k = len(self.thresholds)
        self.hits = np.zeros((n_out, k), dtype=np.int64)
        self.misses = np.zeros((n_out, k), dtype=np.int64)
        self.false_alarms = np.zeros((n_out, k), dtype=np.int64)
        self.abs_err = np.zeros(n_out, dtype=np.float64)
        self.pixels = np.zeros(n_out, dtype=np.int64)
        self.forecasts = 0

    def update(self, pred: np.ndarray, obs: np.ndarray, valid: np.ndarray | None = None) -> None:
        """pred, obs: (n_out, H, W) mm/h.  valid: optional (n_out,) bool for leads with an observation."""
        pred = np.nan_to_num(np.asarray(pred, dtype=np.float32), nan=0.0)
        obs = np.asarray(obs, dtype=np.float32)
        for lead in range(self.n_out):
            if valid is not None and not valid[lead]:
                continue
            p, o = pred[lead], obs[lead]
            for j, thr in enumerate(self.thresholds):
                pp, oo = p >= thr, o >= thr
                self.hits[lead, j] += int(np.count_nonzero(pp & oo))
                self.misses[lead, j] += int(np.count_nonzero(~pp & oo))
                self.false_alarms[lead, j] += int(np.count_nonzero(pp & ~oo))
            self.abs_err[lead] += float(np.abs(p - o).sum(dtype=np.float64))
            self.pixels[lead] += p.size
        self.forecasts += 1

    def merge(self, other: "Scores") -> "Scores":
        self.hits += other.hits
        self.misses += other.misses
        self.false_alarms += other.false_alarms
        self.abs_err += other.abs_err
        self.pixels += other.pixels
        self.forecasts += other.forecasts
        return self

    # ── derived ──────────────────────────────────────────────────────────────
    @staticmethod
    def _ratio(num, den):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / np.maximum(den, 1), np.nan)

    def csi(self):
        return self._ratio(self.hits, self.hits + self.misses + self.false_alarms)

    def pod(self):
        return self._ratio(self.hits, self.hits + self.misses)

    def far(self):
        return self._ratio(self.false_alarms, self.hits + self.false_alarms)

    def mae(self):
        return self._ratio(self.abs_err, self.pixels.astype(np.float64))

    def mean_csi(self) -> float:
        """Single number used to compare models: mean CSI over leads and thresholds."""
        c = self.csi()
        return float(np.nanmean(c)) if np.isfinite(c).any() else float("nan")

    def summary(self) -> dict:
        def per_thr(arr):
            return {f"{t:g}": [None if not np.isfinite(v) else round(float(v), 4) for v in arr[:, j]]
                    for j, t in enumerate(self.thresholds)}
        return {
            "forecasts": self.forecasts,
            "lead_minutes": [(k + 1) * config.STEP // 60 for k in range(self.n_out)],
            "csi": per_thr(self.csi()),
            "pod": per_thr(self.pod()),
            "far": per_thr(self.far()),
            "mae": [None if not np.isfinite(v) else round(float(v), 4) for v in self.mae()],
            "mean_csi": round(self.mean_csi(), 4) if np.isfinite(self.mean_csi()) else None,
        }

    # ── (de)serialisation of raw counts ──────────────────────────────────────
    def to_counts(self) -> dict:
        return {"thresholds": list(self.thresholds), "hits": self.hits.tolist(),
                "misses": self.misses.tolist(), "false_alarms": self.false_alarms.tolist(),
                "abs_err": self.abs_err.tolist(), "pixels": self.pixels.tolist(),
                "forecasts": self.forecasts}

    @classmethod
    def from_counts(cls, d: dict) -> "Scores":
        s = cls(n_out=len(d["abs_err"]), thresholds=d["thresholds"])
        s.hits = np.array(d["hits"], dtype=np.int64)
        s.misses = np.array(d["misses"], dtype=np.int64)
        s.false_alarms = np.array(d["false_alarms"], dtype=np.int64)
        s.abs_err = np.array(d["abs_err"], dtype=np.float64)
        s.pixels = np.array(d["pixels"], dtype=np.int64)
        s.forecasts = int(d.get("forecasts", 0))
        return s
