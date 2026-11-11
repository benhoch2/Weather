"""Offline evaluation of baselines and models on the val/test split.

Writes data/eval/<split>_<timestamp>/summary.json and a per-lead chart.

    python evaluate.py --split test                       # baselines + current model
    python evaluate.py --split val --methods persistence,optflow
    python evaluate.py --model models/nowcast_20261111_1200.pt
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import config
from baselines import METHODS
from common import setup_logging, write_json
from dataset import Timeline, load_timeline
from metrics import Scores

log = logging.getLogger("evaluate")


def select_windows(tl: Timeline, split: str, max_windows: int, stride: int, seed: int = 0) -> np.ndarray:
    """Rainy, gap-free windows of a split, thinned by stride and capped (deterministic)."""
    starts = tl.windows(split)
    starts = starts[tl.rainy(starts)][::max(1, stride)]
    if len(starts) > max_windows:
        rng = np.random.default_rng(seed)
        starts = np.sort(rng.choice(starts, max_windows, replace=False))
    return starts


def evaluate_baseline(tl: Timeline, starts: np.ndarray, name: str) -> Scores:
    fn, scores = METHODS[name], Scores()
    t0 = time.monotonic()
    for k, i in enumerate(starts, 1):
        inputs, targets, valid = tl.window_mm(int(i))
        scores.update(fn(inputs), targets, valid)
        if k % 200 == 0:
            log.info("  %s: %d/%d windows (%.1f/s)", name, k, len(starts), k / (time.monotonic() - t0))
    return scores


def evaluate_model(tl: Timeline, starts: np.ndarray, model, batch: int = 16) -> Scores:
    from model import predict_mm
    scores = Scores()
    for b in range(0, len(starts), batch):
        chunk = starts[b:b + batch]
        pairs = [tl.window_mm(int(i)) for i in chunk]
        preds = predict_mm(model, np.stack([p[0] for p in pairs]))
        for pred, (_, targets, valid) in zip(preds, pairs):
            scores.update(pred, targets, valid)
    return scores


def plot(results: dict[str, dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    thrs = list(next(iter(results.values()))["csi"].keys())
    fig, axes = plt.subplots(1, len(thrs) + 1, figsize=(4.2 * (len(thrs) + 1), 3.6), constrained_layout=True)
    for name, s in results.items():
        leads = s["lead_minutes"]
        for ax, thr in zip(axes, thrs):
            ax.plot(leads, [np.nan if v is None else v for v in s["csi"][thr]], marker="o", ms=3, label=name)
            ax.set_title(f"CSI ≥ {thr} mm/h")
        axes[-1].plot(leads, [np.nan if v is None else v for v in s["mae"]], marker="o", ms=3, label=name)
    axes[-1].set_title("MAE (mm/h)")
    for ax in axes:
        ax.set_xlabel("lead time (min)")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    setup_logging("evaluate")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--methods", default="persistence,optflow,model")
    ap.add_argument("--model", default=None, help="checkpoint path (default: current model)")
    ap.add_argument("--max-windows", type=int, default=1500)
    ap.add_argument("--stride", type=int, default=3, help="use every Nth window (they overlap heavily)")
    a = ap.parse_args()

    tl = load_timeline()
    starts = select_windows(tl, a.split, a.max_windows, a.stride)
    log.info("Evaluating %d rainy %s windows", len(starts), a.split)
    if len(starts) == 0:
        log.error("No complete rainy windows in the %s split yet - backfill more data.", a.split)
        return

    results: dict[str, dict] = {}
    for name in [m.strip() for m in a.methods.split(",") if m.strip()]:
        if name == "model":
            from model import current_model_path, device, load_checkpoint
            path = Path(a.model) if a.model else current_model_path()
            if path is None:
                log.info("No trained model yet - skipping 'model'.")
                continue
            net, _ = load_checkpoint(path, device())
            log.info("Running model %s", path.name)
            results[f"model:{path.stem}"] = evaluate_model(tl, starts, net).summary()
        else:
            log.info("Running %s", name)
            results[name] = evaluate_baseline(tl, starts, name).summary()

    out = config.EVAL_DIR / f"{a.split}_{datetime.now():%Y%m%d_%H%M%S}"
    write_json(out / "summary.json", {"split": a.split, "windows": len(starts), "results": results})
    plot(results, out / "scores.png")
    log.info("Wrote %s", out)
    print_table(results)


def print_table(results: dict[str, dict]) -> None:
    leads = [5, 15, 30, 45, 60]
    for thr in ("0.5", "2", "5"):
        print(f"\nCSI >= {thr} mm/h   " + "  ".join(f"{m:>5}m" for m in leads))
        for name, s in results.items():
            row = [s["csi"][thr][m // 5 - 1] for m in leads]
            print(f"  {name[:22]:<22}" + "  ".join(f"{(v if v is not None else float('nan')):6.3f}" for v in row))
    print("\nmean CSI: " + ", ".join(f"{n}={s['mean_csi']}" for n, s in results.items()))


if __name__ == "__main__":
    main()
