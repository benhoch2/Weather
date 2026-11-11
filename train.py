"""Train the U-Net nowcaster on the GPU.

    python train.py                       # train from scratch, promote if better than current
    python train.py --smoke               # 1 short epoch to check the pipeline
    python train.py --init models/current # fine-tune from the promoted model

Uses bf16 autocast on CUDA, AdamW + cosine LR, a rain-weighted loss, and early
stopping on validation mean CSI.  The new checkpoint replaces the live model
only if it scores better on the same validation windows (see promote_if_better).
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import config
from common import local_str, setup_logging, write_json
from dataset import Timeline, load_timeline
from evaluate import evaluate_model, select_windows
from model import UNet, codes_to_log1p, current_model_path, device, load_checkpoint, save_checkpoint

log = logging.getLogger("train")
STATUS_FILE = config.STATUS_DIR / "training.json"

# Loss weights by observed rain rate (mm/h): heavier rain counts more, so the
# model cannot score well by always predicting "no rain".
_W_EDGES = torch.log1p(torch.tensor([0.5, 2.0, 5.0, 10.0, 30.0]))
_W_VALUES = torch.tensor([1.0, 2.0, 5.0, 10.0, 20.0, 30.0])


def weighted_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    w = _W_VALUES.to(target.device)[torch.bucketize(target, _W_EDGES.to(target.device))]
    err = pred - target
    return (w * (err.abs() + err.pow(2))).mean()


class BatchSampler:
    """Draws windows from the in-RAM timeline: ~90% rainy, ~10% dry."""

    def __init__(self, tl: Timeline, starts: np.ndarray, dry_fraction: float = 0.1, seed: int = 0):
        rainy = tl.rainy(starts)
        self.rainy, self.dry = starts[rainy], starts[~rainy]
        self.dry_fraction = dry_fraction if len(self.dry) else 0.0
        self.frames = tl.frames
        self.rng = np.random.default_rng(seed)
        self.offsets = np.arange(config.N_IN + config.N_OUT)

    def __call__(self, batch: int, dev: torch.device):
        n_dry = self.rng.binomial(batch, self.dry_fraction)
        pick = np.concatenate([self.rng.choice(self.rainy, batch - n_dry),
                               self.rng.choice(self.dry, n_dry) if n_dry else np.zeros(0, np.int64)])
        block = torch.from_numpy(self.frames[pick[:, None] + self.offsets[None, :]])
        block = codes_to_log1p(block.pin_memory().to(dev, non_blocking=True))
        return block[:, :config.N_IN], block[:, config.N_IN:]


def write_status(**kw) -> None:
    write_json(STATUS_FILE, {"updated_at": int(time.time()), **kw})


def train(epochs: int = 40, steps: int = 1000, batch: int = 16, lr: float = 1e-3, base: int = 48,
          patience: int = 6, val_windows: int = 600, init: Path | None = None, smoke: bool = False,
          tl: Timeline | None = None) -> dict:
    dev = device()
    if dev.type != "cuda":
        log.warning("CUDA not available - training on CPU will be very slow.")
    tl = tl or load_timeline()
    train_starts = tl.windows("train")
    n_rainy = int(tl.rainy(train_starts).sum()) if len(train_starts) else 0
    val_starts = select_windows(tl, "val", 64 if smoke else val_windows, stride=3)
    log.info("Train windows: %d (%d rainy) | val windows: %d", len(train_starts), n_rainy, len(val_starts))
    if n_rainy < (10 if smoke else 200) or len(val_starts) == 0:
        raise SystemExit("Not enough rainy data to train yet - run backfill.py first.")

    if init:
        model, _ = load_checkpoint(init, dev)
        log.info("Fine-tuning from %s", init)
    else:
        model = UNet(base=base).to(dev)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model: %s, %.1fM parameters, device %s", model.hparams, n_params / 1e6,
             torch.cuda.get_device_name(dev) if dev.type == "cuda" else dev)

    if smoke:
        epochs, steps = 1, 30
    sampler = BatchSampler(tl, train_starts)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total, warmup = epochs * steps, min(300, epochs * steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warmup) if s < warmup else
        0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))

    best_csi, best_state, best_summary, bad_epochs = -1.0, None, None, 0
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for _ in range(steps):
            x, y = sampler(batch, dev)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                pred = model(x)
            loss = weighted_loss(pred.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        model.eval()
        summary = evaluate_model(tl, val_starts, model).summary()
        csi = summary["mean_csi"] or 0.0
        improved = csi > best_csi
        if improved:
            best_csi, best_summary, bad_epochs = csi, summary, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        elapsed = (time.monotonic() - started) / 60
        log.info("epoch %d/%d  loss %.4f  val mean CSI %.4f %s (%.1f min)", epoch, epochs, float(np.mean(losses)),
                 csi, "*" if improved else "", elapsed)
        write_status(state="training", epoch=epoch, epochs=epochs, loss=float(np.mean(losses)),
                     val_mean_csi=csi, best_val_mean_csi=best_csi, minutes=round(elapsed, 1))
        if bad_epochs >= patience:
            log.info("Early stopping (no improvement for %d epochs)", patience)
            break

    model.load_state_dict(best_state)
    model.eval()
    name = f"nowcast_{datetime.now():%Y%m%d_%H%M%S}{'_smoke' if smoke else ''}.pt"
    path = config.MODELS_DIR / name
    save_checkpoint(model, path, {"created": int(time.time()), "val": best_summary,
                                  "init_from": str(init) if init else None,
                                  "train_windows": int(len(train_starts))})
    log.info("Saved %s (val mean CSI %.4f)", path, best_csi)
    decision = {"checkpoint": name, "val_mean_csi": best_csi}
    if not smoke:
        decision.update(promote_if_better(path, model, tl, val_starts))
    write_status(state="done", **decision)
    return decision


def promote_if_better(path: Path, model, tl: Timeline, val_starts: np.ndarray, margin: float = 0.002) -> dict:
    """Score the current live model on the same windows; promote the new one if better."""
    new_csi = evaluate_model(tl, val_starts, model).mean_csi()
    cur_path = current_model_path()
    cur_csi = None
    if cur_path is not None:
        try:
            cur_model, meta = load_checkpoint(cur_path, device())
            if meta.get("grid") == config.GRID:
                cur_csi = evaluate_model(tl, val_starts, cur_model).mean_csi()
        except Exception as e:
            log.warning("Could not evaluate current model %s: %s", cur_path, e)
    promote = cur_csi is None or not np.isfinite(cur_csi) or new_csi > cur_csi + margin
    log.info("Validation mean CSI: new %.4f vs current %s -> %s", new_csi,
             "none" if cur_csi is None else f"{cur_csi:.4f}", "PROMOTE" if promote else "keep current")
    if promote:
        write_json(config.CURRENT_MODEL_FILE, {
            "file": path.name, "promoted_at": int(time.time()), "promoted_local": local_str(int(time.time())),
            "val_mean_csi": new_csi, "previous": cur_path.name if cur_path else None,
            "previous_val_mean_csi": cur_csi})
    return {"promoted": promote, "new_val_mean_csi": new_csi, "current_val_mean_csi": cur_csi}


def main() -> None:
    setup_logging("train")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=1000, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=48, help="U-Net base width")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--init", default=None, help="checkpoint to fine-tune ('current' = live model)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    init = current_model_path() if a.init == "current" else (Path(a.init) if a.init else None)
    torch.backends.cudnn.benchmark = True
    train(a.epochs, a.steps, a.batch, a.lr, a.base, a.patience, init=init, smoke=a.smoke)


if __name__ == "__main__":
    main()
