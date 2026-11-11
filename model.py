"""U-Net nowcaster (RainNet-style) and checkpoint helpers.

Input:  the last N_IN frames as channels, in log1p(mm/h), plus 2 coordinate
        channels so the network can learn location-specific effects
        (coastline, terrain, persistent clutter).
Output: all N_OUT future frames at once, in log1p(mm/h) (softplus, >= 0).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from radar_store import DECODE_LUT


def _block(cin: int, cout: int) -> nn.Sequential:
    groups = 8 if cout % 8 == 0 else 1
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(groups, cout), nn.SiLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(groups, cout), nn.SiLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, n_in: int = config.N_IN, n_out: int = config.N_OUT, base: int = 48, depth: int = 4):
        super().__init__()
        self.hparams = {"n_in": n_in, "n_out": n_out, "base": base, "depth": depth}
        chans = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList()
        cin = n_in + 2
        for c in chans:
            self.enc.append(_block(cin, c))
            cin = c
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for c_skip, c_deep in zip(reversed(chans[:-1]), reversed(chans[1:])):
            self.up.append(nn.ConvTranspose2d(c_deep, c_skip, 2, stride=2))
            self.dec.append(_block(c_skip * 2, c_skip))
        self.head = nn.Conv2d(chans[0], n_out, 1)

    @staticmethod
    def _coords(x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        yy = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([yy, xx], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, self._coords(x)], dim=1)
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))
        return F.softplus(self.head(x))


# ── value transforms ──────────────────────────────────────────────────────────
_LUT_LOG1P = torch.from_numpy(np.log1p(DECODE_LUT))


def codes_to_log1p(q: torch.Tensor) -> torch.Tensor:
    """uint8 cache codes -> log1p(mm/h) float tensor (on q's device)."""
    return _LUT_LOG1P.to(q.device)[q.long()]


def mm_to_log1p(mm: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(mm, 0, None)).astype(np.float32)


def log1p_to_mm(x: torch.Tensor | np.ndarray):
    return torch.expm1(x).clamp_min(0) if isinstance(x, torch.Tensor) else np.clip(np.expm1(x), 0, None)


# ── checkpoints ───────────────────────────────────────────────────────────────

def save_checkpoint(model: UNet, path: Path, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hparams": model.hparams, "grid": config.GRID,
                "meta": meta}, path)


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[UNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = UNet(**ckpt["hparams"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, {"grid": ckpt["grid"], **ckpt.get("meta", {})}


def current_model_path() -> Path | None:
    """Path of the promoted model from models/current.json, or None."""
    try:
        info = json.loads(config.CURRENT_MODEL_FILE.read_text())
        p = config.MODELS_DIR / info["file"]
        return p if p.exists() else None
    except (OSError, ValueError, KeyError):
        return None


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_mm(model: UNet, inputs_mm: np.ndarray, dev: torch.device | None = None) -> np.ndarray:
    """(B, n_in, G, G) or (n_in, G, G) mm/h -> (B, n_out, G, G) / (n_out, G, G) mm/h."""
    dev = dev or next(model.parameters()).device
    single = inputs_mm.ndim == 3
    x = torch.from_numpy(mm_to_log1p(inputs_mm[None] if single else inputs_mm)).to(dev)
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        y = model(x)
    out = log1p_to_mm(y.float()).cpu().numpy()
    return out[0] if single else out
