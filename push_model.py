"""Copy a trained Pi-compatible model to benh-pi and make it the live model there.

    python push_model.py                    # the promoted model (models/current.json)
    python push_model.py models/x.pt        # a specific checkpoint

The model must use norm='batch' (OpenCV DNN can run it); its .onnx export is
created if missing.  The file is copied over SSH (Tailscale), then the Pi's
models/current.json is replaced atomically - its predictor reloads within 30 s.
Target host/path come from .claude/deploy.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import config
from common import local_str, read_json, setup_logging

log = setup_logging("push_model")


def deploy_target() -> tuple[str, str]:
    cfg = read_json(config.PROJECT_DIR / ".claude" / "deploy.json") or {}
    return f"{cfg.get('user', 'benh')}@{cfg.get('host', 'benh-pi')}", cfg.get("path", "~/Weather")


def run(cmd: list[str]) -> None:
    log.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", nargs="?", help="a .pt checkpoint (default: the promoted model)")
    a = ap.parse_args()

    from model import current_model_path, export_onnx, load_checkpoint
    ckpt = Path(a.checkpoint) if a.checkpoint else current_model_path()
    if ckpt is None or not ckpt.exists():
        sys.exit("No checkpoint to push (train a model first).")
    model, meta = load_checkpoint(ckpt, "cpu")
    if model.hparams.get("norm") != "batch":
        sys.exit(f"{ckpt.name} uses norm={model.hparams.get('norm')!r}; the Pi needs a --norm batch model.")
    onnx = ckpt.with_suffix(".onnx")
    if not onnx.exists():
        export_onnx(model, onnx, meta["grid"])

    target, path = deploy_target()
    val = (meta.get("val") or {}).get("mean_csi")
    current = {"file": onnx.name, "grid": meta["grid"], "source_checkpoint": ckpt.name,
               "promoted_at": int(time.time()), "promoted_local": local_str(int(time.time())),
               "val_mean_csi": val}
    with tempfile.TemporaryDirectory() as tmp:
        cur = Path(tmp) / "current.json.new"
        cur.write_text(json.dumps(current, indent=2))
        run(["ssh", "-o", "BatchMode=yes", target, f"mkdir -p {path}/models"])
        run(["scp", "-q", "-o", "BatchMode=yes", str(onnx), str(cur), f"{target}:{path}/models/"])
        run(["ssh", "-o", "BatchMode=yes", target,
             f"mv {path}/models/current.json.new {path}/models/current.json"])
    log.info("Pi now serves %s (val mean CSI %s)", onnx.name, val)


if __name__ == "__main__":
    main()
