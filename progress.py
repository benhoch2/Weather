"""Show where things stand: data download, cache, training, live system.

    python progress.py            # print once
    python progress.py --watch    # refresh every 30 s (Ctrl+C to stop)
"""

from __future__ import annotations

import argparse
import os
import re
import time

import config
from common import read_json

LOGS = [config.DATA_DIR / "backfill_early.log", config.DATA_DIR / "backfill.log"]
PROGRESS_RE = re.compile(r"^(\d\d:\d\d:\d\d) INFO \[(\w[\w ]*)\] (\d+)/(\d+)\s+saved=(\d+) absent=(\d+) errors=(\d+)\s+ETA (\d+) min")
STAGE_RE = re.compile(r"Stage (A|B|all frames): .*?(\d+) (?:hourly frames|slot\(s\), (\d+) to download)")
LABELS = {"A": "Scan (1 frame/hour)", "B": "Rainy-period frames", "all frames": "All frames"}


def bar(done: int, total: int, width: int = 32) -> str:
    frac = 0 if total <= 0 else min(1, done / total)
    fill = int(round(frac * width))
    return f"[{'#' * fill}{'.' * (width - fill)}] {frac:6.1%}  {done:,}/{total:,}"


def backfill_section() -> list[str]:
    out = []
    for log in LOGS:
        if not log.exists():
            continue
        raw = log.read_bytes()   # PowerShell's *> redirect writes UTF-16
        enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
        lines = raw.decode(enc, errors="replace").splitlines()
        stages, last = {}, {}
        for line in lines:
            if m := STAGE_RE.search(line):
                stages[m.group(1)] = int(m.group(3) or m.group(2))
            if m := PROGRESS_RE.match(line):
                last[m.group(2)] = m
        if not stages:
            continue
        age = (time.time() - log.stat().st_mtime) / 60
        finished = lines and ("Interrupted" in lines[-1] or any(
            int(m.group(3)) == int(m.group(4)) for k, m in last.items() if k == list(stages)[-1]))
        state = "finished" if finished else ("running" if age < 5 else f"idle for {age:.0f} min (stopped?)")
        out.append(f"{log.name}: {state}")
        for stage, total in stages.items():
            m = last.get(stage)
            done = int(m.group(3)) if m else 0
            eta = f"  ETA {int(m.group(8))} min" if m and done < total else ""
            errs = f"  errors {m.group(7)}" if m and int(m.group(7)) else ""
            out.append(f"  {LABELS.get(stage, stage):<22}{bar(done, total)}{eta}{errs}")
    running = any(line.endswith(": running") for line in out)
    if running:   # an idle log while another download runs is just queued behind it
        out = [re.sub(r": idle for \d+ min \(stopped\?\)$", ": paused - resumes after the running download", line)
               for line in out]
    if not out:
        out.append("No backfill logs yet.")
    elif not running:
        out.append("  (no download running)")
    return out


def data_section() -> list[str]:
    raw = config.RAW_DIR
    n_files = sum(1 for _ in raw.rglob("*.png")) if raw.exists() else 0
    size = sum(p.stat().st_size for p in raw.rglob("*.png")) / 2**20 if raw.exists() else 0
    days = sum(1 for p in raw.glob("*/*/*") if p.is_dir()) if raw.exists() else 0
    cache = config.CACHE_DIR / str(config.GRID)
    n_cache = len(list(cache.glob("*.npz"))) if cache.exists() else 0
    missing = len(config.MISSING_FILE.read_text().split()) if config.MISSING_FILE.exists() else 0
    return [f"Radar frames on disk: {n_files:,} ({size:,.0f} MB) across {days} days; "
            f"{missing:,} confirmed missing on IMS",
            f"Training cache:       {n_cache} of {days} days built"]


def training_section() -> list[str]:
    out = []
    t = read_json(config.STATUS_DIR / "training.json")
    if t:
        age = (time.time() - t.get("updated_at", 0)) / 60
        if t.get("state") == "training":
            out.append(f"Training: epoch {bar(t['epoch'], t['epochs'], 20)}  "
                       f"val CSI {t.get('val_mean_csi', 0):.3f} (best {t.get('best_val_mean_csi', 0):.3f})  "
                       f"{t.get('minutes', 0):.0f} min so far" + ("" if age < 30 else f"  [no update for {age:.0f} min]"))
        else:
            out.append(f"Last training: {t.get('checkpoint')}  val CSI {t.get('val_mean_csi', 0):.3f}"
                       + (f"  promoted={t.get('promoted')}" if "promoted" in t else ""))
    else:
        out.append("Training: not started")
    cur = read_json(config.CURRENT_MODEL_FILE)
    out.append(f"Live model: {cur['file']} (val CSI {cur.get('val_mean_csi', 0):.3f})" if cur else "Live model: none yet")
    evals = sorted(config.EVAL_DIR.glob("*/summary.json"), key=lambda p: p.stat().st_mtime) if config.EVAL_DIR.exists() else []
    if evals:
        e = read_json(evals[-1]) or {}
        scores = ", ".join(f"{k} {v.get('mean_csi')}" for k, v in (e.get("results") or {}).items())
        out.append(f"Latest evaluation ({e.get('split')}, {e.get('windows')} windows): mean CSI {scores}")
    return out


def live_section() -> list[str]:
    sup = read_json(config.STATUS_DIR / "supervisor.json")
    if not sup or sup.get("stopped") or time.time() - sup.get("updated_at", 0) > 60:
        return ["Live system: not running (start_viewer.bat)"]
    kids = ", ".join(f"{k.lower()} {'up' if v['running'] else v.get('blocked') or 'down'}"
                     for k, v in sup["children"].items())
    return [f"Live system: running ({kids}) - http://localhost:{config.VIEWER_PORT}"]


def render() -> str:
    parts = [f"Weather nowcast progress - {time.strftime('%H:%M:%S')}", "",
             "DOWNLOAD", *backfill_section(), "", "DATA", *data_section(), "",
             "MODEL", *training_section(), "", *live_section()]
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=30)
    a = ap.parse_args()
    if not a.watch:
        print(render())
        return
    try:
        while True:
            text = render()
            os.system("cls" if os.name == "nt" else "clear")
            print(text + f"\n\n(refreshing every {a.interval}s - Ctrl+C to stop)", flush=True)
            time.sleep(a.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
