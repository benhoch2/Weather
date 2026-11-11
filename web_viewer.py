"""Web viewer: live radar + forecast map, accuracy vs baselines, system status.

Binds to 127.0.0.1 by default.  Pass --host 0.0.0.0 to expose it on your
network (it has no login, so only do that on a trusted network).
"""

from __future__ import annotations

import argparse
import json
import logging
import time

from flask import Flask, abort, render_template, request, send_from_directory

import config
from common import local_str, read_json, sanitize, write_json
from metrics import Scores
from radar_store import RadarStore

app = Flask(__name__)
store = RadarStore()
LEDGER_FILE = config.PRED_DIR / "ledger.jsonl"
METHODS = ("model", "persistence", "optflow")


def _json(data):
    return app.response_class(json.dumps(sanitize(data), allow_nan=False), mimetype="application/json")


# ── static frames (integer-only routes + send_from_directory: no traversal) ──

@app.route("/obs/<int:t>.png")
def observed_png(t: int):
    path = store.path_for(t)
    if not path.exists():
        abort(404)
    return send_from_directory(path.parent, path.name, mimetype="image/png", max_age=86400)


@app.route("/pred/<int:t0>/<int:k>.png")
def forecast_png(t0: int, k: int):
    if not 1 <= k <= config.N_OUT:
        abort(404)
    return send_from_directory(config.PRED_DIR / str(t0), f"lead_{k:02d}.png", mimetype="image/png", max_age=86400)


# ── pages / API ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", bounds=config.RADAR_BOUNDS, legend=config.LEGEND,
                           thresholds=config.THRESHOLDS, step_min=config.STEP // 60, n_out=config.N_OUT)


def _control() -> dict:
    return read_json(config.CONTROL_FILE, {}) or {"predictor_enabled": True}


@app.route("/api/status")
def status():
    return _json({
        "now": int(time.time()),
        "fetcher": read_json(config.STATUS_DIR / "fetcher.json"),
        "predictor": read_json(config.STATUS_DIR / "predictor.json"),
        "supervisor": read_json(config.STATUS_DIR / "supervisor.json"),
        "training": read_json(config.STATUS_DIR / "training.json"),
        "retrain": read_json(config.STATUS_DIR / "retrain.json"),
        "model": read_json(config.CURRENT_MODEL_FILE),
        "control": _control(),
    })


@app.route("/api/control", methods=["POST"])
def control():
    # JSON-only: a cross-site form post cannot send application/json without a preflight.
    if not request.is_json:
        abort(415)
    body = request.get_json(silent=True) or {}
    ctl = _control()
    if "predictor_enabled" in body:
        ctl["predictor_enabled"] = bool(body["predictor_enabled"])
    write_json(config.CONTROL_FILE, ctl)
    return _json(ctl)


def _latest_forecast() -> dict | None:
    if not config.PRED_DIR.exists():
        return None
    folders = sorted((p for p in config.PRED_DIR.iterdir() if p.is_dir() and p.name.isdigit()),
                     key=lambda p: int(p.name), reverse=True)
    for folder in folders[:5]:
        meta = read_json(folder / "meta.json")
        if meta:
            return _forecast_payload(int(folder.name), meta)
    return None


def _forecast_payload(t0: int, meta: dict) -> dict:
    return {"t0": t0, "t0_local": local_str(t0), "model": meta.get("model"), "max_mm": meta.get("max_mm"),
            "frames": [{"t": t, "local": local_str(t, "%H:%M"), "url": f"/pred/{t0}/{k}.png"}
                       for k, t in enumerate(meta["valid_times"], 1)]}


@app.route("/api/live")
def live():
    latest = store.latest_time()
    observed = []
    if latest:
        for t in store.list_times(latest - (config.N_IN - 1) * config.STEP, latest):
            observed.append({"t": t, "local": local_str(t, "%H:%M"), "url": f"/obs/{t}.png"})
    return _json({"observed": observed, "forecast": _latest_forecast()})


@app.route("/api/prediction/<int:t0>")
def prediction(t0: int):
    folder = config.PRED_DIR / str(t0)
    meta = read_json(folder / "meta.json")
    if not meta:
        abort(404)
    payload = _forecast_payload(t0, meta)
    payload["observed"] = [{"t": t, "local": local_str(t, "%H:%M"),
                            "url": f"/obs/{t}.png" if store.has(t) else None} for t in meta["valid_times"]]
    payload["metrics"] = read_json(folder / "metrics.json")
    return _json(payload)


_ledger_cache: dict = {}


def _ledger_entries() -> list[dict]:
    """Parsed ledger, cached until the file changes."""
    try:
        st = LEDGER_FILE.stat()
    except OSError:
        return []
    key = (st.st_size, st.st_mtime)
    if _ledger_cache.get("key") != key:
        entries = []
        with open(LEDGER_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except ValueError:
                    continue
        _ledger_cache.update(key=key, entries=entries)
    return _ledger_cache["entries"]


@app.route("/api/accuracy")
def accuracy():
    days = min(max(int(request.args.get("days", 7)), 1), 3650)
    cutoff = time.time() - days * 86400
    entries = [e for e in _ledger_entries() if e["t0"] >= cutoff]
    totals = {m: None for m in METHODS}
    for e in entries:
        for m in METHODS:
            if m in e["counts"]:
                s = Scores.from_counts(e["counts"][m])
                totals[m] = s if totals[m] is None else totals[m].merge(s)
    recent = []
    for e in sorted(entries, key=lambda e: e["t0"], reverse=True)[:60]:
        if e.get("obs_rain_px", 0) == 0:
            continue
        row = {"t0": e["t0"], "local": local_str(e["t0"]), "model": e.get("model")}
        for m in METHODS:
            if m in e["counts"]:
                row[m] = Scores.from_counts(e["counts"][m]).mean_csi()
        recent.append(row)
    return _json({"days": days, "forecasts": len(entries),
                  "summary": {m: s.summary() for m, s in totals.items() if s is not None},
                  "recent": recent[:30], "offline": _latest_offline_eval()})


def _latest_offline_eval() -> dict | None:
    runs = sorted(config.EVAL_DIR.glob("*/summary.json"), key=lambda p: p.stat().st_mtime) if config.EVAL_DIR.exists() else []
    if not runs:
        return None
    data = read_json(runs[-1]) or {}
    return {"run": runs[-1].parent.name, "split": data.get("split"), "windows": data.get("windows"),
            "mean_csi": {k: v.get("mean_csi") for k, v in (data.get("results") or {}).items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=config.VIEWER_HOST)
    ap.add_argument("--port", type=int, default=config.VIEWER_PORT)
    ap.add_argument("--web-debug", action="store_true", help="show HTTP access log lines")
    a = ap.parse_args()
    if not a.web_debug:
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
    print(f"Weather radar viewer on http://{'localhost' if a.host == '127.0.0.1' else a.host}:{a.port}", flush=True)
    app.run(host=a.host, port=a.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
