"""Web viewer: live radar + forecast map, accuracy vs baselines, system status.

Binds to 127.0.0.1 by default.  Pass --host 0.0.0.0 to expose it on your
network (it has no login, so only do that on a trusted network).
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time

import numpy as np
from flask import Flask, abort, render_template, request, send_from_directory

import config
from common import local_str, read_json, sanitize, write_json
from geo import km_per_cell, latlon_to_cell
from metrics import Scores
from radar_store import FrameCache, RadarStore

app = Flask(__name__)
store = RadarStore()
frames = FrameCache(store, size=48)
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


@app.route("/api/forecasts")
def forecasts():
    """Forecasts of the last N hours (newest first), for the comparison picker."""
    hours = min(max(int(request.args.get("hours", 24)), 1), 24 * 30)
    cutoff = time.time() - hours * 3600
    out = []
    if config.PRED_DIR.exists():
        for folder in config.PRED_DIR.iterdir():
            if not (folder.is_dir() and folder.name.isdigit() and int(folder.name) >= cutoff):
                continue
            metrics = read_json(folder / "metrics.json")
            model = (metrics or {}).get("scores", {}).get("model", {})
            out.append({"t0": int(folder.name), "local": local_str(int(folder.name)),
                        "verified": bool(metrics and not metrics.get("skipped")),
                        "mean_csi": model.get("mean_csi"),
                        "valid_leads": (metrics or {}).get("valid_leads", 0),
                        "obs_rain_px": (metrics or {}).get("obs_rain_px")})
    out.sort(key=lambda r: r["t0"], reverse=True)
    return _json(out)


_pred_cache: dict[int, np.ndarray] = {}
_pred_lock = threading.Lock()


def _forecast_field(t0: int) -> np.ndarray | None:
    with _pred_lock:
        if t0 in _pred_cache:
            return _pred_cache[t0]
    try:
        with np.load(config.PRED_DIR / str(t0) / "pred.npz") as z:
            arr = z["mm"].astype(np.float32)
    except OSError:
        return None
    with _pred_lock:
        _pred_cache[t0] = arr
        for old in sorted(_pred_cache)[:-4]:
            del _pred_cache[old]
    return arr


def _sample(field: np.ndarray, r: int, c: int) -> tuple[float, float]:
    """(mean over the 3x3 cells around the point, max within ~2 cells)."""
    n = field.shape[0]
    near = field[max(0, r - 1):min(n, r + 2), max(0, c - 1):min(n, c + 2)]
    wide = field[max(0, r - 2):min(n, r + 3), max(0, c - 2):min(n, c + 3)]
    return round(float(near.mean()), 2), round(float(wide.max()), 2)


@app.route("/api/point")
def point():
    """Rain at a coordinate: last hour observed + latest forecast."""
    try:
        lat, lon = float(request.args["lat"]), float(request.args["lon"])
    except (KeyError, ValueError):
        return _json({"error": "Give lat and lon as numbers, e.g. 32.08, 34.78"}), 400
    cell = latlon_to_cell(lat, lon)
    if cell is None:
        return _json({"error": "That point is outside the radar coverage (about 29.4–34.5°N, 31.8–37.9°E)."}), 400
    r, c = cell
    latest = store.latest_time()
    observed, forecast, t0 = [], [], None
    if latest:
        for t in store.list_times(latest - (config.N_IN - 1) * config.STEP, latest):
            field = frames.get(t)
            if field is not None:
                mm, near_max = _sample(field, r, c)
                observed.append({"t": t, "local": local_str(t, "%H:%M"), "mm": mm, "near_max": near_max})
    fc = _latest_forecast()
    if fc and latest and fc["t0"] >= latest - 15 * 60:
        field = _forecast_field(fc["t0"])
        if field is not None:
            t0 = fc["t0"]
            for k, f in enumerate(fc["frames"]):
                if f["t"] > (latest or 0):
                    mm, near_max = _sample(field[k], r, c)
                    forecast.append({"t": f["t"], "local": f["local"], "lead": (k + 1) * config.STEP // 60,
                                     "mm": mm, "near_max": near_max})
    return _json({"lat": lat, "lon": lon, "radius_km": round(1.5 * km_per_cell(), 1), "t0": t0,
                  "now": latest, "observed": observed, "forecast": forecast})


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


LOOPBACK = {"127.0.0.1", "::1"}


def require_cloudflare_access() -> None:
    """Refuse any non-local request that did not come through Cloudflare Access.

    Access adds a signed Cf-Access-Jwt-Assertion header to every request it lets
    through.  This is a safety net, not the gate itself: if the Access policy is
    ever removed or misconfigured, the site answers 403 instead of going public.
    """
    @app.before_request
    def _check_access():
        # A local tunnel (cloudflared on the same machine) connects from loopback, so
        # anything carrying Cloudflare's Cf-Connecting-IP header is treated as remote.
        via_cloudflare = "Cf-Connecting-Ip" in request.headers
        if request.remote_addr in LOOPBACK and not via_cloudflare:
            return None
        if not request.headers.get("Cf-Access-Jwt-Assertion"):
            return "Login required (Cloudflare Access).", 403
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=config.VIEWER_HOST)
    ap.add_argument("--port", type=int, default=config.VIEWER_PORT)
    ap.add_argument("--require-access", action="store_true",
                    help="403 for non-local requests without a Cloudflare Access token (use when published)")
    ap.add_argument("--web-debug", action="store_true", help="show HTTP access log lines")
    a = ap.parse_args()
    if a.require_access:
        require_cloudflare_access()
    if not a.web_debug:
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
    hosts = [h.strip() for h in a.host.split(",") if h.strip()]   # e.g. "127.0.0.1,100.79.167.68"
    shown = ", ".join(f"http://{'localhost' if h == '127.0.0.1' else h}:{a.port}" for h in hosts)
    print(f"Weather radar viewer on {shown}"
          f"{' (Cloudflare Access required for remote requests)' if a.require_access else ''}", flush=True)
    try:
        from waitress import serve
    except ImportError:   # fall back to Flask's development server (first host only)
        app.run(host=hosts[0], port=a.port, debug=False, use_reloader=False, threaded=True)
        return
    serve(app, listen=" ".join(f"{h}:{a.port}" for h in hosts), threads=8, ident="rain-nowcast")


if __name__ == "__main__":
    main()
