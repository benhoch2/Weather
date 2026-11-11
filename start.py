"""Supervisor: runs the fetcher, predictor and web viewer, plus a nightly retrain.

* Streams each child's output with a coloured prefix.
* Restarts crashed children with exponential backoff.
* Predictor exit code 3 (no trained model) is not restarted; the supervisor
  starts it again once models/current.json appears.
* Honours data/control.json (the viewer's "Pause forecasts" button).
* Runs retrain.py once a night at RETRAIN_HOUR (local time).
* Refuses to start twice (PID lock verified with psutil, works on Windows/Linux).

    python start.py [--port 5050] [--host 127.0.0.1] [--no-retrain]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

import psutil

import config
from common import LOCAL_TZ, read_json, write_json

APP_LOCK = config.PROJECT_DIR / "app.lock"
STATUS_FILE = config.STATUS_DIR / "supervisor.json"
RETRAIN_HOUR = 3
USE_COLOR = sys.stdout.isatty()
COLORS = {"FETCHER": "\033[36m", "PREDICTOR": "\033[33m", "WEB": "\033[32m", "RETRAIN": "\033[34m",
          "SUPERVISOR": "\033[35m"}


def label(tag: str) -> str:
    return f"{COLORS.get(tag, '')}[{tag}]\033[0m" if USE_COLOR else f"[{tag}]"


def say(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {label('SUPERVISOR')} {msg}", flush=True)


# ── single-instance lock ─────────────────────────────────────────────────────

def acquire_lock() -> None:
    try:
        pid = int(APP_LOCK.read_text().strip())
        proc = psutil.Process(pid)
        if any("start.py" in part for part in proc.cmdline()):
            say(f"Already running (PID {pid}). Close it first.")
            sys.exit(1)
    except (OSError, ValueError, psutil.Error):
        pass  # no lock, unreadable, or stale
    APP_LOCK.write_text(str(os.getpid()))


def release_lock() -> None:
    try:
        if APP_LOCK.read_text().strip() == str(os.getpid()):
            APP_LOCK.unlink()
    except OSError:
        pass


def kill_tree(proc: subprocess.Popen, timeout: float = 5) -> None:
    try:
        parent = psutil.Process(proc.pid)
        procs = parent.children(recursive=True) + [parent]
    except psutil.Error:
        return
    for p in procs:
        try:
            p.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass


# ── children ─────────────────────────────────────────────────────────────────

class Child:
    def __init__(self, tag: str, args: list[str], restart: bool = True):
        self.tag, self.args, self.restart = tag, args, restart
        self.proc: subprocess.Popen | None = None
        self.started = 0.0
        self.failures = 0
        self.restarts = 0
        self.next_start = 0.0
        self.blocked: str | None = None   # reason it must not be (re)started

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, env: dict) -> None:
        self.proc = subprocess.Popen([sys.executable, "-u", *self.args], cwd=config.PROJECT_DIR, env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        self.started = time.monotonic()
        threading.Thread(target=self._pump, daemon=True).start()
        say(f"Started {self.tag} (PID {self.proc.pid})")

    def _pump(self) -> None:
        tag = label(self.tag)
        for raw in iter(self.proc.stdout.readline, b""):
            line = raw.rstrip(b"\r\n").decode("utf-8", errors="replace")
            if line:
                print(f"{tag} {line}", flush=True)

    def stop(self) -> None:
        if self.running:
            kill_tree(self.proc)
        self.proc = None

    def status(self) -> dict:
        return {"running": self.running, "pid": self.proc.pid if self.running else None,
                "restarts": self.restarts, "blocked": self.blocked}


def main() -> None:
    ap = argparse.ArgumentParser(description="Weather radar nowcast supervisor")
    ap.add_argument("--host", default=config.VIEWER_HOST)
    ap.add_argument("--port", type=int, default=config.VIEWER_PORT)
    ap.add_argument("--no-retrain", action="store_true", help="disable the nightly retrain")
    ap.add_argument("--web-debug", action="store_true")
    a = ap.parse_args()

    acquire_lock()
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    web_args = ["web_viewer.py", "--host", a.host, "--port", str(a.port)] + (["--web-debug"] if a.web_debug else [])
    children = {
        "FETCHER": Child("FETCHER", ["fetcher.py"]),
        "PREDICTOR": Child("PREDICTOR", ["predictor.py"]),
        "WEB": Child("WEB", web_args),
    }
    retrain = Child("RETRAIN", ["retrain.py"], restart=False)
    last_retrain_day = None

    say(f"Open http://{'localhost' if a.host == '127.0.0.1' else a.host}:{a.port}  (Ctrl+C to stop)")
    try:
        while True:
            control = read_json(config.CONTROL_FILE, {}) or {}
            pred = children["PREDICTOR"]
            pred_enabled = control.get("predictor_enabled", True)
            if not pred_enabled and pred.running:
                say("Forecasts paused from the viewer - stopping PREDICTOR")
                pred.stop()
            pred.blocked = "paused" if not pred_enabled else (
                pred.blocked if pred.blocked == "no_model" and not config.CURRENT_MODEL_FILE.exists() else None)

            now = time.monotonic()
            for child in children.values():
                if child.proc is not None and not child.running:
                    code = child.proc.returncode
                    child.proc = None
                    if child.tag == "PREDICTOR" and code == config.EXIT_NO_MODEL:
                        child.blocked = "no_model"
                        say("PREDICTOR has no trained model - waiting for models/current.json (run train.py)")
                        continue
                    ran = now - child.started
                    child.failures = 0 if ran > 600 else child.failures + 1
                    delay = min(300, 10 * 2 ** max(0, child.failures - 1))
                    child.next_start = now + delay
                    say(f"{child.tag} exited with code {code} - restarting in {delay}s")
                if not child.running and not child.blocked and now >= child.next_start:
                    if child.started:
                        child.restarts += 1
                    child.start(env)

            local_now = datetime.now(LOCAL_TZ)
            if (not a.no_retrain and local_now.hour == RETRAIN_HOUR and last_retrain_day != local_now.date()
                    and not retrain.running):
                last_retrain_day = local_now.date()
                say("Starting nightly retrain check")
                retrain.start(env)

            write_json(STATUS_FILE, {"updated_at": int(time.time()), "pid": os.getpid(),
                                     "children": {t: c.status() for t, c in children.items()},
                                     "retrain_running": retrain.running})
            time.sleep(3)
    except KeyboardInterrupt:
        say("Stopping...")
    finally:
        for child in [*children.values(), retrain]:
            child.stop()
        write_json(STATUS_FILE, {"updated_at": int(time.time()), "stopped": True})
        release_lock()
        say("All stopped.")


if __name__ == "__main__":
    main()
