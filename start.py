"""
Single-script launcher for the Weather Radar Prediction System.

Starts all three components (fetcher, predictor, web viewer) in one process,
streams their output with labelled prefixes, and shuts everything down cleanly
on Ctrl+C.

If any component crashes, the launcher automatically restarts it after a short
delay — the system self-heals without manual intervention.

Also prevents running the app twice: if start.py is already running, a second
invocation will refuse to start and tell you to close the first window first.

Usage:
    python start.py [--port 5050]
"""

import os
# Must be set before TensorFlow loads (MKL C++ reads this at DLL-load time).
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import sys
import subprocess
import threading
import time
import argparse
from datetime import datetime
from pathlib import Path
from process_utils import is_pid_running

# ── Lock file -----------------------------------------------------------------
# Prevents double-starting the whole app (e.g. clicking start_viewer.bat twice)
PROJECT_DIR = Path(__file__).parent
APP_LOCK = PROJECT_DIR / "app.lock"
CHILD_LOCKS = [PROJECT_DIR / "predictor.lock", PROJECT_DIR / "fetcher.lock"]

# How long to wait before restarting a crashed component
RESTART_DELAY = 10


def _clean_stale_locks() -> None:
    """Remove child lock files, killing the owner if it is still alive.

    Any process holding a child lock when start.py runs is an orphan from a
    previous launcher session.  Kill it so the new child can acquire the lock.
    """
    for lock in CHILD_LOCKS:
        if not lock.exists():
            continue
        try:
            pid = int(lock.read_text().strip())
            if is_pid_running(pid):
                print(f"[{_ts()}] {_label('LAUNCHER')} Killing orphan PID {pid} holding {lock.name}...")
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(pid), "/T", "/F"],
                        capture_output=True, timeout=10,
                    )
                except Exception:
                    pass
            lock.unlink(missing_ok=True)
            print(f"[{_ts()}] {_label('LAUNCHER')} Cleared lock: {lock.name}")
        except Exception:
            lock.unlink(missing_ok=True)


def acquire_app_lock() -> None:
    if APP_LOCK.exists():
        try:
            pid = int(APP_LOCK.read_text().strip())
            if is_pid_running(pid):
                print(f"[LAUNCHER] ERROR: App is already running (PID {pid}).")
                print(f"[LAUNCHER]   Close that window first, or delete {APP_LOCK} if it is stale.")
                sys.exit(1)
        except (ValueError, SystemExit):
            raise
        except Exception:
            pass  # Unreadable / stale — overwrite
    APP_LOCK.write_text(str(os.getpid()))


def release_app_lock() -> None:
    try:
        APP_LOCK.unlink(missing_ok=True)
    except Exception:
        pass


# ── Output streaming ----------------------------------------------------------

USE_COLOR = sys.stdout.isatty()

_COLORS = {
    "FETCHER":   "\033[36m",   # cyan
    "PREDICTOR": "\033[33m",   # yellow
    "WEB":       "\033[32m",   # green
    "LAUNCHER":  "\033[35m",   # magenta
}
_RESET = "\033[0m"


def _label(tag: str) -> str:
    color = _COLORS.get(tag, "") if USE_COLOR else ""
    reset = _RESET if USE_COLOR else ""
    return f"{color}[{tag}]{reset}"


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def stream_output(proc: subprocess.Popen, tag: str) -> None:
    """Read lines from *proc* stdout and print with a labelled prefix (runs in a daemon thread)."""
    label = _label(tag)
    try:
        while True:
            line = proc.stdout.readline()  # bytes
            if not line:
                break
            line = line.rstrip(b"\r\n").decode(errors="replace")
            if line:
                print(f"{label} {line}", flush=True)
    except Exception:
        pass


# ── Main ----------------------------------------------------------------------

# Each entry: (tag, command-args builder)
# The builder is a callable so we can reference args.port lazily.
_web_debug = False  # Set to True in main() when --web-debug is passed

_COMMANDS = {
    "FETCHER":   lambda: [sys.executable, "-u", "fetch_radar_continuous.py"],
    "PREDICTOR": lambda: [sys.executable, "-u", "run_predictions_persistent.py"],
    "WEB":       lambda: [sys.executable, "-u", "web_viewer.py"] + (["--web-debug"] if _web_debug else []),
}


def _start_process(tag: str, env: dict, project_dir: Path) -> subprocess.Popen:
    """Spawn a child process for *tag* and wire up its output stream."""
    proc = subprocess.Popen(
        _COMMANDS[tag](),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=0, cwd=project_dir, env=env,
    )
    threading.Thread(
        target=stream_output, args=(proc, tag), daemon=True,
    ).start()
    return proc


def main() -> None:
    parser = argparse.ArgumentParser(description="Weather Radar Prediction System launcher")
    parser.add_argument("--port", type=int, default=int(os.environ.get("WEATHER_RADAR_VIEWER_PORT", 5050)),
                        help="Port for the web viewer (default: 5050)")
    parser.add_argument("--web-debug", action="store_true",
                        help="Show HTTP access log lines from the web server")
    args = parser.parse_args()

    global _web_debug
    _web_debug = args.web_debug

    acquire_app_lock()

    # Clean any stale child locks *before* starting children
    _clean_stale_locks()

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "TF_ENABLE_ONEDNN_OPTS": "0",
        "WEATHER_RADAR_VIEWER_PORT": str(args.port),
        "PYTHONIOENCODING": "utf-8",
    }

    project_dir = PROJECT_DIR
    processes: dict[str, subprocess.Popen] = {}

    print("=" * 65)
    print(" Weather Radar Prediction System")
    print(f" Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    try:
        # ── Initial launch ────────────────────────────────────────────────────
        for tag in ("FETCHER", "PREDICTOR", "WEB"):
            label_text = {
                "FETCHER": "radar fetcher",
                "PREDICTOR": "prediction engine",
                "WEB": f"web viewer on http://localhost:{args.port}",
            }[tag]
            print(f"[{_ts()}] {_label('LAUNCHER')} Starting {label_text}...")
            processes[tag] = _start_process(tag, env, project_dir)

        print(f"[{_ts()}] {_label('LAUNCHER')} Open http://localhost:{args.port} in your browser")
        print(f"[{_ts()}] {_label('LAUNCHER')} Press Ctrl+C to stop everything")
        print("=" * 65)

        # ── Monitor + auto-restart loop ───────────────────────────────────────
        while True:
            time.sleep(5)
            for tag in list(processes):
                proc = processes[tag]
                code = proc.poll()
                if code is None:
                    continue  # Still running — nothing to do

                print(
                    f"[{_ts()}] {_label('LAUNCHER')} WARNING: {tag} exited with code {code}",
                    flush=True,
                )

                # Clean stale child locks so the restart can acquire them
                _clean_stale_locks()

                print(
                    f"[{_ts()}] {_label('LAUNCHER')} Restarting {tag} in {RESTART_DELAY}s...",
                    flush=True,
                )
                time.sleep(RESTART_DELAY)

                # Double-check that we still need to restart (user might Ctrl+C
                # during the delay, which raises KeyboardInterrupt)
                processes[tag] = _start_process(tag, env, project_dir)
                print(
                    f"[{_ts()}] {_label('LAUNCHER')} {tag} restarted (PID {processes[tag].pid})",
                    flush=True,
                )

    except KeyboardInterrupt:
        print(f"\n[{_ts()}] {_label('LAUNCHER')} Ctrl+C received — stopping all processes...")
        for tag, proc in processes.items():
            if proc.poll() is None:
                print(f"[{_ts()}] {_label('LAUNCHER')}   Stopping {tag}...")
                proc.terminate()

        time.sleep(3)

        for tag, proc in processes.items():
            if proc.poll() is None:
                print(f"[{_ts()}] {_label('LAUNCHER')}   Force-killing {tag}...")
                proc.kill()

        print(f"[{_ts()}] {_label('LAUNCHER')} All processes stopped. Goodbye!")

    finally:
        # Clean child locks so the next start is clean
        _clean_stale_locks()
        release_app_lock()


if __name__ == "__main__":
    main()
