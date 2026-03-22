"""Shared process utilities for lock-file management and PID verification."""

import os
import sys
import subprocess
from pathlib import Path


def is_pid_running(pid: int) -> bool:
    """Return True when a **Python** process with this PID is alive.

    Plain PID checks cause false positives on Windows because PIDs get
    recycled quickly.  We also verify the image name contains 'python'.
    """
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if str(pid) in line and "python" in line.lower():
                return True
        return False
    except Exception:
        return False


def acquire_lock(lock_path: Path, label: str) -> None:
    """Acquire a lock file, cleaning stale locks automatically.

    Args:
        lock_path: Path to the lock file.
        label: Prefix for log messages (e.g. "FETCHER", "PERSISTENT").
    """
    if lock_path.exists():
        try:
            pid = int(lock_path.read_text().strip())
            if is_pid_running(pid):
                print(f"[{label}] ERROR: Already running (PID {pid}).")
                print(f"[{label}]   Stop it first, or delete {lock_path} if it is stale.")
                sys.exit(1)
            else:
                print(f"[{label}] Cleaning stale lock (PID {pid} is gone)")
                lock_path.unlink(missing_ok=True)
        except (ValueError, SystemExit):
            raise
        except Exception:
            lock_path.unlink(missing_ok=True)
    lock_path.write_text(str(os.getpid()))
    print(f"[{label}] Lock acquired (PID {os.getpid()})")


def release_lock(lock_path: Path) -> None:
    """Release a lock file, ignoring errors."""
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass
