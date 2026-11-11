"""Single shared password for the web viewer, remembered in a signed cookie.

* The password is stored only as a salted scrypt hash in data/auth/password.json.
  Set or change it on the server itself (prompts, nothing echoed or logged):

      python3 auth.py set-password

* After a correct login the browser keeps a signed session cookie for ~400 days.
  Changing the password invalidates every existing cookie.
* Requests from the machine itself (not via the Cloudflare tunnel) need no login.
* Failed logins are rate limited per client address.
* If no password has been set, remote visitors get a "not configured" page
  (fails closed).
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import secrets
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import timedelta
from pathlib import Path

import config

AUTH_DIR = config.DATA_DIR / "auth"
PASSWORD_FILE = AUTH_DIR / "password.json"
SECRET_FILE = AUTH_DIR / "secret_key"
LOOPBACK = {"127.0.0.1", "::1"}
MAX_FAILURES, FAILURE_WINDOW = 5, 15 * 60
OPEN_PATHS = {"/login", "/favicon.ico"}

_failures: dict[str, deque] = defaultdict(deque)
_lock = threading.Lock()


def _write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)


def _secret_key() -> bytes:
    if not SECRET_FILE.exists():
        _write_private(SECRET_FILE, secrets.token_hex(32))
    return bytes.fromhex(SECRET_FILE.read_text().strip())


def _password_record() -> dict | None:
    try:
        return json.loads(PASSWORD_FILE.read_text())
    except (OSError, ValueError):
        return None


def _version(record: dict) -> str:
    """Changes whenever the password changes, so old cookies stop working."""
    return hashlib.sha256(record["hash"].encode()).hexdigest()[:16]


def set_password(password: str) -> None:
    from werkzeug.security import generate_password_hash
    if len(password) < 8:
        raise ValueError("Use at least 8 characters.")
    _write_private(PASSWORD_FILE, json.dumps({"hash": generate_password_hash(password), "set_at": int(time.time())}))


def client_ip(request) -> str:
    return request.headers.get("Cf-Connecting-Ip") or request.remote_addr or "?"


def is_local(request) -> bool:
    """Direct request from this machine - not something forwarded by the tunnel."""
    return request.remote_addr in LOOPBACK and "Cf-Connecting-Ip" not in request.headers


def _rate_limited(ip: str) -> bool:
    now = time.time()
    with _lock:
        q = _failures[ip]
        while q and now - q[0] > FAILURE_WINDOW:
            q.popleft()
        return len(q) >= MAX_FAILURES


def _record_failure(ip: str) -> None:
    with _lock:
        _failures[ip].append(time.time())


def _safe_next(target: str | None) -> str:
    return target if target and target.startswith("/") and not target.startswith("//") else "/"


def init_auth(app) -> None:
    """Require the shared password for every non-local request."""
    from flask import jsonify, redirect, render_template, request, session, url_for
    from werkzeug.security import check_password_hash

    app.secret_key = _secret_key()
    app.config.update(SESSION_COOKIE_NAME="rain_session", SESSION_COOKIE_HTTPONLY=True,
                      SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_SECURE=True,
                      PERMANENT_SESSION_LIFETIME=timedelta(days=400))

    @app.before_request
    def _gate():
        if is_local(request) or request.path in OPEN_PATHS:
            return None
        record = _password_record()
        if record is None:
            return "This site isn't configured yet (no password set).", 503
        if session.get("pw") == _version(record):
            return None
        if request.path.startswith(("/api/", "/obs/", "/pred/")):
            return jsonify({"error": "login required"}), 401
        return redirect(url_for("login", next=request.full_path.rstrip("?")))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        record = _password_record()
        if record is None:
            return "This site isn't configured yet (no password set).", 503
        error, ip = None, client_ip(request)
        if request.method == "POST":
            if _rate_limited(ip):
                error = "Too many attempts. Try again in a few minutes."
            elif check_password_hash(record["hash"], request.form.get("password", "")):
                session.clear()
                session.permanent = True
                session["pw"] = _version(record)
                return redirect(_safe_next(request.args.get("next")))
            else:
                _record_failure(ip)
                error = "Wrong password."
        return render_template("login.html", error=error), (429 if error and "Too many" in error else 200)

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] != "set-password":
        sys.exit("usage: python3 auth.py set-password")
    first = getpass.getpass("New password: ")
    if first != getpass.getpass("Repeat it: "):
        sys.exit("Passwords don't match - nothing changed.")
    try:
        set_password(first)
    except ValueError as e:
        sys.exit(str(e))
    print(f"Password set ({PASSWORD_FILE}). Everyone currently logged in will need the new password.")


if __name__ == "__main__":
    main()
