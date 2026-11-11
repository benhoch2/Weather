import importlib

import pytest

import auth

CF = {"Cf-Connecting-Ip": "203.0.113.5"}       # what requests through the tunnel look like


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "AUTH_DIR", tmp_path)
    monkeypatch.setattr(auth, "PASSWORD_FILE", tmp_path / "password.json")
    monkeypatch.setattr(auth, "SECRET_FILE", tmp_path / "secret_key")
    auth._failures.clear()
    import web_viewer
    importlib.reload(web_viewer)
    auth.init_auth(web_viewer.app)
    yield web_viewer.app.test_client()
    importlib.reload(web_viewer)               # don't leak the gate into other tests


def login(c, pw, next_="/"):
    return c.post(f"/login?next={next_}", data={"password": pw}, headers=CF, base_url="https://rain.test")


def test_fails_closed_without_password(client):
    assert client.get("/", headers=CF).status_code == 503
    assert client.get("/api/status").status_code == 200          # local admin access still works


def test_login_flow_and_cookie(client):
    auth.set_password("correct horse")
    r = client.get("/", headers=CF, base_url="https://rain.test")
    assert r.status_code == 302 and "/login" in r.headers["Location"]
    assert client.get("/api/status", headers=CF, base_url="https://rain.test").status_code == 401
    assert b"Wrong password" in login(client, "nope").data
    r = login(client, "correct horse")
    assert r.status_code == 302 and r.headers["Location"].endswith("/")
    cookie = r.headers["Set-Cookie"]
    assert "HttpOnly" in cookie and "Secure" in cookie and "Expires" in cookie
    assert client.get("/api/status", headers=CF, base_url="https://rain.test").status_code == 200


def test_password_change_logs_everyone_out(client):
    auth.set_password("first password")
    login(client, "first password")
    assert client.get("/api/status", headers=CF, base_url="https://rain.test").status_code == 200
    auth.set_password("second password")
    assert client.get("/api/status", headers=CF, base_url="https://rain.test").status_code == 401


def test_rate_limit(client):
    auth.set_password("correct horse")
    for _ in range(auth.MAX_FAILURES):
        login(client, "wrong")
    assert login(client, "correct horse").status_code == 429


def test_no_open_redirect(client):
    auth.set_password("correct horse")
    r = login(client, "correct horse", next_="//evil.example")
    assert r.headers["Location"].endswith("/") and "evil" not in r.headers["Location"]


def test_short_password_rejected():
    with pytest.raises(ValueError):
        auth.set_password("short")
