import importlib


def test_require_access_blocks_remote_without_token(tmp_path, monkeypatch):
    import web_viewer
    importlib.reload(web_viewer)            # fresh app without earlier before_request hooks
    web_viewer.require_cloudflare_access()
    c = web_viewer.app.test_client()
    assert c.get("/api/status").status_code == 200                                    # loopback
    remote = {"REMOTE_ADDR": "100.100.146.14"}
    assert c.get("/api/status", environ_base=remote).status_code == 403
    ok = c.get("/api/status", environ_base=remote, headers={"Cf-Access-Jwt-Assertion": "x.y.z"})
    assert ok.status_code == 200
    importlib.reload(web_viewer)            # don't leak the hook into other tests
