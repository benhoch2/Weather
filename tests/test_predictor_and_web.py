import config
from conftest import DAY0, blob, png_bytes
from predictor import FrameCache, gather_inputs, input_times, target_times


def test_target_times_anchor_on_t0():
    t0 = DAY0 + 6 * 3600
    assert target_times(t0) == [t0 + k * 300 for k in range(1, 13)]
    assert input_times(t0)[-1] == t0 and input_times(t0)[0] == t0 - 55 * 60


def test_gather_inputs_fills_one_gap_only(store):
    t0 = DAY0 + 6 * 3600
    times = input_times(t0)
    for t in times:
        if t != times[5]:
            store.save(t, png_bytes(blob()))
    arr, missing = gather_inputs(FrameCache(store), t0)
    assert arr is not None and arr.shape == (12, 32, 32) and missing == [times[5]]

    store.path_for(times[7]).unlink()
    arr, missing = gather_inputs(FrameCache(store), t0)
    assert arr is None and len(missing) == 2


def test_gather_inputs_requires_latest_frame(store):
    t0 = DAY0 + 6 * 3600
    for t in input_times(t0)[:-1]:
        store.save(t, png_bytes(blob()))
    arr, _ = gather_inputs(FrameCache(store), t0)
    assert arr is None


def test_web_routes_are_safe(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONTROL_FILE", tmp_path / "control.json")
    import web_viewer
    client = web_viewer.app.test_client()
    assert client.get("/images/..%5C..%5Cstart.py").status_code == 404
    assert client.get("/obs/..%5Cstart.png").status_code == 404
    assert client.get("/obs/123.png").status_code == 404
    assert client.get("/pred/1/99.png").status_code == 404
    assert client.post("/api/control", data={"predictor_enabled": "0"}).status_code == 415
    r = client.post("/api/control", json={"predictor_enabled": False})
    assert r.status_code == 200 and r.get_json() == {"predictor_enabled": False}
    assert client.get("/api/status").status_code == 200
    assert client.get("/api/accuracy?days=7").status_code == 200
