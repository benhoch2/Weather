import config
from geo import latlon_to_cell


def test_corners_and_outside():
    (s, w), (n, e) = config.RADAR_BOUNDS
    assert latlon_to_cell(n, w, 256) == (0, 0)
    assert latlon_to_cell(s, e, 256) == (255, 255)
    assert latlon_to_cell(n + 0.1, w, 256) is None
    assert latlon_to_cell(32.0, 30.0, 256) is None


def test_rows_follow_mercator_not_latitude():
    (s, _), (n, _) = config.RADAR_BOUNDS
    row, _ = latlon_to_cell((s + n) / 2, 35.0, 1000)
    assert row > 500   # the latitude midpoint sits south of the Mercator midpoint


def test_point_api_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONTROL_FILE", tmp_path / "control.json")
    import web_viewer
    c = web_viewer.app.test_client()
    assert c.get("/api/point?lat=abc&lon=1").status_code == 400
    r = c.get("/api/point?lat=40&lon=35")
    assert r.status_code == 400 and "outside" in r.get_json()["error"]
    r = c.get("/api/point?lat=32.08&lon=34.78")
    assert r.status_code == 200 and {"observed", "forecast", "radius_km"} <= set(r.get_json())
    assert c.get("/api/forecasts?hours=24").status_code == 200
