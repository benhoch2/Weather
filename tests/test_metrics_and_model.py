import json

import numpy as np
import torch

from common import sanitize, write_json
from metrics import Scores
from model import UNet, codes_to_log1p
from radar_store import decode_mm, encode_mm


def test_perfect_forecast_scores():
    obs = np.zeros((12, 8, 8), dtype=np.float32)
    obs[:, 2:4, 2:4] = 3.0
    s = Scores()
    s.update(obs, obs)
    sm = s.summary()
    assert sm["csi"]["0.5"] == [1.0] * 12 and sm["csi"]["2"] == [1.0] * 12
    assert sm["csi"]["5"] == [None] * 12          # no events at 5 mm/h -> undefined
    assert sm["mae"] == [0.0] * 12


def test_contingency_counts():
    obs = np.zeros((12, 4, 4), dtype=np.float32)
    pred = np.zeros_like(obs)
    obs[0, 0, :2] = 1.0        # 2 observed rain px
    pred[0, 0, 1:3] = 1.0      # 1 hit, 1 false alarm, 1 miss
    s = Scores()
    s.update(pred, obs)
    assert s.hits[0, 0] == 1 and s.misses[0, 0] == 1 and s.false_alarms[0, 0] == 1
    assert abs(s.csi()[0, 0] - 1 / 3) < 1e-9
    r = Scores.from_counts(json.loads(json.dumps(s.to_counts())))
    assert np.array_equal(r.hits, s.hits) and r.forecasts == 1


def test_valid_mask_skips_leads():
    obs = np.ones((12, 4, 4), dtype=np.float32)
    s = Scores()
    valid = np.array([True] * 6 + [False] * 6)
    s.update(obs, obs, valid)
    assert s.pixels[6:].sum() == 0 and s.pixels[:6].sum() == 6 * 16


def test_sanitize_and_json(tmp_path):
    assert sanitize({"a": float("inf"), "b": [float("nan"), 1.5]}) == {"a": None, "b": [None, 1.5]}
    write_json(tmp_path / "x.json", {"psnr": float("inf")})
    assert json.loads((tmp_path / "x.json").read_text()) == {"psnr": None}


def test_unet_shapes_and_lut():
    net = UNet(n_in=12, n_out=12, base=8, depth=3)
    y = net(torch.zeros(2, 12, 64, 64))
    assert y.shape == (2, 12, 64, 64) and (y >= 0).all()
    q = torch.from_numpy(encode_mm(np.array([0.0, 1.0, 10.0], dtype=np.float32)))
    assert torch.allclose(codes_to_log1p(q), torch.log1p(torch.from_numpy(decode_mm(q.numpy()))))
