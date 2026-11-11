import numpy as np

import config
import dataset
from conftest import DAY0, blob, png_bytes
from radar_store import decode_mm, encode_mm, utc_day


def test_encode_decode_roundtrip():
    mm = np.array([0, 0.005, 0.01, 0.1, 0.5, 2, 10, 50, 200, 500], dtype=np.float32)
    out = decode_mm(encode_mm(mm))
    assert out[0] == 0 and out[1] == 0
    nz = mm[2:9]
    assert np.max(np.abs(out[2:9] - nz) / nz) < 0.025
    assert out[-1] == np.float32(200)


def test_missing_slots(store):
    t0 = DAY0 + 12 * 3600
    for t in store.window_times(t0, 12, 0):
        if t != t0 - 3 * config.STEP:
            store.save(t, png_bytes(blob()))
    assert store.missing_slots(t0, 12, 0) == [t0 - 3 * config.STEP]


def test_windows_skip_gaps_and_split_boundaries(store, monkeypatch):
    # 40 consecutive frames with one gap at index 30
    for k in range(40):
        if k != 30:
            store.save(DAY0 + k * config.STEP, png_bytes(blob(shift=k)))
    monkeypatch.setattr(dataset, "split_of", lambda d: "train")
    tl = dataset.Timeline(store, [utc_day(DAY0)])
    starts = tl.windows("train", n_in=4, n_out=2)          # windows of 6 frames
    assert len(starts) > 0
    for s in starts:
        assert not (s <= 30 <= s + 3), "gap inside the input frames"
        assert np.all(np.diff(tl.times[s:s + 6]) == config.STEP)
    # gaps in the target frames are allowed and masked
    assert 25 in starts and 26 in starts                    # gap at target index 1 / 0
    _, _, valid = tl.window_mm(25, n_in=4, n_out=2)
    assert valid.tolist() == [True, False]
    assert len(tl.windows("val", n_in=4, n_out=2)) == 0
    inputs, targets, valid = tl.window_mm(int(starts[0]), n_in=4, n_out=2)
    assert inputs.shape == (4, 32, 32) and targets.shape == (2, 32, 32) and valid.all()
    assert inputs.max() > 1


def test_day_cache_rebuilds_when_new_files_arrive(store):
    store.save(DAY0, png_bytes(blob()))
    _, present = store.load_day(utc_day(DAY0))
    assert present.sum() == 1
    store.save(DAY0 + config.STEP, png_bytes(blob()))
    _, present = store.load_day(utc_day(DAY0))
    assert present.sum() == 2
