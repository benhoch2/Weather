from datetime import datetime, timedelta, timezone

import numpy as np

import config
from common import LOCAL_TZ
from ims_source import decode_png, local_to_utc, parse_ims_time, utc_to_stamp
from conftest import png_bytes


def _transitions(year: int):
    """(spring, autumn) local transition datetimes for the year, found by scanning."""
    out = []
    t = datetime(year, 1, 1, tzinfo=timezone.utc)
    prev = t.astimezone(LOCAL_TZ).utcoffset()
    while t.year == year:
        off = t.astimezone(LOCAL_TZ).utcoffset()
        if off != prev:
            out.append((t, off > prev))
            prev = off
        t += timedelta(minutes=30)
    return out


def test_parse_ims_time_is_israel_local():
    # 06:05 Israel standard time (UTC+2) = 04:05 UTC
    assert parse_ims_time("2026-11-11 06:05:00") == int(datetime(2026, 11, 11, 4, 5, tzinfo=timezone.utc).timestamp())


def test_stamp_roundtrip_normal_time():
    t = int(datetime(2026, 1, 15, 6, 0, tzinfo=timezone.utc).timestamp())
    stamp = utc_to_stamp(t)
    assert stamp == "202601150800"
    assert local_to_utc(datetime.strptime(stamp, "%Y%m%d%H%M")) == t


def test_dst_transitions_are_skipped():
    trans = _transitions(2026)
    assert len(trans) == 2
    for utc_when, spring in trans:
        local_before = (utc_when - timedelta(minutes=30)).astimezone(LOCAL_TZ).replace(tzinfo=None)
        if spring:   # local clock jumps forward: the skipped half hour does not exist
            missing = local_before + timedelta(minutes=30)
            assert local_to_utc(missing) is None
        else:        # local clock falls back: the repeated hour is ambiguous
            repeated = utc_when.astimezone(LOCAL_TZ).replace(tzinfo=None)
            assert local_to_utc(repeated) is None
            assert utc_to_stamp(int(utc_when.timestamp())) is None


def test_decode_matches_legend_values():
    mm = np.zeros((config.NATIVE_SIZE, config.NATIVE_SIZE), dtype=np.float32)
    for i, (_, value) in enumerate(config.LEGEND):
        mm[10 + i * 20: 20 + i * 20, 10:30] = value
    out = decode_png(png_bytes(mm))
    for i, (_, value) in enumerate(config.LEGEND):
        assert np.allclose(out[15 + i * 20, 20], value, rtol=0.02)
    assert out[900, 900] == 0


def test_decode_continuous_values_roundtrip():
    rng = np.random.default_rng(0)
    mm = np.zeros((config.NATIVE_SIZE, config.NATIVE_SIZE), dtype=np.float32)
    vals = 10 ** rng.uniform(-1, 2.3, size=200)
    mm[0, :200] = vals
    out = decode_png(png_bytes(mm))[0, :200]
    assert np.max(np.abs(np.log10(out) - np.log10(vals))) < 0.05   # within ~12 %
