import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from ims_source import colorize  # noqa: E402
from radar_store import RadarStore  # noqa: E402

DAY0 = 1774137600  # 2026-03-22 00:00 UTC


def png_bytes(mm: np.ndarray) -> bytes:
    buf = io.BytesIO()
    colorize(mm).save(buf, format="PNG")
    return buf.getvalue()


def blob(value: float = 5.0, shift: int = 0) -> np.ndarray:
    mm = np.zeros((config.NATIVE_SIZE, config.NATIVE_SIZE), dtype=np.float32)
    mm[300:400, 300 + shift:420 + shift] = value
    return mm


@pytest.fixture
def store(tmp_path) -> RadarStore:
    return RadarStore(raw_dir=tmp_path / "raw", cache_dir=tmp_path / "cache",
                      missing_file=tmp_path / "missing.txt", grid=32)
