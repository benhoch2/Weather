"""Reference forecasts the neural model must beat.

* persistence: the latest observed frame, unchanged, for every lead time.
* optical flow: estimate motion from the last few frames (pysteps Lucas-Kanade)
  and advect the latest frame along it (semi-Lagrangian extrapolation).
  Falls back to OpenCV DIS optical flow if pysteps is not installed.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

import config

log = logging.getLogger(__name__)


def persistence(inputs_mm: np.ndarray, n_out: int = config.N_OUT) -> np.ndarray:
    """inputs (n_in, H, W) mm/h -> (n_out, H, W)."""
    return np.repeat(inputs_mm[-1:], n_out, axis=0).astype(np.float32)


try:
    from pysteps import motion as _ps_motion
    from pysteps import nowcasts as _ps_nowcasts
    _HAVE_PYSTEPS = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_PYSTEPS = False


def _db(mm: np.ndarray) -> np.ndarray:
    """Rain rate -> dB scale with a floor, the usual input for motion estimation."""
    return np.where(mm >= 0.1, 10 * np.log10(np.maximum(mm, 0.1)), -15.0).astype(np.float64)


def optical_flow(inputs_mm: np.ndarray, n_out: int = config.N_OUT) -> np.ndarray:
    """Advect the last frame along motion estimated from the last 4 frames."""
    last = inputs_mm[-1].astype(np.float64)
    if not (inputs_mm[-4:] >= 0.1).any():
        return np.zeros((n_out,) + last.shape, dtype=np.float32)
    try:
        if _HAVE_PYSTEPS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                field = np.stack([_db(f) for f in inputs_mm[-4:]])
                velocity = _ps_motion.get_method("LK")(field, verbose=False)
                extrap = _ps_nowcasts.get_method("extrapolation")
                out = extrap(last, velocity, n_out)
        else:
            out = _opencv_extrapolate(inputs_mm, n_out)
    except Exception as e:
        log.debug("optical flow failed (%s); using persistence", e)
        return persistence(inputs_mm, n_out)
    return np.nan_to_num(np.asarray(out, dtype=np.float32), nan=0.0).clip(min=0)


def _opencv_extrapolate(inputs_mm: np.ndarray, n_out: int) -> np.ndarray:
    import cv2

    def to8(mm):
        return np.clip((_db(mm) + 15) / 45 * 255, 0, 255).astype(np.uint8)

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(to8(inputs_mm[-2]), to8(inputs_mm[-1]), None)  # per-step motion (px)
    h, w = inputs_mm.shape[1:]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    out = []
    for k in range(1, n_out + 1):
        mapx, mapy = xx - k * flow[..., 0], yy - k * flow[..., 1]
        out.append(cv2.remap(inputs_mm[-1].astype(np.float32), mapx, mapy, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0))
    return np.stack(out)


METHODS = {"persistence": persistence, "optflow": optical_flow}
