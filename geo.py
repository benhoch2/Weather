"""Map coordinates <-> radar grid cells.

The IMS PNG is drawn on a Web-Mercator map stretched to RADAR_BOUNDS (that is
how IMS and our viewer overlay it), so rows are linear in Mercator y, not in
latitude.  Using the same projection here makes a clicked point match the
pixel the user sees under the cursor.
"""

from __future__ import annotations

import math

import config

(_SOUTH, _WEST), (_NORTH, _EAST) = config.RADAR_BOUNDS


def _merc_y(lat: float) -> float:
    return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def latlon_to_cell(lat: float, lon: float, size: int = config.GRID) -> tuple[int, int] | None:
    """(row, col) in a size x size grid, or None if outside radar coverage."""
    if not (_SOUTH <= lat <= _NORTH and _WEST <= lon <= _EAST):
        return None
    fy = (_merc_y(_NORTH) - _merc_y(lat)) / (_merc_y(_NORTH) - _merc_y(_SOUTH))
    fx = (lon - _WEST) / (_EAST - _WEST)
    return min(size - 1, int(fy * size)), min(size - 1, int(fx * size))


def km_per_cell(size: int = config.GRID) -> float:
    mid_lat = (_SOUTH + _NORTH) / 2
    return (_EAST - _WEST) * 111.32 * math.cos(math.radians(mid_lat)) / size
