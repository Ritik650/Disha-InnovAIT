"""
disha.optimizer.distances — offline travel-time estimates between cells.

We don't have retailer lat/lon, but we already cache district centroids for
weather.  Approximate cell location = district centroid (1 retailer per
tehsil on average; centroid is a defensible proxy for the demo).

Travel model:
  - intra-tehsil (same district + same tehsil): 5 min flat
  - inter-tehsil within district:               12 min flat
  - cross-district:                             haversine km × road_factor
                                                / avg_speed_kmh
"""
from __future__ import annotations

import math
from functools import lru_cache

from disha.twin.weather import get_district_centroid

AVG_SPEED_KMH = 30.0     # rural Indian road speed
ROAD_FACTOR = 1.4        # haversine → road-distance multiplier
INTRA_TEHSIL_MIN = 5.0
INTRA_DISTRICT_MIN = 12.0


@lru_cache(maxsize=4096)
def _centroid(district: str, state: str) -> tuple[float, float]:
    return get_district_centroid(district, state)


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = a; lat2, lon2 = b
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    h = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    return 2 * R * math.asin(math.sqrt(h))


def travel_min(
    from_tehsil: str, from_district: str, from_state: str,
    to_tehsil:   str, to_district:   str, to_state:   str,
) -> float:
    if from_tehsil == to_tehsil:
        return INTRA_TEHSIL_MIN
    if from_district == to_district and from_state == to_state:
        return INTRA_DISTRICT_MIN
    a = _centroid(from_district, from_state)
    b = _centroid(to_district, to_state)
    km = _haversine_km(a, b) * ROAD_FACTOR
    return km / AVG_SPEED_KMH * 60.0
