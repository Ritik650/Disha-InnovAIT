"""
disha.twin.weather — Open-Meteo historical weather fetch + disk cache + synthetic fallback.

Outputs per-district daily weather DataFrames and weekly aggregates used in L0 twin.
Disease-pressure proxy: days in a week where RH_max > threshold (configurable).
"""
from __future__ import annotations

import json
import logging
import warnings
from datetime import date
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_SETTINGS_PATH = _ROOT / "config" / "settings.yaml"
log = logging.getLogger(__name__)


def _settings() -> dict:
    with open(_SETTINGS_PATH) as f:
        return yaml.safe_load(f)


# ── Per-district fetch + cache ─────────────────────────────────────────────────

def fetch_district_weather(
    district: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    cache_dir: Path,
    timeout_s: float = 8.0,
) -> pd.DataFrame:
    """
    Returns a daily weather DataFrame for a district.
    Hit order: disk cache → Open-Meteo API → synthetic fallback.
    """
    cache_key = district.lower().replace(" ", "_").replace("/", "_")
    cache_file = cache_dir / f"{cache_key}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            raw = json.load(f)
        return _parse_open_meteo(raw)

    cfg = _settings()
    base_url = cfg["weather"]["open_meteo_base"]
    variables = ",".join(cfg["weather"]["variables"])
    url = (
        f"{base_url}?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily={variables}&timezone=Asia%2FKolkata"
    )

    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.get(url)
            resp.raise_for_status()
            raw = resp.json()
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(raw, f)
        log.info("Weather fetched from API for %s", district)
        return _parse_open_meteo(raw)
    except Exception as exc:
        warnings.warn(
            f"Weather API unavailable for {district} ({exc}); using synthetic fallback.",
            stacklevel=2,
        )
        return _synthetic_weather(district, start_date, end_date)


def _parse_open_meteo(raw: dict) -> pd.DataFrame:
    daily = raw.get("daily", {})
    n = len(daily.get("time", []))
    df = pd.DataFrame({
        "date": pd.to_datetime(daily.get("time", [])),
        "temp_max": daily.get("temperature_2m_max", [np.nan] * n),
        "temp_min": daily.get("temperature_2m_min", [np.nan] * n),
        "precip_mm": daily.get("precipitation_sum", [0.0] * n),
        "rh_max": daily.get("relative_humidity_2m_max", [np.nan] * n),
        "rh_min": daily.get("relative_humidity_2m_min", [np.nan] * n),
    })
    return df


def _synthetic_weather(district: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Seeded synthetic daily weather for India Rabi season (Oct–Apr).
    Deterministic: seed derived from district name so the same district
    always produces the same synthetic series.
    """
    seed = sum(ord(c) for c in district) % (2**31)
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(dates)
    doy = dates.day_of_year.values

    # Temperature: peaks ~30°C in Oct/Apr, dips ~10°C in Jan
    temp_mean = 20.0 - 10.0 * np.cos(2 * np.pi * (doy - 15) / 365)
    temp_max = (temp_mean + rng.normal(5, 2, n)).round(1)
    temp_min = (temp_mean - rng.normal(8, 2, n)).round(1)

    # Rainfall: sparse Rabi season (~8% rainy days, avg 5 mm when it rains)
    rain_mask = rng.random(n) < 0.08
    precip = np.where(rain_mask, rng.exponential(5, n), 0.0).round(1)

    # Humidity: naturally higher in cool months; boosted on rainy days
    rh_base = 50.0 - 15.0 * np.cos(2 * np.pi * (doy - 15) / 365) + rng.normal(0, 5, n)
    rh_max = np.clip(rh_base + 20 * rain_mask + rng.normal(5, 3, n), 20, 100).round(1)
    rh_min = np.clip(rh_base - 10 + rng.normal(0, 3, n), 10, 90).round(1)

    return pd.DataFrame({
        "date": dates,
        "temp_max": temp_max,
        "temp_min": temp_min,
        "precip_mm": precip,
        "rh_max": rh_max,
        "rh_min": rh_min,
    })


# ── Weekly aggregation ─────────────────────────────────────────────────────────

def aggregate_to_weekly(
    daily_df: pd.DataFrame,
    rh_threshold: float = 75.0,
    min_pressure_days: int = 3,
) -> pd.DataFrame:
    """
    Aggregate daily weather to ISO week (Monday–Sunday).
    Adds disease_pressure_days and disease_pressure_flag columns.
    week_start is the Monday of each ISO week.
    """
    df = daily_df.copy()
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
    df["high_rh"] = df["rh_max"] > rh_threshold

    weekly = (
        df.groupby("week_start")
        .agg(
            rainfall_mm_7d=("precip_mm", "sum"),
            rh_max_7d=("rh_max", "max"),
            rh_mean_7d=("rh_max", "mean"),
            temp_max_7d=("temp_max", "max"),
            temp_min_7d=("temp_min", "min"),
            disease_pressure_days=("high_rh", "sum"),
        )
        .reset_index()
    )
    weekly["disease_pressure_flag"] = (
        weekly["disease_pressure_days"] >= min_pressure_days
    ).astype(int)
    return weekly.round(2)


# ── Batch fetch for all districts in data ─────────────────────────────────────

def fetch_all_districts(
    districts: list[tuple[str, float, float]],  # (district_name, lat, lon)
    start_date: str,
    end_date: str,
    cache_dir: Path,
    max_api_calls: int = 40,
) -> dict[str, pd.DataFrame]:
    """
    Fetch weekly weather for a list of (district, lat, lon) triples.
    Limits API calls to max_api_calls; excess districts use synthetic fallback.
    Returns {district: weekly_df}.
    """
    cfg = _settings()
    rh_thresh = cfg["weather"].get("disease_pressure_rh_threshold", 75.0)
    min_days = cfg["weather"].get("disease_pressure_min_days", 3)

    result: dict[str, pd.DataFrame] = {}
    api_calls = 0

    for district, lat, lon in districts:
        cache_key = district.lower().replace(" ", "_").replace("/", "_")
        cache_file = cache_dir / f"{cache_key}.json"
        use_api = cache_file.exists() or api_calls < max_api_calls
        if not cache_file.exists():
            api_calls += 1

        daily = fetch_district_weather(
            district=district,
            lat=lat,
            lon=lon,
            start_date=start_date,
            end_date=end_date,
            cache_dir=cache_dir,
        )
        result[district] = aggregate_to_weekly(daily, rh_thresh, min_days)

    return result


# ── District centroid catalogue ────────────────────────────────────────────────

# Fallback centroids for districts not in settings.yaml.
# State-level centroids (approximate geographic centres).
_STATE_CENTROIDS: dict[str, tuple[float, float]] = {
    "Bihar":          (25.09, 85.31),
    "Haryana":        (29.05, 76.08),
    "Maharashtra":    (19.75, 75.71),
    "Rajasthan":      (27.02, 74.21),
    "Uttar Pradesh":  (26.85, 80.91),
    "Madhya Pradesh": (23.47, 77.97),
    "Punjab":         (31.14, 75.34),
    "Gujarat":        (22.26, 71.19),
    "Karnataka":      (15.31, 75.71),
    "West Bengal":    (22.98, 87.85),
}


def get_district_centroid(district: str, state: str = "") -> tuple[float, float]:
    """
    Look up (lat, lon) for a district. Falls back to state centroid.
    Covers all districts present in the Disha dataset.
    """
    cfg = _settings()
    centroids: dict[str, list[float]] = cfg.get("district_centroids", {})

    # Normalize district name: replace underscores, spaces, handle variants
    key = district.replace("_", " ").strip()
    if key in centroids:
        return tuple(centroids[key])

    # Try raw key
    if district in centroids:
        return tuple(centroids[district])

    # State fallback
    lat, lon = _STATE_CENTROIDS.get(state, (20.59, 78.96))  # India centre default
    return lat, lon
