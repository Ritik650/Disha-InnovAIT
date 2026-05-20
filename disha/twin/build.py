"""
disha.twin.build — builds data/processed/twin.parquet and monthly_panel.parquet.

Outputs
-------
twin.parquet          weekly (tehsil × week_start) feature store
monthly_panel.parquet (tehsil × month_start × product) causal panel with twin covariates

Run: python -m disha.twin.build
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from disha.common.data import (
    SKU_TO_PRODUCT,
    build_tehsil_dominant_crops,
    build_tehsil_grower_features,
    build_treatment_panel_monthly,
    load_agronomy,
    load_digital_funnel,
    load_growers,
    load_inventory,
    load_pos,
    load_retailers,
    load_reps,
    load_settings,
    load_visit_log,
    load_whatsapp,
)
from disha.twin.weather import fetch_all_districts, get_district_centroid

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _settings() -> dict:
    with open(_ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def _season_weeks(start: str = "2025-10-06", end: str = "2026-03-29") -> pd.DatetimeIndex:
    """All ISO week-start Mondays within the season."""
    return pd.date_range(start=start, end=end, freq="W-MON")


# ── POS weekly revenue per (tehsil, product) ──────────────────────────────────

def _weekly_pos(pos: pd.DataFrame, retailers: pd.DataFrame) -> pd.DataFrame:
    """Returns (tehsil, week_start, product, weekly_rev, lag_rev_4w)."""
    pos = pos.copy()
    pos["week_start"] = pos["transaction_date"] - pd.to_timedelta(
        pos["transaction_date"].dt.weekday, unit="D"
    )
    pos = pos.merge(retailers[["retailer_id", "tehsil"]], on="retailer_id", how="left")
    pos = pos.dropna(subset=["tehsil"])
    pos["product"] = pos["sku_name"].map(SKU_TO_PRODUCT)
    pos = pos.dropna(subset=["product"])

    weekly = (
        pos.groupby(["tehsil", "product", "week_start"])["revenue"]
        .sum()
        .reset_index()
        .rename(columns={"revenue": "weekly_pos_rev"})
    )
    # Lag 4 weeks
    lag = weekly.copy()
    lag["week_start"] = lag["week_start"] + pd.Timedelta(weeks=4)
    lag = lag.rename(columns={"weekly_pos_rev": "lag_pos_rev_4w"})
    weekly = weekly.merge(lag, on=["tehsil", "product", "week_start"], how="left")
    weekly["lag_pos_rev_4w"] = weekly["lag_pos_rev_4w"].fillna(0.0)

    # Total (all products) per (tehsil, week)
    total = weekly.groupby(["tehsil", "week_start"])["weekly_pos_rev"].sum().reset_index()
    total = total.rename(columns={"weekly_pos_rev": "weekly_pos_rev_total"})

    return weekly, total


def _weekly_oos(inv: pd.DataFrame, retailers: pd.DataFrame) -> pd.DataFrame:
    """Returns per-(tehsil, week_start) OOS rate."""
    inv = inv.copy()
    inv["week_start"] = inv["week_end_date"] - pd.Timedelta(days=6)
    inv = inv.merge(retailers[["retailer_id", "tehsil"]], on="retailer_id", how="left")
    inv = inv.dropna(subset=["tehsil"])
    oos = (
        inv.groupby(["tehsil", "week_start"])["is_oos"]
        .mean()
        .reset_index()
        .rename(columns={"is_oos": "oos_rate"})
    )
    return oos


def _weekly_visit_pressure(vis: pd.DataFrame) -> pd.DataFrame:
    """# of treatment visits per (tehsil, week) + rolling 2-week pressure."""
    treat = vis[vis["visit_type"].isin(["retailer meeting", "campaign_conducted"])].copy()
    treat["week_start"] = treat["visit_date"] - pd.to_timedelta(
        treat["visit_date"].dt.weekday, unit="D"
    )
    weekly = (
        treat.groupby(["visit_tehsil", "week_start"])
        .size()
        .reset_index(name="visit_count_week")
        .rename(columns={"visit_tehsil": "tehsil"})
    )
    # Rolling 2-week pressure: join with previous week
    prev = weekly.copy()
    prev["week_start"] = prev["week_start"] + pd.Timedelta(weeks=1)
    prev = prev.rename(columns={"visit_count_week": "visit_count_prev_week"})
    weekly = weekly.merge(prev, on=["tehsil", "week_start"], how="left")
    weekly["visit_count_prev_week"] = weekly["visit_count_prev_week"].fillna(0)
    weekly["visit_pressure_2w"] = weekly["visit_count_week"] + weekly["visit_count_prev_week"]
    return weekly[["tehsil", "week_start", "visit_count_week", "visit_pressure_2w"]]


# ── Vectorised window features ─────────────────────────────────────────────────

_SAFE_PRODUCT_NAME = str.maketrans(" .", "__", "")


def _safe_col(product: str) -> str:
    return product.translate(_SAFE_PRODUCT_NAME)


def compute_window_features_vectorised(
    base: pd.DataFrame,          # must have tehsil, week_start, dominant_crop
    agronomy: dict,
    products: list[str],
) -> pd.DataFrame:
    """
    Vectorised computation of per-product window features for all rows.

    Adds columns per product (up to 36 columns for 12 products):
        win_open_{p}, win_days_left_{p}, win_decay_{p}

    Strategy:
      1. Build a lookup table of (crop, product, stage) → (window_open, window_close, stage_approx, close_after).
      2. Merge with base on dominant_crop.
      3. Compute open/days_left/decay in one vectorised pass.
      4. Aggregate across multiple stages per (tehsil, week, product) — keep max decay.
      5. Pivot to wide format and merge back.
    """
    fallback = agronomy["fallback_calendars"]
    defaults = agronomy["defaults"]

    # Build compact bounds table
    bounds_rows = []
    for product, pcfg in agronomy.get("products", {}).items():
        for se in pcfg.get("stages", []):
            crop = se["crop"]
            stage_name = se["stage"]
            crop_cal = fallback.get(crop, {})
            approx_str = next(
                (s["approx"] for s in crop_cal.get("stages", []) if s["stage"] == stage_name),
                None,
            )
            if approx_str is None:
                continue
            stage_approx = pd.Timestamp(approx_str)
            ob = se.get("window_open_days_before", defaults["window_open_days_before"])
            ca = se.get("window_close_days_after", defaults["window_close_days_after"])
            bounds_rows.append({
                "dominant_crop": crop,
                "product": product,
                "stage_approx": stage_approx,
                "window_open": stage_approx - pd.Timedelta(days=ob),
                "window_close": stage_approx + pd.Timedelta(days=ca),
                "close_after": ca,
            })
    bounds = pd.DataFrame(bounds_rows)

    base_ts = base.copy()
    base_ts["week_ts"] = pd.to_datetime(base_ts["week_start"])

    # Cross-join on dominant_crop; rows with no bounds entry get NaN
    merged = base_ts.merge(bounds, on="dominant_crop", how="left")

    # Vectorised window math
    merged["win_open_raw"] = (
        (merged["week_ts"] >= merged["window_open"]) &
        (merged["week_ts"] <= merged["window_close"])
    )
    after_approx = merged["week_ts"] > merged["stage_approx"]
    elapsed_days = (merged["week_ts"] - merged["stage_approx"]).dt.days.clip(lower=0)
    merged["win_decay_raw"] = np.where(
        merged["window_open"].isna(), 0.0,
        np.where(
            ~after_approx, 1.0,
            np.clip(1.0 - elapsed_days / merged["close_after"].fillna(7), 0.0, 1.0),
        ),
    )
    merged["win_decay_raw"] = np.where(merged["win_open_raw"].fillna(False), merged["win_decay_raw"], 0.0)
    merged["win_days_left_raw"] = (merged["window_close"] - merged["week_ts"]).dt.days.fillna(-999)
    merged["win_open_raw"] = merged["win_open_raw"].fillna(False)

    # Per (tehsil, week_start, product): keep the stage with max decay
    best = (
        merged.groupby(["tehsil", "week_start", "product"], dropna=False)
        .agg(
            win_open=("win_open_raw", "max"),
            win_days_left=("win_days_left_raw", "min"),   # most urgent remaining
            win_decay=("win_decay_raw", "max"),
        )
        .reset_index()
    )

    # Pivot to wide
    open_wide = best.pivot_table(
        index=["tehsil", "week_start"], columns="product", values="win_open", aggfunc="first"
    )
    open_wide.columns = [f"win_open_{_safe_col(c)}" for c in open_wide.columns]

    decay_wide = best.pivot_table(
        index=["tehsil", "week_start"], columns="product", values="win_decay", aggfunc="first"
    )
    decay_wide.columns = [f"win_decay_{_safe_col(c)}" for c in decay_wide.columns]

    days_wide = best.pivot_table(
        index=["tehsil", "week_start"], columns="product", values="win_days_left", aggfunc="first"
    )
    days_wide.columns = [f"win_days_left_{_safe_col(c)}" for c in days_wide.columns]

    wide = open_wide.join(decay_wide).join(days_wide).reset_index()
    for c in wide.columns:
        if c.startswith("win_open_"):
            wide[c] = wide[c].fillna(False).infer_objects(copy=False)
        elif c.startswith("win_decay_"):
            wide[c] = wide[c].fillna(0.0)
        elif c.startswith("win_days_left_"):
            wide[c] = wide[c].fillna(-999)

    result = base_ts.drop(columns=["week_ts"]).merge(wide, on=["tehsil", "week_start"], how="left")
    return result


# ── Digital funnel feature per (week_start, crop) ─────────────────────────────

def _weekly_digital_funnel(dfun: pd.DataFrame) -> pd.DataFrame:
    """Returns per-(week_start, campaign_crop) impression + lead counts."""
    dfun = dfun.copy()
    dfun["week_start"] = dfun["week_start_date"] - pd.to_timedelta(
        dfun["week_start_date"].dt.weekday, unit="D"
    )
    # Use social_post_impression if available, else impressions
    imp_col = "social_post_impression" if "social_post_impression" in dfun.columns else "impressions"
    result = (
        dfun.groupby(["campaign_crop", "week_start"])
        .agg(
            funnel_impressions=(imp_col, "sum"),
            funnel_leads=("lead_form_submission", "sum"),
        )
        .reset_index()
        .rename(columns={"campaign_crop": "dominant_crop"})
    )
    return result


# ── Main builder ───────────────────────────────────────────────────────────────

def build_twin(seed: int = 42) -> Path:
    """
    Builds twin.parquet at (tehsil × week_start) grain and returns its path.
    """
    np.random.seed(seed)
    cfg = _settings()
    processed = _ROOT / cfg["paths"]["processed"]
    cache_weather = _ROOT / cfg["paths"]["cache_weather"]
    processed.mkdir(parents=True, exist_ok=True)

    log.info("=== Building Disha Territory Digital Twin ===")

    # --- Load raw data ---
    log.info("Loading raw data...")
    pos = load_pos()
    vis = load_visit_log()
    retailers = load_retailers()
    inv = load_inventory()
    growers = load_growers()
    dfun = load_digital_funnel()
    agronomy = load_agronomy()
    reps = load_reps()

    products = list(SKU_TO_PRODUCT.values())
    weeks = _season_weeks(cfg["season"]["start"], cfg["season"]["end"])

    # --- Static per-tehsil features ---
    log.info("Computing per-tehsil static features...")
    grower_feats = build_tehsil_grower_features()
    dominant_crops = build_tehsil_dominant_crops(growers)

    # Retailer count per tehsil
    retailer_count = (
        retailers.groupby("tehsil")["retailer_id"].nunique().rename("n_retailers").reset_index()
    )

    # Tehsil → territory / district / state
    tehsil_meta = (
        retailers[["tehsil", "territory_id", "district", "state"]]
        .drop_duplicates("tehsil")
    )

    # --- Build the base grid: every (tehsil, week) in season ---
    log.info("Building tehsil×week grid...")
    all_tehsils = tehsil_meta["tehsil"].unique()
    grid = pd.DataFrame(
        [(t, w) for t in all_tehsils for w in weeks],
        columns=["tehsil", "week_start"],
    )

    # Join static features
    grid = grid.merge(tehsil_meta, on="tehsil", how="left")
    grid = grid.merge(retailer_count, on="tehsil", how="left")
    grid = grid.merge(grower_feats, on="tehsil", how="left")
    grid = grid.merge(dominant_crops[["tehsil", "dominant_crop", "crop_tier"]], on="tehsil", how="left")
    grid["n_retailers"] = grid["n_retailers"].fillna(0).astype(int)
    grid["n_growers"] = grid["n_growers"].fillna(0).astype(int)

    # --- Weekly POS revenue ---
    log.info("Computing weekly POS features...")
    weekly_pos, pos_total = _weekly_pos(pos, retailers)
    # Aggregate to (tehsil, week_start) total
    grid = grid.merge(pos_total, on=["tehsil", "week_start"], how="left")
    grid["weekly_pos_rev_total"] = grid["weekly_pos_rev_total"].fillna(0.0)

    # --- OOS rate ---
    oos = _weekly_oos(inv, retailers)
    grid = grid.merge(oos, on=["tehsil", "week_start"], how="left")
    grid["oos_rate"] = grid["oos_rate"].fillna(0.0)

    # --- Visit pressure ---
    log.info("Computing visit pressure...")
    vp = _weekly_visit_pressure(vis)
    grid = grid.merge(vp, on=["tehsil", "week_start"], how="left")
    grid["visit_count_week"] = grid["visit_count_week"].fillna(0).astype(int)
    grid["visit_pressure_2w"] = grid["visit_pressure_2w"].fillna(0).astype(int)

    # --- Digital funnel ---
    log.info("Computing digital funnel features...")
    dfun_weekly = _weekly_digital_funnel(dfun)
    grid = grid.merge(dfun_weekly, on=["dominant_crop", "week_start"], how="left")
    grid["funnel_impressions"] = grid["funnel_impressions"].fillna(0.0)
    grid["funnel_leads"] = grid["funnel_leads"].fillna(0.0)

    # --- Weather ---
    log.info("Fetching/loading weather for %d districts...", tehsil_meta["district"].nunique())
    district_state = (
        tehsil_meta[["district", "state"]].drop_duplicates("district").set_index("district")["state"]
    )
    district_triples = []
    for district in district_state.index:
        state = district_state[district]
        lat, lon = get_district_centroid(district, state)
        district_triples.append((district, lat, lon))

    weather_map = fetch_all_districts(
        district_triples,
        start_date=cfg["season"]["start"],
        end_date=cfg["season"]["end"],
        cache_dir=cache_weather,
        max_api_calls=50,
    )
    weather_df = pd.concat(
        [df.assign(district=district) for district, df in weather_map.items()],
        ignore_index=True,
    )
    grid = grid.merge(
        weather_df[["district", "week_start", "rainfall_mm_7d", "rh_max_7d",
                    "disease_pressure_days", "disease_pressure_flag"]],
        on=["district", "week_start"],
        how="left",
    )
    grid["rainfall_mm_7d"] = grid["rainfall_mm_7d"].fillna(0.0)
    grid["rh_max_7d"] = grid["rh_max_7d"].fillna(60.0)
    grid["disease_pressure_days"] = grid["disease_pressure_days"].fillna(0).astype(int)
    grid["disease_pressure_flag"] = grid["disease_pressure_flag"].fillna(0).astype(int)

    # --- Window features (vectorised) ---
    log.info("Computing agronomic window features for %d products...", len(products))
    grid = compute_window_features_vectorised(grid, agronomy, products)

    # --- Season index ---
    season_start = pd.Timestamp(cfg["season"]["start"])
    grid["week_index"] = ((grid["week_start"] - season_start).dt.days // 7).clip(lower=0)

    # --- Save ---
    out_path = _ROOT / cfg["paths"]["twin_parquet"]
    grid.to_parquet(out_path, index=False)
    log.info(
        "twin.parquet saved: %d rows × %d columns → %s",
        len(grid), len(grid.columns), out_path,
    )
    return out_path


def build_monthly_panel_with_features(twin_path: Optional[Path] = None) -> Path:
    """
    Enriches the monthly causal panel with twin (weekly) features
    averaged over each calendar month. Saves monthly_panel.parquet.
    """
    cfg = _settings()
    processed = _ROOT / cfg["paths"]["processed"]

    log.info("Building monthly causal panel...")
    panel = build_treatment_panel_monthly()

    # Load twin if it exists; add monthly-aggregated features
    if twin_path is None:
        twin_path = _ROOT / cfg["paths"]["twin_parquet"]

    if twin_path.exists():
        twin = pd.read_parquet(twin_path)
        twin["month_start"] = pd.to_datetime(twin["week_start"]).dt.to_period("M").dt.start_time

        # Monthly means of weather + visit pressure + funnel (non-product-specific)
        twin_monthly_agg = (
            twin.groupby(["tehsil", "month_start"])
            .agg(
                avg_oos_rate=("oos_rate", "mean"),
                avg_visit_pressure=("visit_pressure_2w", "mean"),
                avg_disease_pressure=("disease_pressure_flag", "mean"),
                avg_rainfall_mm=("rainfall_mm_7d", "mean"),
                avg_rh_max=("rh_max_7d", "mean"),
                avg_funnel_impressions=("funnel_impressions", "mean"),
            )
            .reset_index()
        )
        panel = panel.merge(twin_monthly_agg, on=["tehsil", "month_start"], how="left")

        # Per-product window features: average win_decay over the month
        decay_cols = [c for c in twin.columns if c.startswith("win_decay_")]
        if decay_cols:
            twin_win = twin.groupby(["tehsil", "month_start"])[decay_cols].mean().reset_index()
            panel = panel.merge(twin_win, on=["tehsil", "month_start"], how="left")

            # Add the window_decay for this row's specific product
            def _product_decay(row):
                col = f"win_decay_{_safe_col(row['product'])}"
                return row.get(col, 0.0)

            panel["window_decay_this_product"] = panel.apply(_product_decay, axis=1)
        else:
            panel["window_decay_this_product"] = 0.0
    else:
        warnings.warn("twin.parquet not found; monthly panel will lack twin features.", stacklevel=2)
        panel["avg_oos_rate"] = np.nan
        panel["avg_visit_pressure"] = np.nan
        panel["avg_disease_pressure"] = np.nan
        panel["window_decay_this_product"] = 0.0

    # Fill any remaining NaN in numeric columns
    num_cols = panel.select_dtypes(include=np.number).columns
    panel[num_cols] = panel[num_cols].fillna(0.0)

    out_path = processed / "monthly_panel.parquet"
    panel.to_parquet(out_path, index=False)
    log.info(
        "monthly_panel.parquet saved: %d rows × %d columns → %s",
        len(panel), len(panel.columns), out_path,
    )
    return out_path


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    twin_path = build_twin()

    from disha.twin.balance import write_balance_report
    from disha.twin.lift_probe import run_and_save_lift_probe

    panel_path = build_monthly_panel_with_features(twin_path)
    panel = pd.read_parquet(panel_path)

    processed = _ROOT / _settings()["paths"]["processed"]
    write_balance_report(panel, processed)
    lift_result = run_and_save_lift_probe(panel, processed / "lift_probe.json")

    from disha.twin.dgp_gate import run_and_save_dgp_gate
    run_and_save_dgp_gate(panel, lift_result, processed / "dgp_gate.json")

    log.info("=== L0 build complete. See data/processed/ for artefacts. ===")
