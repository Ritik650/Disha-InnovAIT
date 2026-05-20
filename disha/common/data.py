"""
disha.common.data — typed data loaders, schema validation, and the core modelling bridges.

Key modelling decision (document here, repeated in ARCHITECTURE.md):
  The visit log has no retailer_id — visits are at (rep_id, tehsil, date) granularity.
  POS outcomes are at retailer_id granularity.
  Bridge assumption: treatment is modelled at the (tehsil, week, product) level.
    T=1  iff ≥1 visit of type {retailer meeting, campaign_conducted} recommending that product
             occurred in that tehsil during ISO-week W.
    Y    =  POS revenue for retailers whose tehsil == visit_tehsil, for SKUs matching that
             product, in weeks W+1 through W+outcome_horizon (config).
  This is a conscious approximation: we cannot observe which specific retailer a rep visited.
  The assumption — that a tehsil-level visit affects all retailers in that tehsil — is
  conservative (dilutes uplift) and documented as a limitation.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_AGRONOMY_PATH = _ROOT / "config" / "agronomy.yaml"
_SETTINGS_PATH = _ROOT / "config" / "settings.yaml"
_RAW = _ROOT / "data" / "raw"


# ── Configuration loaders ──────────────────────────────────────────────────────

def load_settings() -> dict:
    with open(_SETTINGS_PATH) as f:
        return yaml.safe_load(f)


def load_agronomy() -> dict:
    with open(_AGRONOMY_PATH) as f:
        return yaml.safe_load(f)


# ── Typed raw loaders (validates schema on load) ───────────────────────────────

_SCHEMAS: dict[str, list[str]] = {
    "reps_territory": ["rep_id", "territory_id", "state", "district", "tehsil_list"],
    "retailers": ["retailer_id", "territory_id", "state", "district", "tehsil"],
    "retailer_visit_log": ["rep_id", "visit_date", "territory_id", "visit_tehsil",
                           "visit_type", "product_recommended"],
    "retailer_pos": ["retailer_id", "transaction_id", "sku_id", "sku_name",
                     "sku_qty", "sku_price", "transaction_date"],
    "retailer_inventory_weekly": ["retailer_id", "sku_id", "sku_name", "sku_qty", "week_end_date"],
    "growers": ["grower_id", "state", "district", "tehsil", "language", "device_type",
                "grower_age", "gender", "grower_crop_calendar", "product_scan",
                "grower_farm_size", "offline_campaign_attended"],
    "digital_funnel_weekly": ["campaign_id", "week_start_date", "campaign_crop", "campaign_product"],
    "whatsapp_campaign": ["id", "campaign_product", "campaign_crop", "grower_id",
                          "message_sent_date", "delivered_status", "opened_status", "clicked_status"],
}


def _load_csv(name: str) -> pd.DataFrame:
    path = _RAW / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw data file: {path}")
    df = pd.read_csv(path, low_memory=False)
    required = _SCHEMAS[name]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}.csv missing columns: {missing}")
    return df


def load_reps() -> pd.DataFrame:
    df = _load_csv("reps_territory")
    df["tehsil_list"] = df["tehsil_list"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else []
    )
    return df


def load_retailers() -> pd.DataFrame:
    return _load_csv("retailers")


def load_visit_log() -> pd.DataFrame:
    df = _load_csv("retailer_visit_log")
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    df["visit_week"] = df["visit_date"].dt.to_period("W-SUN").dt.start_time
    return df


def load_pos() -> pd.DataFrame:
    df = _load_csv("retailer_pos")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["revenue"] = df["sku_qty"] * df["sku_price"]
    df["transaction_week"] = df["transaction_date"].dt.to_period("W-SUN").dt.start_time
    return df


def load_inventory() -> pd.DataFrame:
    df = _load_csv("retailer_inventory_weekly")
    df["week_end_date"] = pd.to_datetime(df["week_end_date"])
    df["is_oos"] = df["sku_qty"] == 0
    return df


def load_growers() -> pd.DataFrame:
    df = _load_csv("growers")
    # Parse crop calendar JSON, fill nulls with empty dict
    def _parse_cal(x):
        if pd.isna(x):
            return {}
        try:
            return json.loads(x)
        except Exception:
            return {}
    df["crop_calendar"] = df["grower_crop_calendar"].apply(_parse_cal)
    df["crop"] = df["crop_calendar"].apply(lambda d: d.get("crop", None))
    return df


def load_digital_funnel() -> pd.DataFrame:
    df = _load_csv("digital_funnel_weekly")
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    return df


def load_whatsapp() -> pd.DataFrame:
    df = _load_csv("whatsapp_campaign")
    df["message_sent_date"] = pd.to_datetime(df["message_sent_date"])
    return df


# ── SKU → product name bridge ──────────────────────────────────────────────────
# Maps sku_name (POS) → product_recommended (visit log).
# Manually verified from DATA_DICTIONARY and the two datasets.
SKU_TO_PRODUCT: dict[str, str] = {
    "Score 250 EC":      "Score 250 EC",
    "Axial 50 EC":       "Axial 50 EC",
    "Vibrance Integral": "Vibrance Integral",
    "Alto 5 SC":         "Alto 5 SC",
    "Topik 15 WP":       "Topik 15 WP",
    "Movondo":           "Movondo",
    "Kavach 75 WP":      "Kavach 75 WP",
    "Tilt 250 EC":       "Tilt 250 EC",
    "Cruiser 350 FS":    "Cruiser 350 FS",
    "Amistar 250 SC":    "Amistar 250 SC",
    "Actara 25 WG":      "Actara 25 WG",
    "Vertimec 1.8 EC":   "Vertimec 1.8 EC",
}

PRODUCT_TO_SKU: dict[str, str] = {v: k for k, v in SKU_TO_PRODUCT.items()}


# ── (tehsil, week, product) treatment-outcome bridge ──────────────────────────

def build_treatment_panel(
    horizon_weeks: int = 3,
    treatment_types: tuple[str, ...] = ("retailer meeting", "campaign_conducted"),
) -> pd.DataFrame:
    """
    Build the core modelling panel at (tehsil, week_start, product) granularity.

    Treatment T = 1 iff ≥1 visit of the relevant type recommending that product
    occurred in that tehsil during that ISO week.

    Outcome Y = total POS revenue from retailers in that tehsil for the matching
    SKU over the following `horizon_weeks` weeks (weeks W+1 to W+horizon_weeks).

    Returns a DataFrame with columns:
        tehsil, week_start, product, T, Y_revenue,
        territory_id (from retailers lookup),
        n_retailers_in_tehsil, n_visits
    """
    visit = load_visit_log()
    pos = load_pos()
    retailers = load_retailers()

    tehsil_territory = retailers.groupby("tehsil")["territory_id"].first().reset_index()
    tehsil_retailer_count = (
        retailers.groupby("tehsil")["retailer_id"].nunique().rename("n_retailers_in_tehsil")
    )

    # --- Treatment side ---
    visit_treatment = visit[visit["visit_type"].isin(treatment_types)].copy()
    treatment = (
        visit_treatment.groupby(
            ["visit_tehsil", "visit_week", "product_recommended"]
        ).size().rename("n_visits").reset_index()
        .rename(columns={"visit_tehsil": "tehsil", "visit_week": "week_start",
                         "product_recommended": "product"})
    )
    treatment["T"] = 1

    # --- Weekly revenue table (tehsil, product, week_start) ---
    pos_with_tehsil = pos.merge(
        retailers[["retailer_id", "tehsil", "territory_id"]], on="retailer_id", how="left"
    ).dropna(subset=["tehsil"])
    pos_with_tehsil["product"] = pos_with_tehsil["sku_name"].map(SKU_TO_PRODUCT)
    pos_with_tehsil = pos_with_tehsil.dropna(subset=["product"])

    weekly_rev = (
        pos_with_tehsil.groupby(["tehsil", "product", "transaction_week"])["revenue"]
        .sum().reset_index().rename(columns={"transaction_week": "week_start"})
    )

    # --- Vectorised outcome computation ---
    # For each (tehsil, product, week_start=W), sum revenue in weeks W+7, W+14 … W+horizon*7
    # Strategy: self-join weekly_rev shifted by each offset, then aggregate.
    outcome_parts = []
    for offset_w in range(1, horizon_weeks + 1):
        shifted = weekly_rev.copy()
        shifted["week_start"] = shifted["week_start"] - pd.Timedelta(days=7 * offset_w)
        shifted = shifted.rename(columns={"revenue": f"rev_shift_{offset_w}"})
        outcome_parts.append(shifted)

    outcome = outcome_parts[0]
    for part in outcome_parts[1:]:
        outcome = outcome.merge(part, on=["tehsil", "product", "week_start"], how="outer")

    rev_cols = [f"rev_shift_{i}" for i in range(1, horizon_weeks + 1)]
    outcome["Y_revenue"] = outcome[rev_cols].fillna(0).sum(axis=1)
    outcome = outcome[["tehsil", "product", "week_start", "Y_revenue"]]

    # --- Union of all (tehsil, week, product) observation units ---
    # Treated units come from treatment; control units from weekly_rev (with observed POS)
    all_units = pd.concat([
        treatment[["tehsil", "week_start", "product", "T", "n_visits"]],
        weekly_rev[["tehsil", "product", "week_start"]].assign(T=0, n_visits=0),
    ], ignore_index=True)
    # Drop duplicate treated units (treated table takes priority)
    all_units = (
        all_units.sort_values("T", ascending=False)
        .drop_duplicates(subset=["tehsil", "week_start", "product"])
        .reset_index(drop=True)
    )

    # Attach outcomes
    panel = all_units.merge(outcome, on=["tehsil", "product", "week_start"], how="left")
    panel["Y_revenue"] = panel["Y_revenue"].fillna(0.0)

    # Attach territory and retailer count
    panel = panel.merge(tehsil_territory, on="tehsil", how="left")
    panel = panel.merge(tehsil_retailer_count, on="tehsil", how="left")

    return panel.sort_values(["tehsil", "week_start", "product"]).reset_index(drop=True)


# ── Agronomic window function ──────────────────────────────────────────────────

@dataclass
class WindowStatus:
    product: str
    crop: str
    stage: str
    open: bool
    days_left: int       # days until window closes (negative = already closed)
    decay_factor: float  # 1.0 = full value; 0.0 = window closed
    source_tier: str = "direct"  # "direct"|"tehsil"|"district"|"default"


def get_tehsil_district_map() -> dict[str, str]:
    """Returns {tehsil: district} using retailers (broader coverage) + growers."""
    retailers = load_retailers()
    growers = load_growers()
    # Growers may cover tehsils not in retailers; retailers take priority (cleaner data)
    grw_map = dict(zip(growers["tehsil"], growers["district"]))
    ret_map = dict(zip(retailers["tehsil"], retailers["district"]))
    return {**grw_map, **ret_map}


def get_tehsil_dominant_crop(
    tehsil: str,
    growers_df: pd.DataFrame,
    tehsil_district_map: Optional[dict[str, str]] = None,
) -> tuple[str, str]:
    """
    3-tier fallback for dominant crop assignment.
    Returns (crop, source_tier) where source_tier ∈ {"tehsil","district","default"}.
    """
    tehsil_crops = growers_df.loc[
        (growers_df["tehsil"] == tehsil) & growers_df["crop"].notna(), "crop"
    ]
    if not tehsil_crops.empty:
        return tehsil_crops.value_counts().index[0], "tehsil"

    district = (tehsil_district_map or {}).get(tehsil)
    if district:
        district_crops = growers_df.loc[
            (growers_df["district"] == district) & growers_df["crop"].notna(), "crop"
        ]
        if not district_crops.empty:
            return district_crops.value_counts().index[0], "district"

    return "wheat", "default"


def window_status(
    tehsil: str,
    product: str,
    query_date: date | datetime | str,
    tehsil_dominant_crop: Optional[str] = None,
    growers_df: Optional[pd.DataFrame] = None,
    district: Optional[str] = None,
) -> list[WindowStatus]:
    """
    Return window status for (tehsil, product, date) per agronomy.yaml.

    3-tier dominant-crop fallback (evaluated in order):
      tier "direct"   — tehsil_dominant_crop was passed explicitly
      tier "tehsil"   — inferred from growers in that specific tehsil
      tier "district" — inferred from growers across the district (district param or from map)
      tier "default"  — falls back to wheat (dominant Rabi crop)

    Each returned WindowStatus carries the source_tier that resolved the crop.
    An empty list means no product↔crop match in agronomy.yaml for this date.
    """
    if isinstance(query_date, str):
        query_date = datetime.fromisoformat(query_date).date()
    elif isinstance(query_date, datetime):
        query_date = query_date.date()

    agro = load_agronomy()
    product_cfg = agro.get("products", {}).get(product)
    if product_cfg is None:
        return []

    # --- Determine crop + tier ---
    source_tier = "direct"
    if tehsil_dominant_crop is not None:
        resolved_crop: Optional[str] = tehsil_dominant_crop
        source_tier = "direct"
    elif growers_df is not None:
        tehsil_crops = growers_df.loc[
            (growers_df["tehsil"] == tehsil) & growers_df["crop"].notna(), "crop"
        ]
        if not tehsil_crops.empty:
            resolved_crop = tehsil_crops.value_counts().index[0]
            source_tier = "tehsil"
        else:
            # District fallback
            eff_district = district
            if eff_district is None:
                dist_col = growers_df.loc[growers_df["tehsil"] == tehsil, "district"]
                eff_district = dist_col.iloc[0] if not dist_col.empty else None

            if eff_district is not None:
                dist_crops = growers_df.loc[
                    (growers_df["district"] == eff_district) & growers_df["crop"].notna(), "crop"
                ]
                if not dist_crops.empty:
                    resolved_crop = dist_crops.value_counts().index[0]
                    source_tier = "district"
                else:
                    resolved_crop = "wheat"
                    source_tier = "default"
            else:
                resolved_crop = "wheat"
                source_tier = "default"
    else:
        resolved_crop = "wheat"
        source_tier = "default"

    fallback_cals = agro.get("fallback_calendars", {})
    results: list[WindowStatus] = []

    for stage_entry in product_cfg.get("stages", []):
        crop = stage_entry["crop"]
        stage_name = stage_entry["stage"]

        if resolved_crop is not None and crop != resolved_crop:
            continue

        open_before = stage_entry.get(
            "window_open_days_before", agro["defaults"]["window_open_days_before"]
        )
        close_after = stage_entry.get(
            "window_close_days_after", agro["defaults"]["window_close_days_after"]
        )

        cal = fallback_cals.get(crop, {})
        stage_approx_str = next(
            (s["approx"] for s in cal.get("stages", []) if s["stage"] == stage_name),
            None,
        )
        if stage_approx_str is None:
            continue

        stage_approx = date.fromisoformat(stage_approx_str)
        window_open = stage_approx - timedelta(days=open_before)
        window_close = stage_approx + timedelta(days=close_after)

        is_open = window_open <= query_date <= window_close
        days_left = (window_close - query_date).days

        if query_date <= stage_approx:
            decay = 1.0
        elif query_date <= window_close:
            decay = max(0.0, 1.0 - (query_date - stage_approx).days / close_after)
        else:
            decay = 0.0

        results.append(WindowStatus(
            product=product,
            crop=crop,
            stage=stage_name,
            open=is_open,
            days_left=days_left,
            decay_factor=round(decay, 4),
            source_tier=source_tier,
        ))

    return results


def best_window_status(
    tehsil: str,
    product: str,
    query_date: date | datetime | str,
    tehsil_dominant_crop: Optional[str] = None,
    growers_df: Optional[pd.DataFrame] = None,
) -> Optional[WindowStatus]:
    """Return the single most urgent open window, or None if no window is open."""
    statuses = window_status(tehsil, product, query_date, tehsil_dominant_crop, growers_df)
    open_windows = [s for s in statuses if s.open]
    if not open_windows:
        return None
    return min(open_windows, key=lambda s: s.days_left)


# ── Per-tehsil static feature tables ──────────────────────────────────────────

def build_tehsil_grower_features() -> pd.DataFrame:
    """
    Per-tehsil grower demographic aggregates.
    Returns one row per tehsil with columns:
        tehsil, n_growers, avg_farm_size_ha, pct_smartphone,
        pct_offline_attended, pct_product_scanned, language_mode
    """
    grw = load_growers()

    def _safe_mode(s: pd.Series) -> str:
        m = s.mode()
        return m.iloc[0] if not m.empty else "Hindi"

    feats = (
        grw.groupby("tehsil")
        .agg(
            n_growers=("grower_id", "count"),
            avg_farm_size_ha=("grower_farm_size", "mean"),
            pct_smartphone=("device_type", lambda x: (x == "smartphone").mean()),
            pct_offline_attended=("offline_campaign_attended", "mean"),
            pct_product_scanned=("product_scan", "mean"),
            language_mode=("language", _safe_mode),
        )
        .reset_index()
    )
    return feats


def build_tehsil_dominant_crops(
    growers_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    For every tehsil present in retailers + growers, compute dominant crop
    using the 3-tier fallback. Returns:
        tehsil, dominant_crop, crop_tier
    """
    if growers_df is None:
        growers_df = load_growers()
    retailers = load_retailers()
    tdm = get_tehsil_district_map()

    all_tehsils = sorted(
        set(retailers["tehsil"].unique()) | set(growers_df["tehsil"].unique())
    )
    rows = []
    for t in all_tehsils:
        crop, tier = get_tehsil_dominant_crop(t, growers_df, tdm)
        rows.append({"tehsil": t, "dominant_crop": crop, "crop_tier": tier})
    return pd.DataFrame(rows)


# ── Monthly causal panel (primary grain for L2 uplift) ────────────────────────

def build_treatment_panel_monthly(
    treatment_types: tuple[str, ...] = ("retailer meeting", "campaign_conducted"),
) -> pd.DataFrame:
    """
    Primary causal panel at (tehsil × month_start × product) grain.

    Treatment T = 1 iff ≥1 relevant visit in that tehsil-month for that product.
    Outcome  Y = POS revenue of retailers in that tehsil for that product
                 in month_start + 1 (the following calendar month).
    Covariate lag_revenue_1m = revenue in month_start - 1 (pre-treatment baseline).

    Also joins per-tehsil grower demographics and WhatsApp engagement.
    """
    visit = load_visit_log()
    pos = load_pos()
    retailers = load_retailers()
    growers_df = load_growers()
    wa = load_whatsapp()

    tehsil_territory = retailers.groupby("tehsil")["territory_id"].first().reset_index()
    tehsil_n_retailers = (
        retailers.groupby("tehsil")["retailer_id"].nunique().rename("n_retailers_in_tehsil")
    )
    grower_feats = build_tehsil_grower_features()

    # ---- Treatment side ----
    visit = visit.copy()
    visit["month_start"] = visit["visit_date"].dt.to_period("M").dt.start_time
    visit_treat = visit[visit["visit_type"].isin(treatment_types)]
    treatment = (
        visit_treat.groupby(["visit_tehsil", "month_start", "product_recommended"])
        .size().rename("n_visits").reset_index()
        .rename(columns={"visit_tehsil": "tehsil", "product_recommended": "product"})
    )
    treatment["T"] = 1

    # ---- Monthly revenue table ----
    pos = pos.copy()
    pos["month_start"] = pos["transaction_date"].dt.to_period("M").dt.start_time
    pos_tehsil = pos.merge(
        retailers[["retailer_id", "tehsil"]], on="retailer_id", how="left"
    ).dropna(subset=["tehsil"])
    pos_tehsil["product"] = pos_tehsil["sku_name"].map(SKU_TO_PRODUCT)
    pos_tehsil = pos_tehsil.dropna(subset=["product"])

    monthly_rev = (
        pos_tehsil.groupby(["tehsil", "product", "month_start"])["revenue"]
        .sum().reset_index()
    )

    # ---- Vectorised outcome: revenue in month+1 ----
    outcome = monthly_rev.copy()
    outcome["month_start"] = outcome["month_start"] - pd.offsets.MonthBegin(1)
    outcome = outcome.rename(columns={"revenue": "Y_revenue"})

    # ---- Lagged revenue: revenue in month-1 (pre-treatment baseline) ----
    lag1 = monthly_rev.copy()
    lag1["month_start"] = lag1["month_start"] + pd.offsets.MonthBegin(1)
    lag1 = lag1.rename(columns={"revenue": "lag_revenue_1m"})

    # ---- Union of all observation units ----
    all_units = pd.concat([
        treatment[["tehsil", "month_start", "product", "T", "n_visits"]],
        monthly_rev[["tehsil", "product", "month_start"]].assign(T=0, n_visits=0),
    ], ignore_index=True)
    all_units = (
        all_units.sort_values("T", ascending=False)
        .drop_duplicates(subset=["tehsil", "month_start", "product"])
        .reset_index(drop=True)
    )

    panel = all_units.merge(
        outcome[["tehsil", "product", "month_start", "Y_revenue"]],
        on=["tehsil", "product", "month_start"], how="left"
    )
    panel["Y_revenue"] = panel["Y_revenue"].fillna(0.0)
    panel = panel.merge(
        lag1[["tehsil", "product", "month_start", "lag_revenue_1m"]],
        on=["tehsil", "product", "month_start"], how="left"
    )
    panel["lag_revenue_1m"] = panel["lag_revenue_1m"].fillna(0.0)

    # ---- WhatsApp engagement per (tehsil, product, month) ----
    wa = wa.copy()
    wa["month_start"] = wa["message_sent_date"].dt.to_period("M").dt.start_time
    wa_growers = wa.merge(
        growers_df[["grower_id", "tehsil"]], on="grower_id", how="left"
    ).dropna(subset=["tehsil"])
    wa_eng = (
        wa_growers.groupby(["tehsil", "campaign_product", "month_start"])
        .agg(
            wa_sent=("delivered_status", "count"),
            wa_opened=("opened_status", "sum"),
            wa_clicked=("clicked_status", "sum"),
        ).reset_index()
        .rename(columns={"campaign_product": "product"})
    )
    wa_eng["wa_engagement_rate"] = wa_eng["wa_opened"] / wa_eng["wa_sent"].clip(lower=1)
    panel = panel.merge(
        wa_eng[["tehsil", "product", "month_start", "wa_engagement_rate"]],
        on=["tehsil", "product", "month_start"], how="left"
    )
    panel["wa_engagement_rate"] = panel["wa_engagement_rate"].fillna(0.0)

    # ---- Static joins ----
    panel = panel.merge(tehsil_territory, on="tehsil", how="left")
    panel = panel.merge(tehsil_n_retailers, on="tehsil", how="left")
    panel = panel.merge(grower_feats, on="tehsil", how="left")

    # ---- Season features ----
    panel["month_index"] = panel["month_start"].dt.month  # 10..3
    season_start = pd.Timestamp("2025-10-01")
    panel["days_since_season_start"] = (panel["month_start"] - season_start).dt.days.clip(lower=0)

    # Restrict to Rabi season months only (Oct 2025 – Mar 2026)
    season_months = {10, 11, 12, 1, 2, 3}
    panel = panel[panel["month_index"].isin(season_months)].copy()

    return panel.sort_values(["tehsil", "month_start", "product"]).reset_index(drop=True)


# ── Data quality report ────────────────────────────────────────────────────────

def data_quality_report() -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("  DISHA — Data Quality Report")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 72)

    # Reps & Territory
    reps = load_reps()
    lines.append(f"\n[reps_territory]  {len(reps):,} rows")
    lines.append(f"  Unique reps:        {reps.rep_id.nunique():,}")
    lines.append(f"  Unique territories: {reps.territory_id.nunique():,}")
    lines.append(f"  States:             {sorted(reps.state.unique())}")
    tehsil_counts = reps["tehsil_list"].apply(len)
    lines.append(f"  Tehsils/territory:  min={tehsil_counts.min()} mean={tehsil_counts.mean():.1f} max={tehsil_counts.max()}")

    # Retailers
    ret = load_retailers()
    lines.append(f"\n[retailers]  {len(ret):,} rows")
    lines.append(f"  Unique retailers:   {ret.retailer_id.nunique():,}")
    lines.append(f"  Unique tehsils:     {ret.tehsil.nunique():,}")
    lines.append(f"  Null territory_id:  {ret.territory_id.isna().sum():,}")

    # Visit log
    vis = load_visit_log()
    lines.append(f"\n[retailer_visit_log]  {len(vis):,} rows")
    lines.append(f"  Date range:         {vis.visit_date.min().date()} → {vis.visit_date.max().date()}")
    lines.append(f"  Unique reps:        {vis.rep_id.nunique():,}")
    lines.append(f"  Unique tehsils:     {vis.visit_tehsil.nunique():,}")
    lines.append(f"  Visit types:        {dict(vis.visit_type.value_counts())}")
    lines.append(f"  Products:           {sorted(vis.product_recommended.dropna().unique())}")
    lines.append(f"  Null product:       {vis.product_recommended.isna().sum():,}")

    # POS
    pos = load_pos()
    lines.append(f"\n[retailer_pos]  {len(pos):,} rows")
    lines.append(f"  Date range:         {pos.transaction_date.min().date()} → {pos.transaction_date.max().date()}")
    lines.append(f"  Unique retailers:   {pos.retailer_id.nunique():,}")
    lines.append(f"  SKUs:               {pos.sku_name.nunique():,}")
    total_rev = pos.revenue.sum()
    lines.append(f"  Total revenue:      ₹{total_rev:,.0f}")
    lines.append(f"  Mean price/unit:    ₹{pos.sku_price.mean():,.2f}")
    sku_rev = pos.groupby("sku_name")["revenue"].sum().sort_values(ascending=False)
    for sku, rev in sku_rev.items():
        lines.append(f"    {sku:<25} ₹{rev:>14,.0f}  ({100*rev/total_rev:.1f}%)")

    # Inventory
    inv = load_inventory()
    lines.append(f"\n[retailer_inventory_weekly]  {len(inv):,} rows")
    lines.append(f"  Week range:         {inv.week_end_date.min().date()} → {inv.week_end_date.max().date()}")
    lines.append(f"  Unique weeks:       {inv.week_end_date.nunique():,}")
    lines.append(f"  OOS rate (qty==0):  {inv.is_oos.mean()*100:.2f}%")

    # Growers
    grw = load_growers()
    lines.append(f"\n[growers]  {len(grw):,} rows")
    lines.append(f"  Null crop calendar: {grw.grower_crop_calendar.isna().sum():,}")
    lines.append(f"  Crop distribution:  {dict(grw.crop.value_counts())}")
    lines.append(f"  Device types:       {dict(grw.device_type.value_counts())}")
    lines.append(f"  Avg farm size (ha): {grw.grower_farm_size.mean():.2f}")
    lines.append(f"  % offline attended: {grw.offline_campaign_attended.mean()*100:.1f}%")
    lines.append(f"  % product scanned:  {grw.product_scan.mean()*100:.1f}%")

    # Digital funnel
    dfun = load_digital_funnel()
    lines.append(f"\n[digital_funnel_weekly]  {len(dfun):,} rows")
    lines.append(f"  Campaigns: {sorted(dfun.campaign_id.unique())}")

    # WhatsApp
    wa = load_whatsapp()
    lines.append(f"\n[whatsapp_campaign]  {len(wa):,} rows")
    lines.append(f"  Open rate:    {wa.opened_status.mean()*100:.1f}%")
    lines.append(f"  Click rate:   {wa.clicked_status.mean()*100:.1f}%")

    # Treatment-outcome bridge summary
    lines.append("\n[treatment-outcome bridge (tehsil, week, product)]")
    vis_treatment = vis[vis["visit_type"].isin(["retailer meeting", "campaign_conducted"])]
    lines.append(f"  Treatment visits:   {len(vis_treatment):,}")
    lines.append(f"  Unique tehsil-week-product combos (T=1): "
                 f"{vis_treatment.groupby(['visit_tehsil','visit_week','product_recommended']).ngroups:,}")
    # Tehsil overlap
    vis_tehsils = set(vis_treatment["visit_tehsil"].unique())
    ret_tehsils = set(ret["tehsil"].unique())
    grw_tehsils = set(grw["tehsil"].unique())
    lines.append(f"  Tehsils in visits ∩ retailers: {len(vis_tehsils & ret_tehsils):,}")
    lines.append(f"  Tehsils in visits ∩ growers:   {len(vis_tehsils & grw_tehsils):,}")
    lines.append(f"  Tehsils in retailers ∩ growers:{len(ret_tehsils & grw_tehsils):,}")

    lines.append("\n" + "=" * 72)
    return "\n".join(lines)


if __name__ == "__main__":
    # Ensure UTF-8 output on Windows
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    if "--validate-only" in sys.argv:
        # Lightweight check: just load all files
        for loader in [load_reps, load_retailers, load_visit_log, load_pos,
                       load_inventory, load_growers, load_digital_funnel, load_whatsapp]:
            loader()
        print("All data files validated OK.")
    else:
        print(data_quality_report())
