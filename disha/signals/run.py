"""
disha.signals.run — Load raw data, run all five L1 detectors, merge into
a unified signal panel, and optionally save to disk.

Output grain: (tehsil, month_start, product) — one row per cell.
Columns include all five signal flags + magnitudes + reason strings.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from disha.signals.demand import compute_demand_spike
from disha.signals.digital import compute_digital_demand
from disha.signals.disease import compute_disease_pressure_alert
from disha.signals.oos import compute_oos_opportunity
from disha.signals.window import compute_window_urgency

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RAW = _PROJECT_ROOT / "data" / "raw"
_PROCESSED = _PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Internal data loaders
# ---------------------------------------------------------------------------

def _load_wpos_df() -> pd.DataFrame:
    """Weekly POS revenue per (tehsil, product, week_start)."""
    pos = pd.read_csv(_RAW / "retailer_pos.csv", parse_dates=["transaction_date"])
    retailers = pd.read_csv(_RAW / "retailers.csv")[["retailer_id", "tehsil"]]
    pos = pos.merge(retailers, on="retailer_id", how="left")

    pos["week_start"] = pos["transaction_date"] - pd.to_timedelta(
        pos["transaction_date"].dt.dayofweek, unit="D"
    )
    pos = pos.rename(columns={"sku_name": "product", "sku_price": "revenue"})
    wpos = (
        pos.groupby(["tehsil", "product", "week_start"])["revenue"]
        .sum()
        .reset_index()
    )
    return wpos


def _load_oos_product_df() -> pd.DataFrame:
    """Weekly OOS rate per (tehsil, product, week_start).

    oos_rate = fraction of stocking retailers whose qty=0 for that product-week.
    Retailers that don't appear for a product in a week are excluded (they may
    not carry that product at all).
    """
    inv = pd.read_csv(
        _RAW / "retailer_inventory_weekly.csv",
        parse_dates=["week_end_date"],
    )
    retailers = pd.read_csv(_RAW / "retailers.csv")[["retailer_id", "tehsil"]]
    inv = inv.merge(retailers, on="retailer_id", how="left")

    inv["week_start"] = inv["week_end_date"] - pd.to_timedelta(6, unit="D")
    inv = inv.rename(columns={"sku_name": "product"})
    inv["is_oos"] = (inv["sku_qty"] == 0).astype(int)

    oos = (
        inv.groupby(["tehsil", "product", "week_start"])
        .agg(oos_rate=("is_oos", "mean"))
        .reset_index()
    )
    return oos


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_all_detectors(
    monthly_panel_df: pd.DataFrame,
    wpos_df: pd.DataFrame,
    oos_product_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run all five signal detectors and merge into a single panel.

    Parameters
    ----------
    monthly_panel_df : enriched monthly panel (from monthly_panel.parquet).
    wpos_df : weekly POS (tehsil, product, week_start, revenue).
    oos_product_df : weekly OOS (tehsil, product, week_start, oos_rate).

    Returns
    -------
    DataFrame at (tehsil, month_start, product) grain with all signal columns.
    """
    demand = compute_demand_spike(wpos_df)
    oos = compute_oos_opportunity(oos_product_df)
    disease = compute_disease_pressure_alert(monthly_panel_df)
    window = compute_window_urgency(monthly_panel_df)
    digital = compute_digital_demand(monthly_panel_df)

    keys = ["tehsil", "month_start", "product"]

    # Start from the monthly panel index to preserve all (tehsil×month×product) cells
    base = monthly_panel_df[keys].copy().drop_duplicates()

    for sig_df in [demand, oos, disease, window, digital]:
        base = base.merge(sig_df, on=keys, how="left")

    # Fill NaN flags/scores from signals that don't cover all rows (e.g. demand/OOS
    # require weekly data which may not exist for every month)
    flag_cols = [c for c in base.columns if c.endswith("_flag")]
    base[flag_cols] = base[flag_cols].fillna(0).astype(int)

    score_cols = [
        "demand_spike_z", "oos_rate_max",
        "disease_alert_score", "window_urgency_decay",
        "digital_demand_score",
    ]
    for col in score_cols:
        if col in base.columns:
            base[col] = base[col].fillna(0.0)

    reason_cols = [c for c in base.columns if c.endswith("_reason")]
    for col in reason_cols:
        if col in base.columns:
            base[col] = base[col].fillna("")

    log.info(
        "signal panel: %d rows, %d flagged demand, %d OOS, %d disease, "
        "%d window, %d digital",
        len(base),
        base["demand_spike_flag"].sum(),
        base["oos_opportunity_flag"].sum(),
        base["disease_alert_flag"].sum(),
        base["window_urgency_flag"].sum(),
        base["digital_demand_flag"].sum(),
    )
    return base


def run_and_save_signals(
    panel_path: Path | None = None,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Load data, run all detectors, save signals_panel.parquet.

    Returns the signal panel DataFrame.
    """
    panel_path = panel_path or (_PROCESSED / "monthly_panel.parquet")
    out_dir = out_dir or _PROCESSED

    monthly = pd.read_parquet(panel_path)
    wpos = _load_wpos_df()
    oos_df = _load_oos_product_df()

    signals = run_all_detectors(monthly, wpos, oos_df)

    out_path = out_dir / "signals_panel.parquet"
    signals.to_parquet(out_path, index=False)
    log.info("saved → %s", out_path)
    return signals


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_and_save_signals()
    # Also recompute the CATE correlations now that signals exist on disk.
    from disha.signals.correlations import run_and_save_correlations
    run_and_save_correlations()
