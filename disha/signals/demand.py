"""
disha.signals.demand — Demand spike detector.

Method: rolling z-score of weekly POS revenue vs 4-week trailing baseline.
Aggregated to (tehsil, month_start, product) by taking the max z-score within
the month.  Flag raised when z > threshold.

CATE modifier hypothesis: a sudden demand surge signals that growers are
actively buying — visit timing is optimal; rep can influence brand choice
and upsell protection products at the moment of willingness-to-buy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from disha.signals.base import SIGNAL_CONFIG


def compute_demand_spike(
    wpos_df: pd.DataFrame,
    threshold: float | None = None,
) -> pd.DataFrame:
    """
    Compute demand-spike signal at (tehsil, month_start, product) grain.

    Parameters
    ----------
    wpos_df : DataFrame with columns (tehsil, product, week_start, revenue).
    threshold : z-score threshold; defaults to SIGNAL_CONFIG value.

    Returns
    -------
    DataFrame with columns:
      tehsil, month_start, product,
      demand_spike_flag, demand_spike_z, demand_spike_reason
    """
    if threshold is None:
        threshold = SIGNAL_CONFIG["demand_spike_z_threshold"]

    df = wpos_df[["tehsil", "product", "week_start", "revenue"]].copy()
    df = df.sort_values(["tehsil", "product", "week_start"])

    grp = df.groupby(["tehsil", "product"])["revenue"]
    df["roll_mean"] = grp.transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).mean()
    )
    df["roll_std"] = grp.transform(
        lambda x: x.shift(1).rolling(4, min_periods=2).std()
    )
    std_safe = df["roll_std"].replace(0.0, np.nan)
    df["z_score"] = ((df["revenue"] - df["roll_mean"]) / std_safe).fillna(0.0)

    df["month_start"] = df["week_start"].dt.to_period("M").dt.start_time

    monthly = (
        df.groupby(["tehsil", "month_start", "product"])
        .agg(demand_spike_z=("z_score", "max"))
        .reset_index()
    )
    monthly["demand_spike_flag"] = (
        monthly["demand_spike_z"] > threshold
    ).astype(int)
    monthly["demand_spike_reason"] = monthly.apply(
        lambda r: (
            f"Demand surge for {r['product']} in {r['tehsil']}: "
            f"weekly sales {r['demand_spike_z']:.1f}σ above 4-week baseline — "
            "peak buying moment, rep visit highly timely."
        )
        if r["demand_spike_flag"]
        else "",
        axis=1,
    )
    return monthly[[
        "tehsil", "month_start", "product",
        "demand_spike_flag", "demand_spike_z", "demand_spike_reason",
    ]]
