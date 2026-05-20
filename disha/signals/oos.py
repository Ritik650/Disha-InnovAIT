"""
disha.signals.oos — Out-of-stock opportunity detector.

Condition: ≥10% of retailers for a (tehsil, product, week) are OOS.
Aggregated to month: flag if the month's max OOS rate exceeds threshold.

CATE modifier hypothesis: when product shelves are empty but demand is
present, a rep visit can directly influence retailer restocking decisions
and timing.  This is a supply-gap uplift opportunity, not a baseline
demand signal — it should be orthogonal to tehsil wealth.
"""
from __future__ import annotations

import pandas as pd

from disha.signals.base import SIGNAL_CONFIG


def compute_oos_opportunity(
    oos_product_df: pd.DataFrame,
    threshold: float | None = None,
) -> pd.DataFrame:
    """
    Compute OOS opportunity signal at (tehsil, month_start, product) grain.

    Parameters
    ----------
    oos_product_df : DataFrame with columns
        (tehsil, product, week_start, oos_rate).
        oos_rate = fraction of retailers stocking this product that are OOS.
    threshold : OOS rate threshold; defaults to SIGNAL_CONFIG value.

    Returns
    -------
    DataFrame with columns:
      tehsil, month_start, product,
      oos_opportunity_flag, oos_rate_max, oos_opportunity_reason
    """
    if threshold is None:
        threshold = SIGNAL_CONFIG["oos_rate_threshold"]

    df = oos_product_df[["tehsil", "product", "week_start", "oos_rate"]].copy()
    df["month_start"] = df["week_start"].dt.to_period("M").dt.start_time

    monthly = (
        df.groupby(["tehsil", "month_start", "product"])
        .agg(oos_rate_max=("oos_rate", "max"))
        .reset_index()
    )
    monthly["oos_opportunity_flag"] = (
        monthly["oos_rate_max"] >= threshold
    ).astype(int)
    monthly["oos_opportunity_reason"] = monthly.apply(
        lambda r: (
            f"Stock-out opportunity for {r['product']} in {r['tehsil']}: "
            f"{r['oos_rate_max']:.0%} of retailers OOS — "
            "rep visit can unlock restocking and influence next order."
        )
        if r["oos_opportunity_flag"]
        else "",
        axis=1,
    )
    return monthly[[
        "tehsil", "month_start", "product",
        "oos_opportunity_flag", "oos_rate_max", "oos_opportunity_reason",
    ]]
