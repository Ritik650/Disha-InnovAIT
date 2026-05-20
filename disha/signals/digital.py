"""
disha.signals.digital — Digital demand signal detector.

Condition: wa_engagement_rate > wa_threshold AND
           pct_product_scanned > scan_threshold.

This is a tehsil-level signal (same value for all products in the tehsil)
because engagement and scan rates measure grower digital readiness, not
product-specific intent.  It is broadcast across all products.

CATE modifier hypothesis: digitally-engaged growers who have already
scanned products are primed to act on rep recommendations.  Rep visits
to these tehsils convert more reliably regardless of the specific product.
This signal explains who-responds, not how-much — it is an effect-modifier
for interaction, not a revenue predictor.
"""
from __future__ import annotations

import pandas as pd

from disha.signals.base import SIGNAL_CONFIG


def compute_digital_demand(
    panel_df: pd.DataFrame,
    wa_threshold: float | None = None,
    scan_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Compute digital demand signal at (tehsil, month_start, product) grain.

    The signal is tehsil-level, broadcast to all products in that tehsil-month.

    Parameters
    ----------
    panel_df : monthly panel with columns
        (tehsil, month_start, product, wa_engagement_rate, pct_product_scanned).
    wa_threshold, scan_threshold : thresholds; defaults from SIGNAL_CONFIG.

    Returns
    -------
    DataFrame with columns:
      tehsil, month_start, product,
      digital_demand_flag, digital_demand_score, digital_demand_reason
    """
    if wa_threshold is None:
        wa_threshold = SIGNAL_CONFIG["digital_wa_threshold"]
    if scan_threshold is None:
        scan_threshold = SIGNAL_CONFIG["digital_scan_threshold"]

    need = ["tehsil", "month_start", "product",
            "wa_engagement_rate", "pct_product_scanned"]
    df = panel_df[need].copy()

    df["digital_demand_score"] = (
        df["wa_engagement_rate"] * df["pct_product_scanned"]
    )
    df["digital_demand_flag"] = (
        (df["wa_engagement_rate"] > wa_threshold) &
        (df["pct_product_scanned"] > scan_threshold)
    ).astype(int)
    df["digital_demand_reason"] = df.apply(
        lambda r: (
            f"Digital demand signal in {r['tehsil']}: "
            f"WhatsApp engagement active ({r['wa_engagement_rate']:.0%}) and "
            f"{r['pct_product_scanned']:.0%} of growers have scanned products — "
            "primed for rep recommendation."
        )
        if r["digital_demand_flag"]
        else "",
        axis=1,
    )
    return df[[
        "tehsil", "month_start", "product",
        "digital_demand_flag", "digital_demand_score", "digital_demand_reason",
    ]]
