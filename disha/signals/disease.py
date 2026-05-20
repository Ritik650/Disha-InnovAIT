"""
disha.signals.disease — Disease pressure alert detector.

Condition: mean disease_pressure_flag > threshold in the month AND
           window_decay_this_product > threshold (crop is in active window).

CATE modifier hypothesis: when weather creates disease risk AND the
protection window is open, rep visits to prescribe the correct fungicide /
pesticide are maximally actionable.  This is the primary CATE heterogeneity
driver identified in L0.
"""
from __future__ import annotations

import pandas as pd

from disha.signals.base import SIGNAL_CONFIG


def compute_disease_pressure_alert(
    panel_df: pd.DataFrame,
    dp_threshold: float | None = None,
    window_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Compute disease-pressure alert at (tehsil, month_start, product) grain.

    Parameters
    ----------
    panel_df : monthly panel with columns
        (tehsil, month_start, product, avg_disease_pressure,
         window_decay_this_product).
    dp_threshold : minimum avg_disease_pressure to trigger; default from config.
    window_threshold : minimum window decay to be in-window; default from config.

    Returns
    -------
    DataFrame with columns:
      tehsil, month_start, product,
      disease_alert_flag, disease_alert_score, disease_alert_reason
    """
    if dp_threshold is None:
        dp_threshold = SIGNAL_CONFIG["disease_pressure_threshold"]
    if window_threshold is None:
        window_threshold = SIGNAL_CONFIG["disease_window_decay_threshold"]

    need = ["tehsil", "month_start", "product",
            "avg_disease_pressure", "window_decay_this_product"]
    df = panel_df[need].copy()

    df["disease_alert_score"] = (
        df["avg_disease_pressure"] * df["window_decay_this_product"]
    )
    df["disease_alert_flag"] = (
        (df["avg_disease_pressure"] > dp_threshold) &
        (df["window_decay_this_product"] > window_threshold)
    ).astype(int)
    df["disease_alert_reason"] = df.apply(
        lambda r: (
            f"Disease pressure alert for {r['product']} in {r['tehsil']}: "
            f"weather risk active (score {r['avg_disease_pressure']:.2f}) "
            f"while protection window open (decay {r['window_decay_this_product']:.2f}) — "
            "targeted fungicide/pesticide prescription timely. "
            "Note: disease pressure is a weather-driven proxy; government "
            "pest-surveillance-bulletin ingestion is roadmap via the same "
            "feed-agnostic connector contract."
        )
        if r["disease_alert_flag"]
        else "",
        axis=1,
    )
    return df[[
        "tehsil", "month_start", "product",
        "disease_alert_flag", "disease_alert_score", "disease_alert_reason",
    ]]
