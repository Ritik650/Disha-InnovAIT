"""
disha.signals.window — Agronomic window urgency detector.

Condition: window_decay_this_product > threshold (peak protection zone).
This flags cells where the biological clock is loudest — protection is
maximally needed and the window is not yet closed.

CATE modifier hypothesis: this is the hardest deadline constraint in the
system.  High decay means: (a) the crop is in peak susceptibility now,
(b) the window is still open but closing.  Rep visit at this moment has
the highest marginal causal value.  After the window closes, the visit
has near-zero uplift regardless of other signals.
"""
from __future__ import annotations

import pandas as pd

from disha.signals.base import SIGNAL_CONFIG


def compute_window_urgency(
    panel_df: pd.DataFrame,
    threshold: float | None = None,
) -> pd.DataFrame:
    """
    Compute window urgency signal at (tehsil, month_start, product) grain.

    Parameters
    ----------
    panel_df : monthly panel with columns
        (tehsil, month_start, product, window_decay_this_product).
    threshold : minimum decay to flag; defaults to SIGNAL_CONFIG value.

    Returns
    -------
    DataFrame with columns:
      tehsil, month_start, product,
      window_urgency_flag, window_urgency_decay, window_urgency_reason
    """
    if threshold is None:
        threshold = SIGNAL_CONFIG["window_urgency_decay_threshold"]

    need = ["tehsil", "month_start", "product", "window_decay_this_product"]
    df = panel_df[need].copy()
    df = df.rename(columns={"window_decay_this_product": "window_urgency_decay"})

    df["window_urgency_flag"] = (
        df["window_urgency_decay"] > threshold
    ).astype(int)
    df["window_urgency_reason"] = df.apply(
        lambda r: (
            f"Agronomic window peak urgency for {r['product']} in {r['tehsil']}: "
            f"protection decay {r['window_urgency_decay']:.2f} — "
            "window is open and time-sensitive; missing this visit forfeits uplift."
        )
        if r["window_urgency_flag"]
        else "",
        axis=1,
    )
    return df[[
        "tehsil", "month_start", "product",
        "window_urgency_flag", "window_urgency_decay", "window_urgency_reason",
    ]]
