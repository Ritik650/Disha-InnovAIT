"""
disha.signals.correlations — Signal–CATE correlation analysis.

For each signal we compute statistics that decide whether the signal is a
genuine CATE *effect-modifier* (use as L2 feature + L4 reason text) or
something else (scale artifact / too rare / weak).

Three load-bearing numbers per signal:
  1. spearman_r(signal_magnitude, cate_mean) — does signal strength rank CATE?
  2. delta_mean_cate   = E[CATE | flag=1] − E[CATE | flag=0]  (RAW delta)
  3. delta_mean_cate_residualized = same, but on CATE RESIDUALIZED by 4
     scale features (lag_revenue_1m, n_retailers_in_tehsil, n_growers,
     avg_farm_size_ha).  This isolates the economic component.

Role assignment (post pre-L3 corrections):
  context_only        — prevalence < 0.5%  OR  |residualized delta| < 100 INR
                        AND |spearman_r| < 0.05
  scale_artifact      — raw |delta| >= 100 but residualized |delta| < 25%
                        of raw |delta| (signal fires preferentially in
                        big tehsils where CATE is mechanically lower)
  effect_modifier     — residualized |delta| >= 100 INR and survives
                        residualization (shrink < 70%)

The residualization is the test that catches the trap "signal fires in big
cells; big cells have lower CATE → signal looks like CATE modifier but isn't".
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED = _PROJECT_ROOT / "data" / "processed"

# (signal_name, flag_col, magnitude_col)
_SIGNAL_COLS = [
    ("demand_spike",          "demand_spike_flag",      "demand_spike_z"),
    ("oos_opportunity",       "oos_opportunity_flag",   "oos_rate_max"),
    ("disease_pressure_alert", "disease_alert_flag",    "disease_alert_score"),
    ("window_urgency",        "window_urgency_flag",    "window_urgency_decay"),
    ("digital_demand_signal", "digital_demand_flag",    "digital_demand_score"),
]

# Scale features used to residualize CATE before measuring the "real"
# (non-scale-artifact) signal delta.  These are EXCLUDED from
# AGRONOMIC_CATE_FEATURES but predict baseline tehsil size/wealth.
SCALE_FEATURES = [
    "lag_revenue_1m",
    "n_retailers_in_tehsil",
    "n_growers",
    "avg_farm_size_ha",
]

KEEP_SPEARMAN_THRESHOLD = 0.05
KEEP_RAW_DELTA_THRESHOLD = 100.0       # INR/month, raw delta worth investigating
KEEP_RESIDUAL_DELTA_THRESHOLD = 100.0  # INR/month, post-scale-residualization
MIN_PREVALENCE = 0.005
SCALE_ARTIFACT_SHRINK_THRESHOLD = 0.70  # ≥70% of raw delta evaporates → artifact


def _residualize_cate(cate: np.ndarray, scale_df: pd.DataFrame) -> np.ndarray:
    """Remove the linear projection of CATE on standardized scale features."""
    S = scale_df.astype(float).fillna(0.0).values
    S = StandardScaler().fit_transform(S)
    mdl = Ridge(alpha=1.0).fit(S, cate)
    return cate - mdl.predict(S)


def compute_signal_cate_correlations(
    signals_panel_df: pd.DataFrame,
    cate_df: pd.DataFrame,
    panel_df: pd.DataFrame | None = None,
) -> dict:
    """
    Compute per-signal Spearman + raw delta + residualized delta against CATE.

    Parameters
    ----------
    signals_panel_df : output of run_all_detectors()
    cate_df          : cate_frozen.parquet
    panel_df         : monthly_panel.parquet (needed for scale features used
                       in residualization).  If None, attempts to load from
                       data/processed/monthly_panel.parquet.
    """
    keys = ["tehsil", "month_start", "product"]
    merged = signals_panel_df.merge(
        cate_df[keys + ["cate_mean"]], on=keys, how="inner"
    )

    if panel_df is None:
        panel_df = pd.read_parquet(_PROCESSED / "monthly_panel.parquet")
    merged = merged.merge(panel_df[keys + SCALE_FEATURES], on=keys, how="inner")

    n_total = len(merged)
    cate = merged["cate_mean"].astype(float).values
    cate_resid = _residualize_cate(cate, merged[SCALE_FEATURES])
    scale_var_explained = 1.0 - cate_resid.var() / cate.var()

    log.info("signal–CATE merge: %d rows; scale features explain %.1f%% of CATE variance",
             n_total, 100 * scale_var_explained)

    results: dict = {
        "n_total": int(n_total),
        "cate_mean_overall": float(cate.mean()),
        "cate_std_overall": float(cate.std()),
        "scale_variance_explained_by_residualization": float(scale_var_explained),
        "scale_features": list(SCALE_FEATURES),
        "signals": {},
    }

    for sig_name, flag_col, mag_col in _SIGNAL_COLS:
        flag = merged[flag_col].astype(int).values
        mag = merged[mag_col].astype(float).values
        n_flagged = int(flag.sum())
        prevalence = float(n_flagged / n_total) if n_total else 0.0

        if mag.std() > 0 and n_flagged > 0:
            r, p = spearmanr(mag, cate)
            spearman_r = float(r) if not np.isnan(r) else 0.0
            spearman_p = float(p) if not np.isnan(p) else 1.0
        else:
            spearman_r = 0.0
            spearman_p = 1.0

        if n_flagged > 0 and n_flagged < n_total:
            cate_on = float(cate[flag == 1].mean())
            cate_off = float(cate[flag == 0].mean())
            delta = cate_on - cate_off
            cate_resid_on = float(cate_resid[flag == 1].mean())
            cate_resid_off = float(cate_resid[flag == 0].mean())
            delta_resid = cate_resid_on - cate_resid_off
        else:
            cate_on = cate_off = float("nan")
            delta = 0.0
            cate_resid_on = cate_resid_off = float("nan")
            delta_resid = 0.0

        shrink = (
            1.0 - abs(delta_resid) / abs(delta) if abs(delta) > 1e-9 else 0.0
        )

        # Role assignment — order matters: prevalence guard first, then
        # scale-artifact check takes priority over effect_modifier.
        if prevalence < MIN_PREVALENCE:
            role = "context_only"
            keep_as_l2 = False
        elif (abs(delta) >= KEEP_RAW_DELTA_THRESHOLD
              and shrink >= SCALE_ARTIFACT_SHRINK_THRESHOLD):
            # Raw delta was meaningful but ≥70% of it disappears under
            # residualization → the negative delta was mechanical scale
            # correlation, not an economic effect.
            role = "scale_artifact"
            keep_as_l2 = False
        elif abs(delta_resid) >= KEEP_RESIDUAL_DELTA_THRESHOLD:
            role = "effect_modifier"
            keep_as_l2 = True
        elif abs(spearman_r) >= KEEP_SPEARMAN_THRESHOLD:
            role = "effect_modifier"
            keep_as_l2 = True
        else:
            role = "context_only"
            keep_as_l2 = False

        results["signals"][sig_name] = {
            "n_flagged": n_flagged,
            "prevalence": prevalence,
            "spearman_r_magnitude_vs_cate": spearman_r,
            "spearman_p": spearman_p,
            "mean_cate_flag_on": cate_on,
            "mean_cate_flag_off": cate_off,
            "delta_mean_cate": delta,
            "delta_mean_cate_residualized": delta_resid,
            "shrink_after_residualization": float(shrink),
            "keep_as_l2_feature": keep_as_l2,
            "role": role,
        }
        log.info(
            "%-26s prev=%.1f%% r=%+.3f  raw_d=%+.0f  resid_d=%+.0f  shrink=%.0f%%  -> %s",
            sig_name, prevalence * 100, spearman_r, delta, delta_resid,
            100 * shrink, role,
        )

    return results


def run_and_save_correlations(
    signals_panel: pd.DataFrame | None = None,
    cate_df: pd.DataFrame | None = None,
    panel_df: pd.DataFrame | None = None,
    out_dir: Path | None = None,
) -> dict:
    out_dir = out_dir or _PROCESSED
    if signals_panel is None:
        signals_panel = pd.read_parquet(_PROCESSED / "signals_panel.parquet")
    if cate_df is None:
        cate_df = pd.read_parquet(_PROCESSED / "cate_frozen.parquet")
    if panel_df is None:
        panel_df = pd.read_parquet(_PROCESSED / "monthly_panel.parquet")

    results = compute_signal_cate_correlations(signals_panel, cate_df, panel_df)
    out_path = out_dir / "signal_cate_correlations.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    log.info("saved → %s", out_path)
    return results
