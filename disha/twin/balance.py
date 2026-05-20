"""
disha.twin.balance — propensity overlap histogram + standardised mean difference report.

Outputs
-------
data/processed/balance_report.txt    text table of SMDs + overlap stats
data/processed/propensity_overlap.png overlap histogram (treated vs control)

Called by disha.twin.build after monthly_panel.parquet is built.
"""
from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Features used for the propensity model (pre-treatment, no Y, no T, no n_visits)
_PROPENSITY_FEATURES = [
    "lag_revenue_1m",
    "n_retailers_in_tehsil",
    "n_growers",
    "pct_smartphone",
    "pct_offline_attended",
    "pct_product_scanned",
    "avg_farm_size_ha",
    "wa_engagement_rate",
    "month_index",
    "days_since_season_start",
]


def _safe_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return covariate matrix with only available, non-null numeric columns."""
    avail = [c for c in _PROPENSITY_FEATURES if c in panel.columns]
    X = panel[avail].copy()
    X["log_lag_rev"] = np.log1p(X["lag_revenue_1m"].fillna(0))
    X = X.drop(columns=["lag_revenue_1m"], errors="ignore")
    X = X.fillna(0)
    features = [c for c in X.columns if X[c].std() > 0]
    return X[features], features


def fit_propensity_model(panel: pd.DataFrame) -> np.ndarray:
    """
    Fit a logistic regression P(T=1|X) on the monthly panel.
    Returns propensity score array aligned with panel index.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X, features = _safe_features(panel)
    y = panel["T"].values

    if len(features) == 0 or y.mean() in (0.0, 1.0):
        warnings.warn("Degenerate panel for propensity model; returning constant 0.5.")
        return np.full(len(y), 0.5)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    model.fit(Xs, y)
    propensity = model.predict_proba(Xs)[:, 1]
    return propensity


def standardised_mean_difference(
    panel: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    Compute SMD = (mean_treated - mean_control) / pooled_std for each feature.
    |SMD| < 0.1 is the conventional 'well-balanced' threshold.
    """
    treated = panel[panel["T"] == 1]
    control = panel[panel["T"] == 0]

    rows = []
    for feat in features:
        if feat not in panel.columns:
            continue
        m1 = treated[feat].mean()
        m0 = control[feat].mean()
        s1 = treated[feat].std(ddof=1)
        s0 = control[feat].std(ddof=1)
        pooled = np.sqrt((s1 ** 2 + s0 ** 2) / 2) if (s1 + s0) > 0 else 1.0
        smd = (m1 - m0) / pooled if pooled > 0 else 0.0
        rows.append({
            "feature": feat,
            "mean_treated": round(m1, 4),
            "mean_control": round(m0, 4),
            "SMD": round(smd, 4),
            "balanced": abs(smd) < 0.1,
        })
    return pd.DataFrame(rows).sort_values("SMD", key=abs, ascending=False)


def write_balance_report(panel: pd.DataFrame, output_dir: Path) -> dict:
    """
    Full balance pipeline: fit propensity, compute SMDs, plot histogram, write report.
    Returns a summary dict (also written to balance_summary.json).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_treated = int((panel["T"] == 1).sum())
    n_control = int((panel["T"] == 0).sum())
    treatment_rate = n_treated / max(len(panel), 1)

    log.info("Balance report: %d treated, %d control (rate=%.1f%%)", n_treated, n_control, treatment_rate * 100)

    propensity = fit_propensity_model(panel)
    panel = panel.copy()
    panel["propensity"] = propensity

    # Overlap stats
    p_treated = propensity[panel["T"] == 1]
    p_control = propensity[panel["T"] == 0]
    overlap_support = (
        p_treated.min() <= p_control.max() and p_control.min() <= p_treated.max()
    )
    min_overlap = max(p_treated.min(), p_control.min())
    max_overlap = min(p_treated.max(), p_control.max())
    overlap_width = max(0.0, max_overlap - min_overlap)

    # SMD table
    _, features = _safe_features(panel)
    smd_df = standardised_mean_difference(panel, _PROPENSITY_FEATURES)
    n_unbalanced = int((smd_df["SMD"].abs() > 0.1).sum())

    # ---- Text report ----
    lines = [
        "=" * 72,
        "  DISHA L0 — Propensity Overlap & Covariate Balance Report",
        "=" * 72,
        f"  Units: {len(panel):,}  |  Treated: {n_treated:,}  |  Control: {n_control:,}",
        f"  Treatment rate: {treatment_rate:.1%}",
        "",
        "  Propensity score (P(T=1|X)) distribution:",
        f"    Treated  — mean={p_treated.mean():.3f}  std={p_treated.std():.3f}  "
        f"[{p_treated.min():.3f}, {p_treated.max():.3f}]",
        f"    Control  — mean={p_control.mean():.3f}  std={p_control.std():.3f}  "
        f"[{p_control.min():.3f}, {p_control.max():.3f}]",
        f"    Overlap region: [{min_overlap:.3f}, {max_overlap:.3f}]  "
        f"width={overlap_width:.3f}  support={overlap_support}",
        "",
        "  Covariate balance (Standardised Mean Differences):",
        "  |SMD| < 0.10 = well balanced  |  > 0.25 = problematic",
        "",
        f"  {'Feature':<35} {'Mean(T=1)':>10} {'Mean(T=0)':>10} {'SMD':>8} {'OK?':>6}",
        "  " + "-" * 72,
    ]
    for _, row in smd_df.iterrows():
        ok = "YES" if row["balanced"] else "NO "
        lines.append(
            f"  {row['feature']:<35} {row['mean_treated']:>10.4f} {row['mean_control']:>10.4f} "
            f"{row['SMD']:>8.4f} {ok:>6}"
        )
    lines += [
        "",
        f"  Features with |SMD| > 0.10: {n_unbalanced} of {len(smd_df)}",
        "  (IPW / doubly-robust estimators correct for imbalance — see L2 uplift.)",
        "=" * 72,
    ]
    report_text = "\n".join(lines)

    report_path = output_dir / "balance_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    log.info("Balance report saved to %s", report_path)

    # ---- Histogram plot ----
    _plot_overlap(p_treated, p_control, output_dir / "propensity_overlap.png")

    summary = {
        "n_treated": n_treated,
        "n_control": n_control,
        "treatment_rate": round(treatment_rate, 4),
        "propensity_mean_treated": round(float(p_treated.mean()), 4),
        "propensity_mean_control": round(float(p_control.mean()), 4),
        "overlap_width": round(overlap_width, 4),
        "overlap_support": bool(overlap_support),
        "n_features_unbalanced": n_unbalanced,
        "n_features_checked": len(smd_df),
    }
    with open(output_dir / "balance_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def _plot_overlap(p_treated: np.ndarray, p_control: np.ndarray, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.linspace(0, 1, 31)
        ax.hist(p_control, bins=bins, alpha=0.55, color="#6baed6",
                label=f"Control (n={len(p_control):,})", density=True)
        ax.hist(p_treated, bins=bins, alpha=0.55, color="#fd8d3c",
                label=f"Treated (n={len(p_treated):,})", density=True)
        ax.set_xlabel("Propensity score  P(T=1 | X)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title("Propensity Score Overlap\n(good overlap = broad common support region)", fontsize=11)
        ax.legend(fontsize=10)
        ax.text(
            0.5, 0.93,
            "Key diagnostic: treated and control should overlap broadly.",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555555",
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
        log.info("Propensity overlap chart saved to %s", out_path)
    except Exception as exc:
        warnings.warn(f"Could not save propensity overlap plot: {exc}", stacklevel=2)
