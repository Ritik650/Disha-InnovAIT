"""
tests/test_signals.py

Pinned numeric regression tests for L1 signal detectors.

These tests pin BOTH:
  (a) signal-construction calibration (flag counts, top-event tuples), and
  (b) the honest CATE-correlation finding (all five signals are weak
      effect-modifiers: |spearman_r| < 0.05, |delta_mean_cate| < 25).

The CATE-correlation pinning is the load-bearing part — if a future refactor
silently makes one signal look like a strong CATE driver, it almost certainly
indicates a data-leakage or merge bug and we want the suite to scream.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from disha.signals import (
    SIGNAL_CONFIG,
    compute_demand_spike,
    compute_digital_demand,
    compute_disease_pressure_alert,
    compute_oos_opportunity,
    compute_signal_cate_correlations,
    compute_window_urgency,
    run_all_detectors,
)
from disha.signals.run import _load_oos_product_df, _load_wpos_df

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROCESSED = _PROJECT_ROOT / "data" / "processed"


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures (session-scoped — heavy I/O once)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def monthly_panel():
    path = _PROCESSED / "monthly_panel.parquet"
    if not path.exists():
        pytest.skip("monthly_panel.parquet not built")
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def cate_frozen():
    path = _PROCESSED / "cate_frozen.parquet"
    if not path.exists():
        pytest.skip("cate_frozen.parquet not built")
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def wpos_df():
    return _load_wpos_df()


@pytest.fixture(scope="session")
def oos_product_df():
    return _load_oos_product_df()


@pytest.fixture(scope="session")
def signals_panel(monthly_panel, wpos_df, oos_product_df):
    return run_all_detectors(monthly_panel, wpos_df, oos_product_df)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke / contract
# ──────────────────────────────────────────────────────────────────────────────

class TestSignalContract:
    """Each detector returns a DataFrame with the documented columns and grain."""

    def test_demand_spike_columns(self, wpos_df):
        out = compute_demand_spike(wpos_df)
        for col in ["tehsil", "month_start", "product",
                    "demand_spike_flag", "demand_spike_z", "demand_spike_reason"]:
            assert col in out.columns

    def test_oos_columns(self, oos_product_df):
        out = compute_oos_opportunity(oos_product_df)
        for col in ["tehsil", "month_start", "product",
                    "oos_opportunity_flag", "oos_rate_max", "oos_opportunity_reason"]:
            assert col in out.columns

    def test_disease_columns(self, monthly_panel):
        out = compute_disease_pressure_alert(monthly_panel)
        for col in ["tehsil", "month_start", "product",
                    "disease_alert_flag", "disease_alert_score", "disease_alert_reason"]:
            assert col in out.columns

    def test_window_columns(self, monthly_panel):
        out = compute_window_urgency(monthly_panel)
        for col in ["tehsil", "month_start", "product",
                    "window_urgency_flag", "window_urgency_decay", "window_urgency_reason"]:
            assert col in out.columns

    def test_digital_columns(self, monthly_panel):
        out = compute_digital_demand(monthly_panel)
        for col in ["tehsil", "month_start", "product",
                    "digital_demand_flag", "digital_demand_score", "digital_demand_reason"]:
            assert col in out.columns

    def test_panel_grain_unique(self, signals_panel):
        keys = ["tehsil", "month_start", "product"]
        assert not signals_panel.duplicated(subset=keys).any()

    def test_panel_row_count_matches_monthly(self, signals_panel, monthly_panel):
        # signals panel inherits its grain from the monthly panel
        assert len(signals_panel) == monthly_panel[
            ["tehsil", "month_start", "product"]
        ].drop_duplicates().shape[0]


# ──────────────────────────────────────────────────────────────────────────────
# Pinned flag-count calibration
# ──────────────────────────────────────────────────────────────────────────────

class TestFlagCalibration:
    """Pin total flag counts at default thresholds — catches threshold drift
    or detector-logic regressions."""

    def test_demand_spike_total(self, signals_panel):
        assert signals_panel["demand_spike_flag"].sum() == 19954

    def test_oos_opportunity_total(self, signals_panel):
        assert signals_panel["oos_opportunity_flag"].sum() == 6322

    def test_disease_alert_total(self, signals_panel):
        assert signals_panel["disease_alert_flag"].sum() == 16819

    def test_window_urgency_total(self, signals_panel):
        assert signals_panel["window_urgency_flag"].sum() == 3366

    def test_digital_demand_total(self, signals_panel):
        assert signals_panel["digital_demand_flag"].sum() == 104


# ──────────────────────────────────────────────────────────────────────────────
# Pinned top-of-list events — catch silent reordering / merge bugs
# ──────────────────────────────────────────────────────────────────────────────

class TestPinnedEvents:

    def test_top_demand_spike_is_bardhaman_t141(self, signals_panel):
        top = signals_panel[signals_panel.demand_spike_flag == 1].nlargest(
            1, "demand_spike_z"
        ).iloc[0]
        assert top["tehsil"] == "Bardhaman_T141"
        assert top["product"] == "Vibrance Integral"
        assert top["month_start"] == pd.Timestamp("2025-12-01")

    def test_pinned_oos_event_agra_t002_jan_tilt(self, signals_panel):
        row = signals_panel[
            (signals_panel.tehsil == "Agra_T002")
            & (signals_panel.month_start == pd.Timestamp("2026-01-01"))
            & (signals_panel["product"] == "Tilt 250 EC")
        ]
        assert len(row) == 1
        assert int(row.iloc[0]["oos_opportunity_flag"]) == 1
        assert row.iloc[0]["oos_rate_max"] == pytest.approx(1.0)

    def test_pinned_window_urgency_kavach_jan(self, signals_panel):
        """Kavach 75 WP hits peak window decay 0.964 in tehsils starting 2026-01-01."""
        sub = signals_panel[
            (signals_panel["product"] == "Kavach 75 WP")
            & (signals_panel.month_start == pd.Timestamp("2026-01-01"))
            & (signals_panel.window_urgency_flag == 1)
        ]
        assert len(sub) > 0
        assert sub["window_urgency_decay"].max() == pytest.approx(0.964286, abs=1e-4)

    def test_pinned_digital_signal_agra_t137_nov(self, signals_panel):
        row = signals_panel[
            (signals_panel.tehsil == "Agra_T137")
            & (signals_panel.month_start == pd.Timestamp("2025-11-01"))
            & (signals_panel["product"] == "Tilt 250 EC")
        ]
        assert len(row) == 1
        assert int(row.iloc[0]["digital_demand_flag"]) == 1


# ──────────────────────────────────────────────────────────────────────────────
# CATE-correlation finding — the most load-bearing assertion in the suite
# ──────────────────────────────────────────────────────────────────────────────

class TestSignalCateCorrelationsContract:
    """Contract-level checks on the correlations output (merge size, expected
    keys).  The substantive role-landscape assertion lives in
    tests/test_independence.py — it requires scale residualization to
    distinguish scale_artifact from effect_modifier (a distinction this
    file's previous tests missed)."""

    @pytest.fixture(scope="class")
    def results(self, signals_panel, cate_frozen):
        return compute_signal_cate_correlations(signals_panel, cate_frozen)

    def test_merge_size_full_coverage(self, results):
        assert results["n_total"] == 80128

    def test_all_signals_have_residualized_delta(self, results):
        """Defends against a future refactor that drops the residualized
        delta field — that field is what distinguishes scale_artifact
        from effect_modifier."""
        for name, r in results["signals"].items():
            assert "delta_mean_cate_residualized" in r, (
                f"{name} missing delta_mean_cate_residualized — refactor "
                "in correlations.py broke the residualized-delta pipeline."
            )
            assert "shrink_after_residualization" in r
