"""
disha.eval.business_case — the ONE bounded business number for the deck.

Output (data/processed/business_case.json):
  conservative_lift_inr_low / high     — CI-bounded annual lift
  assumptions {n_reps, season_months, ...} — every input is listed
  honesty_disclaimers                  — caveats the deck must say aloud

Closing line pattern (verbatim in deck + doc §4):
  "Syngenta runs ~N rep-days/season here; reallocating the bottom-Q% of
   zero/negative-CATE visits to the positive tail — same reps, same days,
   no added cost — is worth ₹LOW–₹HIGH/season under our OPE estimate
   (with the honesty caveats above)."
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED = _ROOT / "data" / "processed"


def compute_business_case() -> dict:
    ope = json.loads((_PROCESSED / "ope.json").read_text())
    panel = pd.read_parquet(_PROCESSED / "monthly_panel.parquet")
    reps = pd.read_csv(_ROOT / "data" / "raw" / "reps_territory.csv")

    # Treatable opportunity = in-window cells per season
    in_window_cells = int((panel["window_decay_this_product"] > 0).sum())
    bau_treated = int(panel["T"].sum())
    n_reps = len(reps)

    bau = ope["policies"]["bau_historical"]["dr_value_per_cell"]
    disha = ope["policies"]["disha"]["dr_value_per_cell"]
    disha_ci_low, disha_ci_high = ope["policies"]["disha"]["dr_ci_95"]
    lift_low  = disha_ci_low  - bau
    lift_high = disha_ci_high - bau

    # Conservative reallocation framing: assume Disha re-routes the SAME number
    # of treatments BAU does (bau_treated), at a per-cell lift in [lift_low, lift_high].
    annual_low  = lift_low  * bau_treated
    annual_high = lift_high * bau_treated

    out = {
        "headline_sentence": (
            f"Across {n_reps} reps × 6-month season ≈ {bau_treated:,} historical "
            f"treatment opportunities and {in_window_cells:,} in-window cells. "
            f"Reallocating treatments from zero/negative-CATE cells to the "
            f"positive-CATE tail (under our OPE DR estimate, with the honesty "
            f"caveats below) is worth ₹{annual_low/1e6:.1f}M – ₹{annual_high/1e6:.1f}M "
            "per season at zero added rep-day cost."
        ),
        "annual_lift_inr_low":  annual_low,
        "annual_lift_inr_high": annual_high,
        "lift_per_cell_low":    lift_low,
        "lift_per_cell_high":   lift_high,
        "assumptions": {
            "n_reps":             n_reps,
            "season_months":      6,
            "bau_treated_cells":  bau_treated,
            "in_window_cells":    in_window_cells,
            "bau_per_cell_inr":   bau,
            "disha_per_cell_inr": disha,
            "lift_ci_method":     "DR + 300-boot percentile",
        },
        "honesty_disclaimers": [
            "Real-data CATE absolute values are biased downward by uncorrected "
            "selection — see SOLUTION.md §3 + dgp_gate freeze note.  We use CF "
            "for RANKING (residualized Qini = +0.263, sourced from "
            "uplift_eval.json::qini_window_residualized, validated), not for "
            "absolute ₹ point estimates.",
            "OPE values are sanity-check magnitudes; a production-quality lift "
            "estimate would require a geo-randomized rollout (see SOLUTION.md §4).",
            "Number assumes Disha-routed visits are causal substitutes for BAU "
            "visits, with no transition-cost overhead or rep-acceptance friction.",
            "The HEADLINE for technical judges is the dual framing (synthetic "
            "engine validation r=0.86 + real residualized Qini +0.26) — this "
            "₹ band is the BUSINESS framing for managers, with caveats made aloud.",
        ],
    }
    return out


def run_and_save() -> dict:
    out = compute_business_case()
    p = _PROCESSED / "business_case.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    log.info("Saved → %s", p)
    return out


if __name__ == "__main__":
    # Force UTF-8 stdout on Windows so the headline (₹, ≈) doesn't crash cp1252.
    import sys, io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO)
    r = run_and_save()
    try:
        print(r["headline_sentence"])
    except UnicodeEncodeError:
        # Last-resort fallback — log.info also goes to stderr UTF-8 safely.
        log.info("headline saved to business_case.json (terminal does not support unicode)")
