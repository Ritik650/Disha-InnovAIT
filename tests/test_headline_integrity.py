"""tests/test_headline_integrity.py — close the prose-vs-data gap forever.

What this test pins
-------------------
Before this pass, "Causal Forest residualized window-Qini = +0.263" was
asserted in every narrative surface but lived nowhere in the data files.
The only serialized value was the RAW window-Qini (+0.155). A judge
looking at the dashboard saw +0.155 while the deck said +0.263.

This module makes that class of mismatch un-regressable:

  1. The residualized headline IS serialized to uplift_eval.json under
     `qini_window_residualized`.
  2. The number on disk EQUALS what the shared computation function
     returns (so the test and the artifact can never drift).
  3. Every headline figure quoted as the real-data residualized Qini in
     README, SOLUTION, and PITCH must round to that serialized number at
     2 dp — i.e. the slide and the data are provably the same.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_PROCESSED = _ROOT / "data" / "processed"
_UPLIFT_EVAL = _PROCESSED / "uplift_eval.json"


def _load_eval() -> dict:
    if not _UPLIFT_EVAL.exists():
        pytest.skip("uplift_eval.json not built")
    return json.loads(_UPLIFT_EVAL.read_text(encoding="utf-8"))


def test_residualized_qini_is_serialized():
    """uplift_eval.json carries a finite residualized CF window-Qini value
    that meets the load-bearing economic-signal floor."""
    blob = _load_eval()
    rq = blob.get("qini_window_residualized")
    assert rq is not None, (
        "uplift_eval.json missing `qini_window_residualized` — run "
        "`python -m disha.eval.residualized_qini` to populate it."
    )
    cf = rq.get("causal_forest", {}).get("resid")
    assert cf is not None, "qini_window_residualized.causal_forest.resid missing"
    assert isinstance(cf, (int, float)), f"resid must be numeric, got {type(cf)}"
    assert cf == cf, "resid is NaN"  # NaN check
    assert cf >= 0.15, (
        f"Serialized CF residualized window-Qini = {cf:+.4f} < 0.15.  "
        "The economic-uplift evidence used in the headline no longer holds."
    )


def test_serialized_equals_test_computation():
    """The number on disk must equal what disha.eval.residualized_qini
    actually computes — proves the artifact is the test's computation,
    not a stale or independently-edited value."""
    from disha.eval.residualized_qini import compute_residualized_window_qini

    blob = _load_eval()
    serialized = blob["qini_window_residualized"]["causal_forest"]["resid"]
    fresh = compute_residualized_window_qini()["causal_forest"]["resid"]
    assert round(serialized, 4) == round(fresh, 4), (
        f"Serialized CF resid = {serialized:.6f} but fresh compute = "
        f"{fresh:.6f}.  The artifact is stale relative to the code — "
        "re-run `python -m disha.eval.residualized_qini`."
    )


def test_docs_headline_matches_serialized():
    """Every Qini headline figure quoted as the real-data residualized
    metric in README/SOLUTION/PITCH must round to the serialized value at
    2 dp.  Closes the exact gap (prose +0.263, data +0.155) this pass fixed."""
    blob = _load_eval()
    cf_resid = blob["qini_window_residualized"]["causal_forest"]["resid"]
    cf_resid_2dp = round(cf_resid, 2)

    # Heuristic: pick out any "+0.NN" or "+0.NNN" that appears within ~80
    # chars of "residualized" / "headline" / "scale-stripped" /
    # "scale-residualized" in the headline-bearing docs.  Each such
    # number must round to cf_resid at 2 dp.
    headline_re = re.compile(
        r"(?i)(?:residualized|scale[- ]stripped|scale[- ]residualized|headline|"
        r"honest economic signal|honest scale-stripped)"
        r"[^\n.]{0,160}?\+\s*(0\.\d{1,3})"
    )
    secondary_re = re.compile(
        r"(?i)\+\s*(0\.\d{1,3})"
        r"[^\n.]{0,80}?(?:after\s+stripping|after\s+residualization|"
        r"scale[- ]stripped|residualized)"
    )

    files = [
        _ROOT / "README.md",
        _ROOT / "docs" / "SOLUTION.md",
        _ROOT / "docs" / "PITCH.md",
    ]
    failures: list[str] = []
    for f in files:
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        for m in list(headline_re.finditer(text)) + list(secondary_re.finditer(text)):
            quoted = float(m.group(1))
            if round(quoted, 2) != cf_resid_2dp:
                snippet = text[max(0, m.start() - 40): m.end() + 40].replace("\n", " ")
                failures.append(
                    f"{f.name}: quoted +{quoted} (rounds to {round(quoted,2)}) "
                    f"≠ serialized +{cf_resid:.4f} (rounds to {cf_resid_2dp})  "
                    f"near: …{snippet}…"
                )
    assert not failures, (
        "Headline Qini in docs does not match serialized residualized value:\n  "
        + "\n  ".join(failures)
    )
