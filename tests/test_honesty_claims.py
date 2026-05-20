"""tests/test_honesty_claims.py — pin honest claims in the public docs.

If a future edit re-introduces an overclaim phrase that contradicts what
the code actually does, this test fails.  The list is deliberately tiny
and exact-substring (case-insensitive) — every entry maps to a specific
gap between code and prior wording that the honesty pass closed.
"""
from __future__ import annotations

from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"

# Forbidden substrings (case-insensitive).  Each one was either present in
# an earlier draft or is a phrase a future editor might reach for that
# would silently overclaim what Disha actually does today.
FORBIDDEN = (
    "learns in real time",
    "continuously retrains",
    "live NDVI",
    "real-time competitor feed",
    "live pest bulletin",
    "live government pest surveillance",
)

# Positive presence checks — these strings MUST appear so a future edit
# can't quietly strip the proxy/roadmap labelling we landed.  Each one
# pins a specific external-feed gap (NDVI, competitor, pest bulletins).
REQUIRED_LABELS_SOLUTION = (
    "NDVI",
    "competitor",
    "pest-surveillance",
    "feed-agnostic",
    "roadmap",
)
REQUIRED_LABELS_PITCH = (
    "NDVI",
    "competitor",
    "pest-surveillance",
    "feed-agnostic",
)


def _read(name: str) -> str:
    return (DOCS / name).read_text(encoding="utf-8")


def test_pitch_md_has_no_overclaim_phrases():
    text = _read("PITCH.md").lower()
    hits = [p for p in FORBIDDEN if p.lower() in text]
    assert not hits, (
        f"docs/PITCH.md contains forbidden overclaim phrase(s): {hits}.  "
        "Each phrase implies behaviour Disha does not yet deliver; "
        "rephrase to match the instrumented-loop / proxy reality."
    )


def test_solution_md_has_no_overclaim_phrases():
    text = _read("SOLUTION.md").lower()
    hits = [p for p in FORBIDDEN if p.lower() in text]
    assert not hits, (
        f"docs/SOLUTION.md contains forbidden overclaim phrase(s): {hits}.  "
        "Each phrase implies behaviour Disha does not yet deliver; "
        "rephrase to match the instrumented-loop / proxy reality."
    )


def test_solution_md_labels_all_three_proxy_signals():
    """SOLUTION.md must list NDVI, competitor, and pest-surveillance as
    proxies with the feed-agnostic connector framing.  Pins the labelling
    so a future edit can't silently strip the architectural caveat."""
    text = _read("SOLUTION.md").lower()
    missing = [s for s in REQUIRED_LABELS_SOLUTION if s.lower() not in text]
    assert not missing, (
        f"docs/SOLUTION.md missing required proxy/roadmap labels: {missing}.  "
        "These were added in the honesty pass; a regression here would silently "
        "drop the NDVI / competitor / pest-surveillance caveats."
    )


def test_pitch_md_labels_all_three_proxy_signals():
    """PITCH.md's Q&A must explicitly cover NDVI, competitor, and
    pest-surveillance proxies — so a judge cross-referencing the data-signals
    list against the implementation gets a spoken answer that covers all three."""
    text = _read("PITCH.md").lower()
    missing = [s for s in REQUIRED_LABELS_PITCH if s.lower() not in text]
    assert not missing, (
        f"docs/PITCH.md missing required proxy/roadmap labels: {missing}.  "
        "The Q&A must cover all three proxies (NDVI, competitor, pest-surveillance) "
        "so the spoken answer matches the data-signals list."
    )
