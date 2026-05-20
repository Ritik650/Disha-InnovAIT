# Disha — On-Stage Pitch Run Sheet
**Syngenta AgriTech Hackathon 2026 · Finals · IIT Madras · 15 min + Q&A**

Use this as the spoken run sheet. The video script in `docs/VIDEO_SCRIPT.md`
covers the recorded ≤3-min demo; this document covers the LIVE 15-minute
on-campus presentation.

---

## The 7-beat arc (≈15 minutes)

### Beat 1 — Hook (0:00 → 0:45)
*"Every field-AI tool in this space optimises **propensity** — who buys
anyway. Agriculture is the one domain where that is the wrong objective,
because crops have a **biological deadline**. A fungicide visit two weeks
after the protection window closes has zero value, no matter how high the
predicted purchase probability."*

→ Two-sentence stake in the ground. Pause. No slide change.

---

### Beat 2 — The reframe (0:45 → 1:45)
*"Disha optimises causal **uplift** under a hard **agronomic-window
constraint**. A rep visit only counts if it happens while the crop's
protection window is open. Among feasible visits, we route to cells where
the marginal effect of the visit is highest — not where the absolute sales
would be. That single reframe — propensity → causal uplift under a
biological deadline — is the contribution."*

→ Show the architecture diagram (SOLUTION.md §3 system layers).

---

### Beat 3 — Method works (1:45 → 3:15)
*"On a pre-registered controlled DGP — coefficients written into code
before any estimator was run — our engine recovers the known true CATE at
Spearman r ≈ 0.86. That's the R-learner; T-learner gets 0.97 because the
DGP is linear, Causal Forest gets 0.86. All three agree on the ranking.
We then audited our R-learner against econml's standard implementation;
they agree at Spearman 0.83 cross-implementation on the same synthetic
data. The engine works."*

→ Switch to dashboard's SYNTHETIC ArmPanel (r=0.86 chip) + SyntheticRecovery chart.

---

### Beat 4 — Honesty (3:15 → 5:45)  [the differentiator]
*"On real Syngenta data, here is what we found and what we corrected
on ourselves:"*

1. *"A treatment-derived feature, `avg_visit_pressure`, was leaking into
   our CATE — visit pressure is literally a function of the treatment
   itself. Removing it dropped our cross-seed R-learner Spearman from
   0.66 to 0.43."*
2. *"Our R-learner Stage-2 prediction had a bug — it was returning
   Y-tilde at validation instead of the true τ̂ function evaluated at X.
   We caught it, fixed it, audited against econml. True r is 0.88."*
3. *"Two of our agronomic signals — demand spikes and out-of-stock —
   looked like CATE modifiers, but 74-79% of their effect disappeared
   when we partialled out scale. They were firing in bigger tehsils
   where CATE is mechanically lower. We demoted them to context-only,
   reason-text only."*

*"The honest, scale-residualized economic signal on real data is a +0.26
window-constrained Qini with Causal Forest. We refuse to inflate it.
The naive T-learner on the same data produces an apparent +0.56 Qini, but
**76 percent** of that disappears under the same residualization — that's
the 'send reps to bigger tehsils' baseline made explicit, and Disha beats
it provably and honestly."*

→ Dashboard QiniTable, highlighting T-learner row (the rust-tinted scale-router baseline) vs CF row (the leaf-tinted L3 source). Then HonestyArc section.

---

### Beat 5 — Feel it (5:45 → 8:45)  [LIVE PHONE DEMO]

→ Hand phone to second team member. Live walkthrough:

1. *"It's 7:30 Tuesday morning in Patna district. The rep opens Disha."*
2. *"Today's ordered stops. Already in his pocket, even if the cell tower's down."*
3. *"He taps the first one. The model says where this ranks and why —
   second of eleven candidates in his territory, top 18%, window closing
   in four days, driven by offline-campaign reach in this tehsil."*
4. *"Goes offline."* [toggle]
5. *"Logs the outcome. Queues locally."* [tap Completed]
6. *"Reconnects."* [toggle online]
7. *"Syncs automatically."*

*"That's a rep's morning. Plan, why, offline, sync. Real-data CATE
ranking — the exact +0.26 signal you just saw on the dashboard, not the
synthetic version. We don't substitute the synthetic data into the
field-facing surface."*

→ Three minutes. Do not rush. The phone moment IS the pitch.

---

### Beat 6 — Worth it (8:45 → 10:45)
*"Across 500 reps over a six-month season, that's roughly twenty thousand
historical treatment opportunities. Reallocating from zero- or
negative-CATE cells to the positive-CATE tail — same reps, same days, no
added cost — is worth ₹24.5 to ₹31.8 million per season under our OPE
estimate."*

*"Three caveats, stated aloud: First, OPE absolute values are biased
sanity-check magnitudes; production-grade validation needs a
geo-randomized rollout — we describe that in section 4.4 of the doc.
Second, our number assumes Disha-routed visits substitute cleanly for
BAU visits; transition costs are unmodeled. Third, the headline for the
technical judges is still the +0.26 Qini, not this rupee band — this is
the manager framing, with the data scientist's framing already shown."*

→ Dashboard BusinessCase card (LOW + HIGH bands with all disclaimers visible).

---

### Beat 7 — Close (10:45 → 11:30)
*"We are not showing you the biggest number. We are showing you the true
one — and it still pays for itself. That's Disha."*

→ Pause. All team members visible. End on the BusinessCase lower-bound number.

---

### Q&A buffer (11:30 → 15:00) — see bank below

---

## Q&A bank

These will be asked. Practice the answers. Each is in **strict honesty
mode** — the honesty arc IS the differentiator, do not retreat from it
when challenged.

### Q1. "Your ATE is zero — so why does this work?"
The population-average effect of rep visits, after proper two-way FE
identification, is statistically indistinguishable from zero — CI
straddles. Disha's value is in **heterogeneity capture** under the
deadline constraint, not ATE recovery. The positive-CATE tail exists
(57% of cells in our estimate), and re-routing finite rep-days *into*
that tail and *out of* the negative-CATE tail is where the lift comes
from. If we were trying to claim "rep visits work on average," we would
be lying. We're claiming the targeting is exploitable, and we have the
ranking metric to back it up.

### Q2. "Your real Qini is only 0.26 — why should we trust this?"
On known truth our method recovers it at r ≈ 0.86. On your real data we
show the honest scale-stripped number and refuse to inflate. The
T-learner row on our dashboard shows what inflation would look like —
apparent +0.56 Qini, 76% of which is just "visit bigger tehsils".
Causal Forest at +0.26 *after* stripping that out is modest but real.
The number is computed by `disha.eval.residualized_qini`, serialized to
`uplift_eval.json::qini_window_residualized.causal_forest.resid`, served
via `/qini`, and test-pinned by `tests/test_headline_integrity.py` — the
deck, doc, dashboard, and data are provably equal. Production
confirmation needs a randomized rollout, which we describe in
SOLUTION.md §4.4.

### Q3. "Is the rep app's ranking real or synthetic?"
**Real-data CATE ranking**, the exact same +0.26 signal you saw on the
dashboard. Synthetic is dashboard-only, clearly labelled, used for
engine validation against pre-registered truth. We don't substitute
synthetic data into the field-facing surface — that would be the exact
credibility problem we built the architecture to avoid.

### Q4. "How do you handle selection bias / confounding?"
Three things stacked: (a) two-way fixed effects on entity × time absorb
persistent rep-routing patterns, (b) cross-fit residualization in the
R-learner partials out the agronomic + grower features, (c) we
adversarially test our independence claims against a model-free
FE-stratified estimator that doesn't share the R-learner's known finite-
sample bias. The adversarial-test template is in
`tests/test_independence.py`.

### Q5. "Why tehsil-month grain? Why not weekly?"
Treatment is too sparse weekly — under 8% of cells per week — to give
the FE estimator enough within-entity variation. Monthly gets to ~26%
treated and absorbs the 1-3 week outcome lag without explicit
lead-lag bookkeeping. Weather and windows operate on multi-day
timescales anyway.

### Q6. "What if the agronomic windows are wrong / region-specific?"
We have a three-tier fallback: per-tehsil grower data → district → crop
default (wheat). 100% coverage on all 2,849 tehsils despite 450 null
grower calendars in the raw data. Critically, the window is a
**feasibility filter not a multiplier** — modest window-error degrades
gracefully (a few non-optimal stops) instead of inverting recommendations
(which a multiplicative formulation would do).

### Q7. "Cost / scalability at Syngenta production scale?"
Parquet files + FastAPI + no GPU. Compute on a single VM. The
optimizer's per-rep candidate set after window filtering is ≤ ~10
cells/day, so greedy-by-density is provably within constants of optimal
and finishes in milliseconds per rep. A Syngenta CRM/PoS integration is
a ~30-line connector reading from Cropwise / Salesforce / SAP feeds into
the same three parquet contracts — zero changes to optimizer, API, or apps.

### Q8. "Why mobile-web, not native?"
Rural India runs heterogeneous entry-tier Android. A mobile-web app
installs with one URL and updates without an app-store cycle. Our
IndexedDB sync-then-offline pattern matches the field constraint exactly:
the rep never sees "no internet" failure modes; outcomes queue locally
and flush on reconnect. We tested this in the demo flow.

### Q9. "What's your production-validation plan?"
Geo-randomized rollout. Pick a matched cluster of ~50 territories,
randomly assign 25 to Disha and 25 to BAU rotation for a quarter,
compare actual revenue per rep-quarter. Balanced design yields an
unbiased ATE in three months — exactly what the OPE was trying to
approximate. Full methodology in SOLUTION.md §4.4.

### Q10. "What would actually break this in the field?"
Three things, all stated as limitations in the doc, not hidden:
(a) rep acceptance friction — if reps don't trust the model they'll
override it; the WHY text is designed to short-circuit that, but we have
no field data on adoption.
(b) window data quality — our agronomic windows come from a 3-tier
fallback; in regions with sparse grower data the fallback hits more often
and the constraint becomes less precise.
(c) transition-cost overhead — moving from BAU rotation to Disha-routed
visits has frictions we haven't priced in.

### Q11. "What is genuinely novel here vs Aktana / Axtria / IQVIA?"
Those tools optimise propensity-of-purchase — they predict who buys.
None of them reason about a biological deadline as a hard constraint on
the optimization itself. Our contribution is the reframe — agriculture
is the one domain where the temporal feasibility constraint is
non-negotiable, and that changes the entire problem from "predict demand"
to "reallocate finite rep-days under a deadline to where presence
causes uplift." The downstream architecture — feasibility filter rather
than multiplier, residualization to defeat scale routing, dual-arm
honesty framing — is all consequences of that one insight.

### Q12. "You caught two bugs and demoted two of your own signals during
the build — does that worry you?"
The opposite. We built adversarial tests per layer that caught our own
errors, and they're pinned in `tests/test_independence.py` so they can't
regress. The template — characterize the estimator's bias on known
truth, re-prove the claim with a method that doesn't share that bias,
assert the two evidences give different numbers — is the discipline we
recommend any future team add to this project. The science is correct,
and we know exactly what it can and can't tell us.

### Q13. "Could a simple greedy heuristic over current visit logs do
this?"
No — and that's exactly the T-learner row on our dashboard. The naive
visit-rate-based ranking shows +0.56 apparent Qini, of which 76%
disappears under scale residualization. "Visit bigger tehsils" is what
a naive heuristic recovers; the marginal +0.26 of genuine economic
signal is what Disha specifically contributes. Both are visible on the
dashboard side-by-side; the contrast is the whole point.

### Q14. "Why dual demo path — isn't real data enough?"
On real data the absolute INR predictions are biased downward by
uncorrected selection, which we disclose openly. We use the *ranking*
of real-data CATE — that's the +0.26 Qini — but the absolute values
aren't credible without a randomized rollout. The synthetic arm proves
the engine recovers truth when truth is known; the real arm proves we
show honest numbers when truth isn't. Showing only one side would leave
the other unanswered.

### Q15. "Does it learn from outcomes — is the loop closed?"
The loop is fully instrumented — outcomes are captured offline, synced,
and drive the acceptance metric and a weekly recalibration trigger.
Closed-loop model retraining on logged outcomes is the documented next
step, not yet live. We deliberately don't claim it learns in the moment
because it doesn't yet — the trigger is wired, the retraining job is the
next sprint.

### Q16. "Are the NDVI, competitor, and pest-surveillance signals live feeds?"
No — all three are proxies in the prototype; the connector contract is
feed-agnostic, so swapping in live satellite NDVI ingestion, an
external competitor data feed, or government pest-surveillance-bulletin
ingestion requires zero model change. We're explicit about this rather
than implying integrations we don't have. The crop-health field is a
derived proxy from disease pressure and weather context; the competitor
field is a derived proxy from non-focus SKU velocity at the outlet;
the pest/disease alert is a weather-driven proxy (disease pressure ×
agronomic window), not from external government pest bulletins. All
three are the same architectural shape — proxy now, live feed via the
same connector when one is wired in.

---

## On-stage checklist (pre-pitch, 30 min before)

- [ ] `make demo` (or `docker compose up`) running, both URLs reachable
- [ ] Dashboard loaded on laptop, scrolled to top
- [ ] Rep app loaded on phone at `/rep`, sync pill says "Synced"
- [ ] Network ON for the live phone-offline-reconnect moment
- [ ] All team members present, mic if applicable
- [ ] PITCH.md and SOLUTION.md open on a second screen for reference
- [ ] Water + a deep breath before Beat 1
