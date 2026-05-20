# Disha (दिशा) — Solution Document
**Syngenta AgriTech Hackathon 2026 · Track 2 · Field-Force Intelligence**

---

## §1 — Problem and Insight

Every field-AI tool currently optimises **propensity**: who will buy anyway.
Agriculture is the one domain where this is the wrong objective, because
crops have a **biological deadline**. A fungicide visit two weeks after the
protection window closes has zero value, no matter how high the predicted
purchase probability.

**Disha optimises causal uplift under a hard agronomic-window constraint.**
A rep visit only "counts" if it happens while the crop's protection window
is open; among feasible visits, we route to cells where the *marginal*
effect of a visit is highest — not where the *absolute* sales would be.

This reframes the field-force problem from "predict demand" to
"reallocate finite rep-days to the cells where presence causes uplift,
under a non-negotiable temporal constraint." The optimization, model and
narrative are all consequences of this single insight.

### §1.1 — Why this is non-obvious

Naïve uplift formulations would multiply CATE by window pressure — they
treat the window as a continuous heterogeneity driver. We tested this.
On real Syngenta data, the agronomic window is **bottom-ranked** in our
top-driver decomposition (|coef|=93 vs the top driver pct_offline_attended
at 726, an 8× gap), and the model-free FE-stratified Spearman across
window strata is +0.10 (p=0.87). The window does *not* modify CATE.
It modifies **feasibility**: visit inside the window or the uplift is
literally zero. The right architecture is a hard constraint, not a
multiplier. That is the L3 design.

### §1.2 — Why this is hard

- Data has severe selection bias: reps systematically visit lower-revenue
  tehsils (`lag_revenue` SMD = −0.65, `n_retailers` SMD = −1.06). A naive
  T-learner reproduces 76% of "send reps to bigger tehsils" — a useless
  baseline that has nothing to do with agronomy.
- Treatment-derived features (like `avg_visit_pressure`) silently leak
  into the CATE estimator if you let them; we caught this and excluded
  them after seeing a +0.25 inflation in cross-seed Spearman.
- Real-data Y is heavily centred on negative residuals after FE
  identification (ATE ≈ −₹1.7k with CI [−₹3.3k, +₹0.1k]), so absolute
  per-cell uplift numbers are not credible without a randomised rollout.
  Our targeting evidence is therefore the **ranking** (scale-residualized
  Qini), not the absolute INR.

---

## §3 — System Design

Six-layer build, each frozen before the next begins:

```
L0  Territory Digital Twin     →  monthly_panel.parquet (80,128 cells)
L1  Signal / Anomaly Detection →  5 detectors, 4 economic, 1 scale, 1 context
L2  Causal Uplift Engine       →  T/S/R-learner + Causal-Forest, cross-validated
L3  Deadline-Aware Optimiser   →  dual-arm router, window = hard constraint
L4  FastAPI                    →  7 endpoints, serves frozen artifacts
L5  Off-Policy Evaluation       →  IPS + DR sanity-check; rollout in §4
```

### §3.1 — L0: Territory Digital Twin

`disha/twin/`. Builds `(tehsil × month × product)` causal panel from raw
POS / inventory / visit-log / weather / agronomy data. Treatment T = 1
iff any retailer-meeting or campaign visit in tehsil-month recommended
that product. Outcome Y = next-3-week POS revenue for that SKU in that
tehsil. Agronomic window function `window_status(tehsil, product, date)`
encodes per-crop, per-stage protection windows with decay. Three-tier
fallback (tehsil → district → default) gives 100% coverage on 2,849
tehsils.

### §3.2 — L1: Signals

`disha/signals/`. Five detectors (demand_spike, oos_opportunity,
disease_pressure_alert, window_urgency, digital_demand_signal). After
scale residualization (partialling out lag_revenue + n_retailers +
n_growers + avg_farm_size from CATE), each signal's role is determined:

| Signal | Role | Used as |
|---|---|---|
| demand_spike | **scale_artifact** (74% shrink) | L4 reason text only |
| oos_opportunity | **scale_artifact** (79% shrink) | L4 reason text only |
| disease_pressure_alert | **effect_modifier** (−₹977 resid) | L2 negative feature |
| window_urgency | **effect_modifier** (−₹1,481 resid) | L2 negative feature |
| digital_demand_signal | context_only (prevalence 0.13%) | L4 reason text only |

The scale-residualization step was the key correction: two signals
*looked* like CATE modifiers under raw ΔCATE but were actually firing in
bigger tehsils where CATE is mechanically lower. We caught this and
demoted them.

### §3.3 — L2: Causal Uplift Engine

`disha/uplift/`. Four estimators trained and validated on **two** panels:

1. **Synthetic** — pre-registered DGP (`disha.uplift.dgp.DGP_SPEC_V1`).
   τ_true is a linear function of five grower behavioural / structural
   features with `β_window = β_disease = 0` *as a load-bearing
   independence test*. Coefficients were written into code BEFORE any
   estimator ran (`scripts/audit_r_learner.py` is the receipt).

2. **Real** — `monthly_panel.parquet`. Estimators run on raw Y, T with
   cross-fit residualisation on a confound set (`lag_revenue_1m`,
   `n_retailers_in_tehsil`, `n_growers`) plus the seven effect-modifier
   features.

| Estimator | Synthetic Spearman(τ̂,τ_true) | Real CF residualized window-Qini |
|---|---|---|
| T-learner | +0.975 | +0.133 (76% scale router) |
| S-learner | +0.017 (Ridge weakness, expected) | +0.015 |
| R-learner | **+0.864** (audited vs econml.LinearDML +0.83 agreement) | +0.165 |
| **Causal Forest** | **+0.859** | **+0.263** |

The R-learner is the cross-seed stability metric in `dgp_gate.py`
(r=0.888). Causal Forest is the L3 CATE source (best residualized Qini).

### §3.4 — L3: Deadline-Aware Route Optimiser

`disha/optimizer/router.py`. Greedy by prize-density per rep-day, taking
the highest `(CF_CATE × revenue_potential) / (service+travel)` cell each
iteration until the daily capacity (480 min) is exhausted. Window-open
is a hard pre-filter; travel time uses cached district centroids with a
haversine fallback. Dual-arm interface (`arm="real" | "synthetic"`)
emits both plans from the same solver/capacity/window logic.

The decision to use greedy (over CP-SAT) was a time-budget call: the
per-rep candidate set after window filtering is small (≤ ~10 cells/rep/
day), and greedy-by-density is provably within constants of optimal for
this sparse, low-coupling structure.

### §3.5 — L4: FastAPI

`disha/api/main.py`. Seven endpoints (`/health`, `/plan/{rep_id}`,
`/retailer/{id}/why`, `/territory/{id}/signals`, `/sim/run`,
`/business_case`, `/qini`). All endpoints serve precomputed frozen
artifacts; the API does no modeling. CORS open so the static rep-app
and dashboard can call it directly.

---

## §2 — Data Strategy

We work with eight raw CSVs (retailers, growers, reps, POS, inventory,
visits, WhatsApp, digital funnel) plus Open-Meteo weather cached per
district. The load-bearing modelling choice is the **grain**:
`(tehsil × month × product)`, not `(tehsil × week × product)`. Why:

- **Treatment is sparse weekly** (≤8% of cells treated/week) but
  reasonably dense monthly (~26%), giving the FE estimator enough
  within-entity variation to identify ATE.
- **Outcomes lag treatment** by 1–3 weeks; the monthly grain absorbs the
  lag without explicit lead-lag bookkeeping.
- **Weather + windows operate on a multi-day timescale**; monthly
  averages preserve the agronomic signal while smoothing out daily noise.

Three-tier fallback for the agronomic window (`tehsil → district →
default = wheat`) gives full coverage on all 2,849 tehsils despite 450
null grower calendars in the raw data.

**NDVI, competitor activity, and pest-surveillance bulletins are
modelled proxies in this prototype.** The crop-health field is a
derived proxy from disease pressure and weather context; the competitor
field is a derived proxy from non-focus SKU velocity at the outlet;
the pest/disease alert is a weather-driven proxy (disease pressure ×
agronomic window), not from external government pest-surveillance
bulletins. The architecture accepts external feeds via the same
connector with no model change — labelling the prototype's proxies
honestly turns a caveat into an architecture strength.

---

## §4 — Expected Impact

### §4.1 — Decision quality (Qini, the headline metric)

The window-constrained, scale-residualized Qini is our primary technical
metric (Causal Forest, **+0.263**). This says: among in-window cells in
real Syngenta data, the CATE ranking carries decision-quality
information that survives partialling out the four scale features.

A T-learner baseline on the same data gives an apparent +0.557 raw Qini,
but **76% of it disappears** under the same residualization — that's the
"send reps to bigger tehsils" baseline made explicit. Disha beats it
honestly.

> **Sourced from `uplift_eval.json::qini_window_residualized` —**
> computed by `disha.eval.residualized_qini`, served via `/qini`,
> rendered by the dashboard's Recommendation-acceptance and Qini-gate
> chips, test-pinned by `tests/test_headline_integrity.py` so the
> deck, doc, dashboard, and serialized number are provably equal.

### §4.2 — Return-per-field-day (operational lens)

OPE: Doubly-Robust value per cell, 300-bootstrap percentile CI.

| Policy | DR value / cell | 95% CI |
|---|---|---|
| BAU historical rotation | ₹1,590 | — |
| Disha (top-CATE in-window) | **₹2,948** | [₹2,771, ₹3,125] |
| **Disha lift** | **+₹1,358** | per cell |

Caveat: the absolute INR are biased downward by uncorrected selection in
the CF outcome model. We use these as ordinal sanity-check magnitudes,
not as point estimates of expected production lift. The HEADLINE for
technical judges remains the residualized Qini above.

### §4.3 — Bounded business case (for managers)

Across 500 reps × 6-month season ≈ 20,744 historical treatment
opportunities. Reallocating treatments from zero/negative-CATE cells to
the positive-CATE tail (under our OPE DR estimate, with the honesty
caveats below) is worth **₹24.5 M – ₹31.8 M per season at zero added
rep-day cost**.

Disclaimers (every one is stated in the demo aloud):

1. Real-data CATE absolute values are biased; we use CF for RANKING.
2. OPE is a sanity-check magnitude; production-quality value requires a
   geo-randomised rollout.
3. The number assumes Disha-routed visits substitute cleanly for BAU
   visits, with no transition cost or rep-acceptance friction.
4. **Outcome loop is instrumented, retraining is roadmap.** Visit
   outcomes are captured offline, synced, and drive the acceptance
   metric and a weekly recalibration trigger. Automatic CATE retraining
   on logged outcomes is the documented next step, not yet live — we
   refuse to claim the model adapts in the moment when it doesn't yet.

### §4.4 — Production validation methodology

We **do not** claim the OPE numbers translate 1:1 to production. The
correct production validation is a **geo-randomised rollout**: pick a
matched cluster of ~50 territories, randomly assign 25 to Disha and 25
to BAU rotation for a full quarter, compare actual revenue per
rep-quarter. With balanced design, this yields an unbiased ATE estimate
in 3 months — exactly the value the OPE was trying to approximate.

---

## §5 — Limitations (the honesty arc)

We are explicit about three live limitations:

1. **ATE ≈ 0 with CI straddling zero.** After proper FE identification,
   the population-average effect of rep visits is statistically
   indistinguishable from zero. Disha's value lives entirely in
   *heterogeneity* — a positive-CATE tail. If that heterogeneity didn't
   exist, Disha would have nothing to recommend.

2. **CATE has a 37.5% scale component.** Across the 4 scale features
   we tested (lag_revenue, n_retailers, n_growers, farm_size), 37.5% of
   real-data CATE variance is predictable from scale alone. Causal
   Forest's Qini survives partialling this out (+0.263 vs +0.155 raw),
   but the scale-aware framing is non-negotiable in the deck.

3. **Three corrections we caught on ourselves** (see §3.6). The fact
   that we caught them is the point — we treat the honesty arc as
   evidence of process discipline, not as a weakness.

**Future work (roadmap, not promises):**

- **Reinforcement learning** over multi-day routing trajectories with
  the budget-of-rep-quarters as the reward signal. Current optimiser is
  greedy-myopic per day; an RL agent could learn cross-day deferrals
  (e.g., "defer this visit by one day because tomorrow's window
  pressure is sharper").
- **Active calibration via micro-randomisation.** Treat ~5% of rep-days
  as randomised exploration to continuously re-estimate CATE under the
  current policy, closing the OPE-to-A/B gap without a full rollout.
- **Grower-level CATE** with consented data. Current grain is tehsil ×
  product; finer grain would tighten the agronomic-window precision.

---

## §6 — Deployment & Platform-Fit

**Stack (all chosen for platform-fit, not novelty):**

| Layer | Tech | Why |
|---|---|---|
| Compute | Python 3.13 + pandas + scikit-learn + econml + OR-Tools | Standard data-science; no GPU required |
| Storage | Parquet files in `data/processed/` | Portable; no DB to provision |
| API | FastAPI + uvicorn | Auto-docs, easy CORS, fits Syngenta's existing Python stack |
| Rep app | React + Vite + Tailwind + IndexedDB | Mobile-web works on any Android; sync-then-offline is exactly the field constraint |
| Dashboard | Same React project, second route | One codebase, two products; faster iteration |
| Compose | Docker Compose | `docker compose up` and judges have it running |

**Field-fit considerations:**

- **Mobile-web, not native.** Reps' phones are heterogeneous (entry
  Android dominant in tier-2/3 districts); a mobile-web app installs
  with one URL and updates without an app-store cycle.
- **Sync-then-offline.** The rep app caches the day's plan in IndexedDB
  on first connect; everything (read + outcome logging) works offline;
  outcomes auto-sync on reconnect. The rep never sees "no internet"
  failure modes.
- **WHY text is template-rendered.** No on-device LLM dependency; works
  with or without an LLM cache. Optional gated rephrase into local
  language (e.g., Hindi/Bengali) is a future improvement.
- **CRM/PoS integration is a 30-line connector.** The optimiser consumes
  three parquets (panel, CATE, signals); a Syngenta integration would
  swap our offline data builder for a connector reading from their
  existing Cropwise / Salesforce / SAP feeds, with zero changes to the
  optimiser, API, or apps.
- **NDVI, competitor activity, and pest-surveillance bulletins are
  proxies in this prototype;** the connector contract is feed-agnostic,
  so swapping in external satellite ingestion, an external competitor
  data feed, or government pest-surveillance-bulletin ingestion requires
  zero model change. Explicit labelling beats silent integration claims.

---

## §7 — Honesty arc (detail)

The three corrections, each pinned by an adversarial test in
`tests/test_independence.py`:

1. **`avg_visit_pressure` leakage.** A treatment-derived feature was
   inflating cross-seed CATE r by +0.25. Removed; r dropped 0.66 → 0.43.
2. **R-learner Stage-2 prediction bug.** Was returning `Y_tilde · T_tilde`
   at validation instead of `τ̂ = X·β`. Fixed; r corrected 0.43 → 0.88.
   Audited against `econml.LinearDML` cross-implementation (Spearman = +0.83).
3. **Signal scale-artifact misclassification.** `demand_spike` and
   `oos_opportunity` looked like CATE modifiers under raw ΔCATE but
   74–79% of their effect was firing preferentially in bigger tehsils
   where CATE is mechanically lower. Demoted to L4-text-only.

**Adversarial-test template** (used per-layer from L2 onward, and is the
template we recommend for any future causal layer added to Disha):

> 1. Characterise the estimator's bias on a known-truth synthetic.
> 2. Re-prove the substantive claim with a method that does NOT share
>    that bias.
> 3. Assert the two evidences give *different* numbers — otherwise
>    they aren't really independent and the claim is circular.

We treat this discipline as our single biggest differentiator. The
science is correct, and we know exactly what it can and cannot tell us.
