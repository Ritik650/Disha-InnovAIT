# Disha (दिशा) — Deadline-Aware Field-Force Co-Pilot

**Syngenta AgriTech Hackathon 2026 · Track 2 · AI-Guided Field Force Intelligence**

> Every field-AI tool optimises *propensity* — who will buy anyway.
> Agriculture has a biological deadline. Disha optimises **causal uplift
> under a hard agronomic-window constraint**: rep visits during the open
> protection window where the marginal effect is highest, ignoring cells
> outside the window entirely.

---

## Quick start

### Option A — Linux / macOS (with `make`)

```bash
make install      # python + node deps              (~2 min, one-off)
make build        # rebuild ALL frozen artifacts    (~70 s, measured)
make demo         # API on :8000 + rep-app/dashboard on :5173
```

### Option B — Windows (no `make`) or anywhere with Docker

```bash
docker compose up --build
```

### Option C — manual (any OS)

```bash
pip install -e .
cd app/web && npm install --no-audit --no-fund && cd ../..
# Build artifacts (~70 s on a modern laptop, deterministic, fully offline
# after the first run thanks to cached weather in data/cache/weather/):
python -m disha.twin.build
python -m disha.signals.run
python -m disha.uplift.train
python -m disha.ope.evaluate
python -m disha.eval.business_case
python -m disha.optimizer.router
# Run:
python -m uvicorn disha.api.main:app --port 8000 &
cd app/web && npm run dev
```

Then open:
- **http://localhost:5173/** — manager dashboard + dual-arm simulator
- **http://localhost:5173/rep** — rep mobile-web app + "Demo Morning" guided flow
- **http://localhost:8000/docs** — FastAPI auto-docs

### Offline guarantee

After the first `make build` (which fetches Open-Meteo weather for 33
districts and caches them under `data/cache/weather/`), the entire stack
runs without network access. Subsequent rebuilds, the API, the
optimizer, and both web apps make zero outbound calls in the hot path.
Tested: disable network after first build → `make demo` still works end-to-end.

---

## What's in here

| Path | What |
|---|---|
| `disha/twin/` | L0 — Territory digital twin: weather, balance, FE+DML lift probe, cross-seed CATE stability |
| `disha/signals/` | L1 — Five detectors with scale-residualized role assignment |
| `disha/uplift/` | L2 — T-/S-/R-learner + Causal Forest, pre-registered synthetic DGP |
| `disha/eval/` | Qini (incl. window-constrained) + bounded business case |
| `disha/ope/` | L5 — IPS + Doubly-Robust policy value (sanity check) |
| `disha/optimizer/` | L3 — dual-arm deadline-aware router (window = hard constraint) |
| `disha/api/` | L4 — 7-endpoint FastAPI serving precomputed artifacts |
| `app/web/` | Single React + Vite + Tailwind app with `/` dashboard and `/rep` mobile app |
| `tests/` | 130+ pytest cases including the adversarial-test suite |
| `docs/` | `SOLUTION.md` (≤10 pp), `PROGRESS.md` (audit trail), `VIDEO_SCRIPT.md` |
| `scripts/` | One-off diagnostics (R-learner audit, scale residualization, etc.) |

---

## The headline numbers (frozen)

> Every number in this table is computed by the build pipeline and
> serialized to `data/processed/uplift_eval.json` (Qini headlines) or
> `business_case.json` (rupee band). The residualized Qini is sourced
> from `uplift_eval.json::qini_window_residualized.causal_forest.resid`
> via `disha.eval.residualized_qini`, test-pinned by
> `tests/test_headline_integrity.py`.

| | |
|---|---|
| Demo path | `dual_synthetic_led` |
| Synthetic engine recovery | R-learner Spearman(τ̂, τ_true) = **+0.864** (audited vs `econml.LinearDML` at +0.83 cross-impl agreement) |
| Real CATE cross-seed | R-learner Spearman = **+0.888** |
| Real headline metric | Causal-Forest window-constrained, **scale-residualized** Qini = **+0.263** |
| T-learner counterfactual | Apparent Qini +0.557, but 76% shrink under residualization → "naive = scale routing" baseline |
| Bounded business case | **₹24.5 M – ₹31.8 M / season**, same reps, same days, no added cost (CI from 300-boot DR) |

---

## Deck sentence (verbatim, never +0.263 standalone)

> Our method recovers known truth at r ≈ 0.86 on a controlled DGP. On real
> Syngenta data the honest economic signal is a +0.26 window-constrained
> Qini after stripping scale — modest but provably not the scale-routing
> artifact a naive T-learner would produce.

---

## Honesty arc (our single biggest differentiator)

Three corrections we caught on ourselves during the build, pinned by
adversarial tests in `tests/test_independence.py`:

1. **`avg_visit_pressure` leakage** — a treatment-derived feature inflated
   cross-seed CATE r by +0.25. Removed; r dropped 0.66 → 0.43.
2. **R-learner Stage-2 prediction bug** — was returning `Y_tilde` at
   validation instead of `τ̂ = X·β`. Fixed; true r is **0.88**, audited
   vs `econml.LinearDML` cross-implementation at +0.83 Spearman.
3. **Signal scale-artifact misclassification** — `demand_spike` and
   `oos_opportunity` looked like CATE modifiers under raw Δ but 74–79%
   of the effect was the signal firing in bigger tehsils. Demoted to
   L4-reason-text-only.

The adversarial-test template (used per-layer from L2 onward):

1. Characterise the estimator's bias on a known-truth synthetic.
2. Re-prove the substantive claim with a method that does NOT share that bias.
3. Assert the two evidences give *different* numbers — else the framing is circular.

---

## Repo links + team

- **Team:** _TBD — to be filled in by team lead before submission._
- **Members + contact:** _TBD._
- **Demo deploy:** _TBD (Vercel/Render link if hosted)._
- **Source archive:** `disha.zip` (this repo, packaged).

---

## Tests

```bash
make test
```

130+ pytest cases. Highlights:
- `test_uplift.py::TestSyntheticRecovery` — pre-registered τ_true recovery, per estimator
- `test_independence.py::TestWindowIndependenceAdversarial` — model-free + bias-band defense
- `test_independence.py::TestSignalScaleArtifactAdversarial` — per-signal residualization landscape
- `test_independence.py::TestQiniScaleArtifactAdversarial` — L2 Qini doesn't collapse on scale removal
- `test_optimizer.py` — dual-arm feasibility (window + capacity + territory)
- `test_api.py` — 3 API contract tests

---

## License

For the Syngenta AgriTech Hackathon 2026; rights as specified by the
competition terms.
