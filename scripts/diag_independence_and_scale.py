"""
Pre-L3 diagnostic: produce the evidence we need to either keep or rewrite
two claims that currently sit on shaky ground:

  CLAIM A — Window does not modify CATE
  ----------------------------------------
  Current (rejected) evidence: R-learner win_rho ≈ -0.16
  Problem: the R-learner produces win_rho ≈ -0.16 even when the true effect
  is zero (we showed this on the synthetic DGP).  So the number is consistent
  with both "true zero" and "true small negative" — it discriminates nothing.

  Replace with:
    (i)  Driver-magnitude ranking (already in dgp_gate top_drivers; window is
         dead last, |coef|=93 vs top=726).
    (ii) Model-free Spearman: within window-decay deciles, compute FE-demeaned
         (Y_dm, T_dm) ATE; Spearman across deciles vs decile window mean.
         No R-learner involved.

  CLAIM B — All five signals are NEGATIVE CATE effect-modifiers (economic)
  ----------------------------------------------------------------------
  Current evidence: ΔCATE = −₹300 to −₹1,500 across the four kept signals.
  Concern: this could be a CATE-scale artifact — if real-data CATE happens
  to be negatively correlated with baseline tehsil revenue / lag_revenue,
  then ANY signal that fires preferentially in big tehsils would mechanically
  show negative ΔCATE.  That would be an artifact, not an economic finding.

  Diagnose: Spearman(CATE_real, lag_revenue_1m / n_retailers / Y_revenue).
  If |r| is large and negative → scale artifact, rewrite L1 narrative.
  If |r| is near zero → economic finding holds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ──────────────────────────────────────────────────────────────────────────────
# Load inputs
# ──────────────────────────────────────────────────────────────────────────────
panel = pd.read_parquet("data/processed/monthly_panel.parquet")
cate_frozen = pd.read_parquet("data/processed/cate_frozen.parquet")

merged = panel.merge(
    cate_frozen[["tehsil", "month_start", "product", "cate_mean"]],
    on=["tehsil", "month_start", "product"],
    how="inner",
)
assert len(merged) == len(panel), "join lost rows — check keys"

# ──────────────────────────────────────────────────────────────────────────────
# A. Model-free window-CATE independence test
#    Approach: two-way FE demean Y and T, then within window-decay deciles
#    compute stratum ATE = Cov(Y_dm, T_dm) / Var(T_dm).
# ──────────────────────────────────────────────────────────────────────────────

def two_way_demean(values, entity_grp, time_grp, n_iter=20):
    s = pd.Series(np.asarray(values, dtype=float))
    eg = pd.Series(entity_grp)
    tg = pd.Series(time_grp)
    for _ in range(n_iter):
        s = s - s.groupby(eg).transform("mean")
        s = s - s.groupby(tg).transform("mean")
    return s.values

entity = (merged["tehsil"].astype(str) + "___" + merged["product"].astype(str)).values
time = merged["month_index"].values

Y_dm = two_way_demean(merged["Y_revenue"].astype(float).values, entity, time)
T_dm = two_way_demean(merged["T"].astype(float).values, entity, time)
window = merged["window_decay_this_product"].astype(float).values

# 76% of cells have window_decay == 0 (window closed).  Equal-width or rank
# deciles collapse them into one bin.  Use a two-level stratification:
#   stratum 0  : window_decay == 0 (closed)
#   stratum 1-4: quartiles within the window-open subset
# Then compute Spearman across the 5 ordered strata.

print("\n" + "=" * 78)
print("CLAIM A: window-CATE independence -- model-free FE-stratified test")
print("=" * 78)
print(f"window=0 cells: {int((window == 0).sum())} ({100*(window==0).mean():.1f}%)")
print(f"window>0 cells: {int((window > 0).sum())} ({100*(window>0).mean():.1f}%)")

def stratum_ate(mask):
    Y_d = Y_dm[mask]; T_d = T_dm[mask]
    var_T = T_d.var()
    if var_T == 0:
        return float("nan")
    cov = np.mean(Y_d * T_d) - Y_d.mean() * T_d.mean()
    return cov / var_T

strata = []
# Stratum 0: window closed
m0 = window == 0
strata.append(("closed", float(window[m0].mean() if m0.sum() else 0.0), m0))
# Strata 1-4: rank-quartiles within the open subset (handles discrete values)
open_mask = window > 0
open_rank = pd.Series(window[open_mask]).rank(method="first").values
n_open = open_mask.sum()
quart = (open_rank / (n_open + 1) * 4).astype(int).clip(0, 3)
for q in range(4):
    mk = np.zeros(len(window), dtype=bool)
    open_idx = np.where(open_mask)[0]
    mk[open_idx[quart == q]] = True
    if mk.sum() > 0:
        strata.append((f"open Q{q+1}", float(window[mk].mean()), mk))

strat_ates = []
strat_window_means = []
print(f"\n  {'stratum':<10s} {'n':>7s} {'window_mean':>12s} {'stratum_ATE':>14s}")
print("  " + "-" * 50)
for label, w_mean, mk in strata:
    ate_d = stratum_ate(mk)
    strat_ates.append(ate_d)
    strat_window_means.append(w_mean)
    print(f"  {label:<10s} {int(mk.sum()):>7d} {w_mean:>12.3f} {ate_d:>+14.0f}")

rho, p = spearmanr(strat_ates, strat_window_means)
print(f"\nSpearman(stratum_ATE, window_mean) across 5 strata = {rho:+.4f}  (p={p:.3f})")
print(f"|rho| {'<' if abs(rho) < 0.50 else '>='} 0.50 (5-point band) -- "
      f"{'INDEPENDENCE supported (model-free)' if abs(rho) < 0.50 else 'WINDOW MODIFIES CATE'}")

# Effect-size check: max stratum ATE - min stratum ATE.  If ~0, truly flat.
ate_range = max(strat_ates) - min(strat_ates)
print(f"Stratum-ATE range (max - min) = {ate_range:+.0f} INR/month")

# Compare to R-learner's win_rho (the biased one)
rho_rlearner, _ = spearmanr(merged["cate_mean"].values, window)
print(f"\n(Reference, NOT load-bearing): R-learner win_rho = {rho_rlearner:+.4f}")
print("  On synthetic DGP with beta_window=0, our R-learner gives ~ -0.16 -- known bias.")

# ──────────────────────────────────────────────────────────────────────────────
# B. Scale-artifact check: CATE vs baseline-wealth proxies
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("CLAIM B: signals-negative pattern — scale artifact or economic?")
print("=" * 78)

cate = merged["cate_mean"].astype(float).values

probes = {
    "lag_revenue_1m":         merged["lag_revenue_1m"].astype(float).values,
    "Y_revenue":              merged["Y_revenue"].astype(float).values,
    "n_retailers_in_tehsil":  merged["n_retailers_in_tehsil"].astype(float).values,
    "n_growers":              merged["n_growers"].astype(float).values,
    "avg_farm_size_ha":       merged["avg_farm_size_ha"].astype(float).values,
}

print(f"\n  {'probe':<25s} {'Spearman(CATE, probe)':>22s} {'verdict':>20s}")
print("  " + "-" * 70)
for name, vals in probes.items():
    rho_p, _ = spearmanr(cate, vals)
    artefact = "SCALE ARTIFACT" if abs(rho_p) > 0.40 else (
        "watch" if abs(rho_p) > 0.20 else "clean"
    )
    print(f"  {name:<25s} {rho_p:>+22.4f} {artefact:>20s}")

# Repeat for each signal-flagged subset to see if their negative ΔCATE is
# driven by the signal correlating with scale.
print("\nDirect check: do flagged cells systematically have higher baseline revenue?")
print(f"  {'signal':<25s} {'mean lag_rev (flag=1)':>22s} {'mean lag_rev (flag=0)':>22s} {'ratio':>8s}")
print("  " + "-" * 80)
signals = pd.read_parquet("data/processed/signals_panel.parquet")
sig_keys = ["tehsil", "month_start", "product"]
m2 = merged.merge(signals[sig_keys + [c for c in signals.columns if c.endswith("_flag")]],
                  on=sig_keys, how="inner")
for flag in ["demand_spike_flag", "oos_opportunity_flag",
             "disease_alert_flag", "window_urgency_flag",
             "digital_demand_flag"]:
    on = m2[m2[flag] == 1]["lag_revenue_1m"].mean()
    off = m2[m2[flag] == 0]["lag_revenue_1m"].mean()
    ratio = on / off if off else float("nan")
    name = flag.replace("_flag", "")
    print(f"  {name:<25s} {on:>22.0f} {off:>22.0f} {ratio:>8.2f}")

# ──────────────────────────────────────────────────────────────────────────────
# B'. THE KEY TEST: residualize CATE for lag_revenue + n_growers + farm_size,
#     then recompute signal ΔCATE on the residual.  If signals still show
#     negative ΔCATE after partialling out scale, the economic story holds.
#     If residual ΔCATE collapses to ~0, the original ΔCATE WAS a scale artifact.
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print("CLAIM B' (decisive): residualize CATE on scale, recompute signal deltas")
print("=" * 78)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

scale_cols = ["lag_revenue_1m", "n_retailers_in_tehsil", "n_growers", "avg_farm_size_ha"]
S = m2[scale_cols].astype(float).fillna(0.0).values
S = StandardScaler().fit_transform(S)
cate_orig = m2["cate_mean"].astype(float).values
mdl = Ridge(alpha=1.0).fit(S, cate_orig)
cate_resid = cate_orig - mdl.predict(S)
print(f"  Fraction of CATE variance explained by scale features = "
      f"{1 - cate_resid.var() / cate_orig.var():.3f}")

print(f"\n  {'signal':<25s} {'orig delta':>12s} {'resid delta':>12s} {'shrink':>10s}")
print("  " + "-" * 65)
for flag in ["demand_spike_flag", "oos_opportunity_flag",
             "disease_alert_flag", "window_urgency_flag",
             "digital_demand_flag"]:
    on_mask = m2[flag] == 1
    off_mask = m2[flag] == 0
    if on_mask.sum() == 0 or off_mask.sum() == 0:
        continue
    delta_orig = cate_orig[on_mask].mean() - cate_orig[off_mask].mean()
    delta_resid = cate_resid[on_mask].mean() - cate_resid[off_mask].mean()
    shrink = (1 - abs(delta_resid) / abs(delta_orig)) * 100 if delta_orig else 0
    name = flag.replace("_flag", "")
    print(f"  {name:<25s} {delta_orig:>+12.0f} {delta_resid:>+12.0f} {shrink:>9.0f}%")
print("\nIf 'shrink' > ~70% --> signal deltas were mostly scale-artifact")
print("If 'shrink' < ~30% --> economic effect holds after partialling out scale")

