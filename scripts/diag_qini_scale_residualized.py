"""
L2 adversarial diagnostic: does the window-constrained Qini lift survive
when we partial scale features out of the CATE used for ranking?

Analogue of the signal scale-artifact check.  Same logic:
  Original Qini answers: "how well does CATE rank cells for treatment?"
  Residualized Qini answers: "how well does CATE rank cells AFTER we strip
                              out what scale (lag_rev, n_growers,
                              n_retailers, farm_size) was already telling us?"

If residualized Qini ~ 0 -> the CATE ranking was mostly a scale ranking;
  Disha could be replaced by "send reps to bigger tehsils" with no L2.
If residualized Qini survives close to original -> CATE-targeting has
  economic value over and above scale-based routing.

We use the QINI_WINDOW_ONLY variant (in-window cells only) because that
matches the locked L3 architecture and is the headline metric for the deck.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from disha.eval.qini import qini_curve, qini_window_constrained

SCALE_FEATURES = [
    "lag_revenue_1m",
    "n_retailers_in_tehsil",
    "n_growers",
    "avg_farm_size_ha",
]

real_cate = pd.read_parquet("data/processed/uplift_real_cate.parquet")
panel = pd.read_parquet("data/processed/monthly_panel.parquet")

keys = ["tehsil", "month_start", "product"]
m = real_cate.merge(panel[keys + SCALE_FEATURES], on=keys, how="inner")
assert len(m) == len(real_cate), "join lost rows"

S = StandardScaler().fit_transform(m[SCALE_FEATURES].astype(float).fillna(0.0).values)
window_open = (m["window_decay_this_product"].astype(float).values > 0).astype(int)
Y = m["Y_revenue"].astype(float).values
T = m["T"].astype(int).values

print("=" * 78)
print("L2 ADVERSARIAL DIAGNOSTIC")
print("  window-constrained Qini under scale residualization")
print("=" * 78)
print(f"  n cells total       = {len(m)}")
print(f"  n cells in-window   = {window_open.sum()} ({100*window_open.mean():.1f}%)")
print(f"  scale features      = {SCALE_FEATURES}")
print()

results = []
for est in ["t_learner", "s_learner", "r_learner", "causal_forest"]:
    col = f"cate_{est}"
    if col not in m.columns:
        continue
    cate = m[col].astype(float).values
    if not np.isfinite(cate).all():
        print(f"{est}: contains non-finite values, skipping")
        continue

    # Variance explained by scale features
    var_orig = cate.var()
    mdl = Ridge(alpha=1.0).fit(S, cate)
    cate_resid = cate - mdl.predict(S)
    var_exp = 1.0 - cate_resid.var() / var_orig if var_orig > 0 else 0.0

    q_orig_all = qini_curve(cate, Y, T)
    q_orig_win = qini_window_constrained(cate, Y, T, window_open)
    q_resid_all = qini_curve(cate_resid, Y, T)
    q_resid_win = qini_window_constrained(cate_resid, Y, T, window_open)

    def shrink(orig, resid):
        return (1.0 - abs(resid) / abs(orig)) * 100 if abs(orig) > 1e-9 else 0.0

    results.append({
        "estimator": est,
        "scale_var_explained": var_exp,
        "qini_all_orig": q_orig_all.qini_coefficient,
        "qini_all_resid": q_resid_all.qini_coefficient,
        "qini_all_shrink": shrink(q_orig_all.qini_coefficient, q_resid_all.qini_coefficient),
        "qini_win_orig": q_orig_win.qini_coefficient,
        "qini_win_resid": q_resid_win.qini_coefficient,
        "qini_win_shrink": shrink(q_orig_win.qini_coefficient, q_resid_win.qini_coefficient),
    })

print(f"{'estimator':<16s}  {'scale_var%':>10s}  "
      f"{'qini_win_orig':>14s} {'qini_win_resid':>15s} {'shrink':>8s}")
print("-" * 78)
for r in results:
    print(f"{r['estimator']:<16s}  {r['scale_var_explained']*100:>9.1f}%  "
          f"{r['qini_win_orig']:>+14.4f} {r['qini_win_resid']:>+15.4f} "
          f"{r['qini_win_shrink']:>7.0f}%")

print()
print(f"{'estimator':<16s}  "
      f"{'qini_all_orig':>14s} {'qini_all_resid':>15s} {'shrink':>8s}")
print("-" * 60)
for r in results:
    print(f"{r['estimator']:<16s}  "
          f"{r['qini_all_orig']:>+14.4f} {r['qini_all_resid']:>+15.4f} "
          f"{r['qini_all_shrink']:>7.0f}%")

print()
print("Interpretation guide (window_only is the headline metric):")
print("  shrink <= 30%  ->  CATE-targeting holds economically after scale removed")
print("  shrink >= 70%  ->  CATE-targeting was mostly scale routing -> L2 problem")
print("  in between     ->  mixed; CATE has value but scale is a meaningful component")
