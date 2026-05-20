"""
disha.eval.qini — Qini curve and window-constrained Qini under the locked
"uplift ⊥ window, combined in optimizer" architecture.

Qini curve
----------
The Qini curve plots the cumulative uplift gained by treating the top-k%
units ranked by estimated CATE.  For a binary treatment with realized Y, T,
the empirical Qini at rank k (over n total units, n_t treated, n_c control) is:

    Q(k) = Σ_{i ∈ top-k, T=1} Y_i  −  (n_t(k) / n_c(k)) · Σ_{i ∈ top-k, T=0} Y_i

where n_t(k), n_c(k) are the treated/control counts in the top-k.  The
Qini coefficient is the area between Q(k) and the random-targeting line,
divided by the area of the perfect-targeting curve.  Larger = better
discrimination.

Window-constrained Qini
-----------------------
Because the L3 architecture enforces the window as a hard feasibility
constraint, the practically achievable Qini is computed ONLY over cells
where window_open == 1 (or window_decay > 0).  Cells outside the window
are infeasible — no rep can convert them — so including them in the
ranking pretends we can route to impossible targets.

We report BOTH:
  qini_all          — unconstrained ranking, all 80,128 cells
  qini_window_only  — ranking restricted to in-window cells (the real-world
                      decision quality under the locked optimizer architecture)

For synthetic the gap should be near zero (CATE ⊥ window by construction);
for real it shows how much potential is filtered out by the constraint.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class QiniResult:
    qini_coefficient: float       # in [-1, +1]; larger = better
    auuc: float                    # area under uplift curve, raw units
    curve_x: np.ndarray            # fraction of population (0..1)
    curve_y: np.ndarray            # cumulative uplift at each x
    random_line_y: np.ndarray      # the diagonal (random targeting)
    n: int
    n_treated: int
    n_control: int


def qini_curve(
    cate: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    n_points: int = 200,
) -> QiniResult:
    """
    Compute the Qini curve by ranking units in descending CATE order and
    summing the IPW-corrected treated-minus-control outcome difference.

    Parameters
    ----------
    cate : (n,) estimated CATE per unit
    y    : (n,) realized outcome
    t    : (n,) binary treatment indicator
    n_points : number of points to sample along the curve
    """
    cate = np.asarray(cate, dtype=float)
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=int)
    n = len(cate)
    if not (len(y) == n and len(t) == n):
        raise ValueError("cate, y, t must have the same length")

    # Sort by CATE descending
    order = np.argsort(-cate)
    y_s = y[order]
    t_s = t[order]

    cum_t_y = np.cumsum(y_s * t_s)
    cum_c_y = np.cumsum(y_s * (1 - t_s))
    cum_t_n = np.cumsum(t_s).astype(float)
    cum_c_n = np.cumsum(1 - t_s).astype(float)

    # IPW-style cumulative uplift: treated sum minus (n_t/n_c)-scaled control sum
    # Avoid /0 at the very start
    safe_c_n = np.where(cum_c_n > 0, cum_c_n, 1.0)
    cum_uplift = cum_t_y - (cum_t_n / safe_c_n) * cum_c_y

    # Subsample to n_points for plotting
    idx = np.linspace(0, n - 1, min(n_points, n)).astype(int)
    curve_x = (idx + 1) / n
    curve_y = cum_uplift[idx]

    # Random-targeting baseline = linear from 0 to total uplift (treated-control diff)
    n_t = int(t.sum())
    n_c = int(n - n_t)
    if n_t > 0 and n_c > 0:
        total_uplift = (y[t == 1].sum()) - (n_t / n_c) * (y[t == 0].sum())
    else:
        total_uplift = 0.0
    random_line_y = curve_x * total_uplift

    # Qini coefficient (signed area between curve and random line, normalized
    # by area between perfect and random).  Using trapezoidal integration.
    auuc = float(np.trapezoid(curve_y, curve_x))
    area_random = float(np.trapezoid(random_line_y, curve_x))
    # Perfect targeting: treat all positive-CATE units first.  Approximate by
    # sorting realized (treated y - matched control y) descending — we use
    # |total_uplift|/2 as the perfect-area normalizer, which gives a Qini in
    # roughly [-1, +1] without requiring the matched-pair imputation.
    norm = max(abs(total_uplift) / 2.0, 1e-9)
    qini_coef = (auuc - area_random) / norm

    return QiniResult(
        qini_coefficient=float(qini_coef),
        auuc=auuc,
        curve_x=curve_x,
        curve_y=curve_y,
        random_line_y=random_line_y,
        n=n,
        n_treated=n_t,
        n_control=n_c,
    )


def qini_window_constrained(
    cate: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    window_open: np.ndarray,
    n_points: int = 200,
) -> QiniResult:
    """
    Compute Qini restricted to feasibility-permitted cells (window_open == 1).

    Out-of-window cells are excluded entirely — they cannot be routed under
    the locked L3 architecture (window is a hard constraint).
    """
    mask = np.asarray(window_open).astype(bool)
    if mask.sum() == 0:
        raise ValueError("No window-open cells; cannot compute constrained Qini")
    return qini_curve(
        cate=np.asarray(cate)[mask],
        y=np.asarray(y)[mask],
        t=np.asarray(t)[mask],
        n_points=n_points,
    )


def evaluate_cate_recovery(
    cate_estimated: np.ndarray,
    cate_true: np.ndarray,
) -> dict:
    """
    Synthetic-only recovery metrics (requires known ground truth).

    Returns Spearman rank correlation, mean ATE bias, and per-row MSE.
    """
    from scipy.stats import spearmanr

    cate_estimated = np.asarray(cate_estimated, dtype=float)
    cate_true = np.asarray(cate_true, dtype=float)
    rho, _ = spearmanr(cate_estimated, cate_true)
    ate_hat = float(cate_estimated.mean())
    ate_true = float(cate_true.mean())
    return {
        "spearman_r": float(rho) if rho is not None and not np.isnan(rho) else 0.0,
        "ate_hat": ate_hat,
        "ate_true": ate_true,
        "ate_bias": ate_hat - ate_true,
        "mse": float(np.mean((cate_estimated - cate_true) ** 2)),
    }
