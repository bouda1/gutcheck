# gutcheck - distributed-lag model and stability selection
# Copyright (C) 2026 The gutcheck authors (see the AUTHORS file)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Distributed-lag model + stability selection.

Principle
---------
The pain observed at time t is the sum of the contributions of ALL past
meals. Each food acts with its own lag profile, described by a smooth
kernel basis (raised-cosine) covering 0-84 h.

All foods are estimated SIMULTANEOUSLY (so several culprits are possible,
and each is assessed "holding competing foods constant"), subject to two
constraints:
  - positivity: we're looking for triggers, not protectors;
  - sparsity (lasso): few foods are actually responsible.

The question "which ones?" is not settled by a p-value (invalid here:
multiple testing, lag chosen post hoc, autocorrelated pain) but by the
SELECTION FREQUENCY under block resampling by days
(stability selection, Meinshausen & Bühlmann 2010).
"""

import os
import unicodedata
from collections import defaultdict

import numpy as np

from i18n import _

# Lag basis: raised-cosine kernels, 0-84 h. Centers and half-widths in hours.
LAG_CENTERS = np.array([2.0, 5.0, 9.0, 15.0, 24.0, 40.0, 60.0])
LAG_WIDTHS = np.array([3.0, 4.0, 5.0, 8.0, 12.0, 18.0, 24.0])
MIN_DELAY = 0.25          # A meal never acts on a simultaneous pain
MAX_SPAN = float(LAG_CENTERS[-1] + LAG_WIDTHS[-1])

# Selection by stability (Meinshausen & Bühlmann 2010)
DAY_BLOCK_SIZE = 3     # contiguous blocks: preserves the autocorrelation
SUBSAMPLE_FRACTION = 0.5
N_REPLICATES = 200
N_LAMBDAS = 6
STABILITY_THRESHOLD = 0.70

MIN_OCCURRENCES = 3       # minimum number of meals containing the food
JACCARD_THRESHOLD = 0.85      # above: inseparable foods, merged


def normalize(name):
    """ Normalize a food name to a canonical form: lowercase, no accents, unified separators.

    Examples:
        >>> normalize("Chocolate bread")
        'chocolate_bread'
        >>> normalize("Crème brûlée")
        'creme_brulee'
        >>> normalize("Fruit salad - fresh")
        'fruit_salad_fresh'

    Args:
        name (str): The food name to normalize.

    Returns:
        str: The normalized food name.
    """
    name = unicodedata.normalize("NFKD", name.strip().lower())
    name = "".join(c for c in name if not unicodedata.combining(c))
    return "_".join(name.replace("-", " ").replace("_", " ").split())


# ══════════════════════════════════════════════════════════════════
#  Exposure matrix
# ══════════════════════════════════════════════════════════════════

def kernels_weights(delays):
    """
    This function creates a matrix.

    Each column corresponds to a raised-cosine kernel centered at a specific lag,
    in other words, each column represents a function almost equal to 0 everywhere
    except around its center lag, where it takes values between 0 and 1, with
    smooth transitions, exactly equal to 1 at the center lag and equal to 0 at
    the edges of the kernel width. It looks like a dome as a cosine function
    restricted to [-pi, pi] and shifted to be centered at the lag center.

    Each row corresponds to a specific delay (in hours) between a meal and an
    observation of pain. The value in each cell is the weight of the corresponding
    kernel for that delay, which is 0 if the delay is outside the kernel width,
    and smoothly varies between 0 and 1 within the kernel width.

    The resulting matrix has shape (n, K), where n is the number of delays and K
    is the number of kernels.

    Args:
        delays (array-like): An array of shape (n,) containing the delays in
        hours between meals and pain observations.

    Returns:
        np.ndarray: A 2D array of shape (n, K) containing the weights of the
        raised-cosine kernels for each delay.
    """
    # Conversion of delays to a float array and transposition in column.
    d = np.asarray(delays, dtype=float)[:, None]

    # Computation of the dome functions for each delay. These functions are named 'weights'.
    u = (d - LAG_CENTERS[None, :]) / LAG_WIDTHS[None, :]
    w = 0.5 * (1.0 + np.cos(np.pi * u))
    w[np.abs(u) >= 1.0] = 0.0
    w[(d < MIN_DELAY).ravel(), :] = 0.0
    return w

def build_exposition(t_obs, meal, foods):
    """Build the design matrix for the distributed-lag model.

    For each pain observation, accumulates the contribution of every past
    meal, per food and per lag kernel: contributions from different meals
    of the same food that overlap in time for a given observation are
    summed (a food's total exposure at time t is the sum over all meals
    containing it).

    Args:
        t_obs: (n,) array-like of pain observation times, in hours.
        meal: List of (meal_time_hours, [normalized foods]) tuples
            describing the meal history.
        foods: Ordered list of foods retained for the model.

    Returns:
        (n, F*K) ndarray design matrix, where F is the number of foods
        and K the number of lag kernels (len(LAG_CENTERS)). Column
        block (f, k) — i.e. column f * K + k — holds, for each
        observation, the cumulative dose of food f as seen through lag
        kernel k.
    """
    n, F, K = len(t_obs), len(foods), len(LAG_CENTERS)
    idx = {a: i for i, a in enumerate(foods)}
    X = np.zeros((n, F * K))
    t_obs = np.asarray(t_obs, dtype=float)

    for meal_time_hours, norm_foods in meal:
        targets = [i for i, a in enumerate(norm_foods) if a in idx]
        if not targets:
            continue
		# Subtract the meal_time_hours from each observation time of t_obs.
        delays = t_obs - meal_time_hours
		# We check that the delays are within the range of MIN_DELAY and MAX_SPAN,
        # and we only keep the relevant ones.
        relevant = (delays >= MIN_DELAY) & (delays <= MAX_SPAN)
        if not relevant.any():
            continue
        w = kernels_weights(delays[relevant])          # (m, K)
        lines = np.flatnonzero(relevant)
        for i in targets:
            j = idx[norm_foods[i]] * K
            X[np.ix_(lines, np.arange(j, j + K))] += w
    return X


import subprocess
import tempfile

def plot_exposition_gnuplot(X, t_obs, foods, K, lag_centers,
                             columns=None, outfile=None, title=None,
                             xlabel="Time (h)", ylabel="Cumulative dose",
                             terminal="qt"):
    """
    Plot one or several columns of the matrix returned by
    build_exposition, using gnuplot.

    Args:
        X: (n, F*K) design matrix (output of build_exposition).
        t_obs: (n,) observation times (x axis).
        foods: ordered list of foods (same order as build_exposition).
        K: number of lag kernels (len(LAG_CENTERS)).
        lag_centers: kernel centers (LAG_CENTERS), used for the legend.
        columns: list of series to plot. Each item may be:
            - an int: direct column index into X
            - (food, k): food + kernel index (0 <= k < K)
            - (food, None): sum over every kernel for that food
              (total exposure, all lags together)
            If None: plot the total (all lags) of every food in `foods`.
        outfile: image path (.png/.svg/.pdf); if None, an interactive window.
        title, xlabel, ylabel: graph decoration.
        terminal: interactive gnuplot terminal (qt, wxt, x11...).

    Returns:
        The path of the produced file (if outfile is given) or None.
    """
    t_obs = np.asarray(t_obs, dtype=float)
    idx = {a: i for i, a in enumerate(foods)}

    if columns is None:
        columns = [(a, None) for a in foods]

    series = []  # (label, values (n,))
    for item in columns:
        if isinstance(item, int):
            values = X[:, item]
            label = f"col{item}"
        else:
            food, k = item
            if food not in idx:
                raise ValueError(f"unknown food: {food}")
            base = idx[food] * K
            if k is None:
                values = X[:, base:base + K].sum(axis=1)
                label = f"{food} (total)"
            else:
                if not (0 <= k < K):
                    raise ValueError(f"invalid kernel index: {k}")
                values = X[:, base + k]
                lag_lbl = lag_centers[k] if lag_centers is not None else k
                label = f"{food} (lag={lag_lbl})"
        series.append((label, values))

    # Sort by time for a clean plot
    order = np.argsort(t_obs)
    t_sorted = t_obs[order]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
        data_path = f.name
        f.write("# t\t" + "\t".join(f'"{lbl}"' for lbl, _ in series) + "\n")
        for row_i, ti in zip(order, t_sorted):
            vals = "\t".join(f"{v[row_i]:.6g}" for _, v in series)
            f.write(f"{ti:.6g}\t{vals}\n")

    plot_terms = [
        f'"{data_path}" using 1:{i + 2} with lines lw 2 title "{label}"'
        for i, (label, _) in enumerate(series)
    ]

    script_lines = [
        f'set xlabel "{xlabel}"',
        f'set ylabel "{ylabel}"',
        "set grid",
        "set key outside right",
    ]
    if title:
        script_lines.append(f'set title "{title}"')

    if outfile:
        ext = os.path.splitext(outfile)[1].lstrip(".").lower()
        gp_term = {"png": "pngcairo", "svg": "svg", "pdf": "pdfcairo"}.get(ext, "pngcairo")
        script_lines.append(f"set terminal {gp_term} size 1000,600")
        script_lines.append(f'set output "{outfile}"')
    else:
        script_lines.append(f"set terminal {terminal} persist")

    script_lines.append("plot " + ", \\\n     ".join(plot_terms))
    script = "\n".join(script_lines) + "\n"

    subprocess.run(["gnuplot"], input=script, text=True, check=True)

    if outfile:
        os.remove(data_path)
        return outfile
    return None

def build_controls(t_obs, time_of_day):
    """Build the unpenalized confounder matrix Z (24h and 12h circadian
    harmonics, an intercept, and a linear time drift).

    Meant to be used as the control block Z in FWL-style residualization
    (see `residualize`) before fitting a regularized model on the food
    exposure columns: these confounders are absorbed via plain least
    squares, unpenalized, rather than shrunk by Lasso alongside the food
    coefficients.

    Included confounders:
      - intercept: overall baseline level.
      - circadian rhythm (24h and 12h harmonics): without this, a food
        eaten every morning could pick up variance from the evening
        pain peak, since its exposure curve would itself vary with
        time of day. The 24h harmonic (sin/cos at 1 cycle/day) can
        represent any single daily peak; the 12h harmonic (sin/cos at
        2 cycles/day) additionally captures a twice-daily pattern
        (e.g. morning and evening peaks) that the 24h term alone
        cannot represent.
      - linear drift over the diary's duration: captures slow trends
        in the outcome unrelated to diet (e.g. gradual improvement or
        worsening over the tracked period).

    Note: if a given food is eaten at a very regular time of day, its
    exposure column may be substantially correlated with these
    circadian harmonics, and residualizing against Z can attenuate
    part of that food's true effect. This is an unavoidable
    identifiability trade-off, not a bug: without these controls, the
    model risks the opposite error of attributing circadian noise to
    diet.

    Args:
        t_obs: (n,) array-like of observation times since the start of
            the diary, in hours.
        time_of_day: (n,) array-like of the time of day for each
            observation, in hours (0-24).

    Returns:
        (n, 6) ndarray Z, with columns:
          0: intercept (all ones)
          1-2: sin/cos of the 24h circadian harmonic
          3-4: sin/cos of the 12h circadian harmonic
          5: linear drift (t_obs centered and scaled to days)
    """
    t = np.asarray(t_obs, dtype=float)
    h = np.asarray(time_of_day, dtype=float)
    days = (t - t.mean()) / 24.0
    return np.column_stack([
        np.ones_like(t),
        np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
        np.sin(4 * np.pi * h / 24), np.cos(4 * np.pi * h / 24),
        days,
    ])


def whiten_ar1(y, X, Z, t_obs):
    """
    Cochrane-Orcutt transformation: pain is autocorrelated from one meal to
    the next. Without whitening, the model "explains" that inertia by
    attributing the slow drift of the pain level to innocent foods.
    The coefficient is damped as a power of the actual time gap, since the
    observations are not regularly spaced. Assumes `t_obs` is sorted.
    → (y, X, Z, rho) — the first observation is consumed by the transformation.
    """
    r = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    rho = float(np.clip((r[:-1] @ r[1:]) /
                max(r[:-1] @ r[:-1], 1e-9), 0.0, 0.9))
    dt = np.diff(np.asarray(t_obs, dtype=float))
    w = rho ** (dt / max(np.median(dt), 1e-6))

    def tr(M):
        M = np.asarray(M)
        return M[1:] - w[:, None] * M[:-1] if M.ndim == 2 else M[1:] - w * M[:-1]

    return tr(y), tr(X), tr(Z), rho


def residualize(y, X, Z):
    """Residualize y and X with respect to Z (Frisch-Waugh-Lovell theorem).

    Projects y and each column of X orthogonally onto span(Z), and
    returns the residuals — i.e. y and X with the component explained
    by Z removed. By construction, y_res and every column of X_res are
    orthogonal to span(Z).

    By the Frisch-Waugh-Lovell theorem, the coefficients obtained by
    regressing y_res on X_res are exactly the coefficients on X in the
    full regression of y on [X, Z] — the coefficients on Z are not
    recovered by this function.

    Args:
        y: (n,) array-like of the response variable.
        X: (n, p) array-like of predictor variables to residualize
            (i.e. partial out Z from).
        Z: (n, q) array-like of control variables to partial out.
            Does not need to be full column rank; the projection onto
            span(Z) is still well-defined via least squares.

    Returns:
        y_res: (n,) ndarray, residuals of y after projecting onto
            span(Z).
        X_res: (n, p) ndarray, residuals of X (column-wise) after
            projecting onto span(Z).
    """

    # Stack y and X so both are projected in a single lstsq call.
    S = np.column_stack([y, X])
    coef, *_ = np.linalg.lstsq(Z, S, rcond=None)
    # Residual = original − fitted (i.e. original − its projection onto span(Z)).
    res = S - Z @ coef
    return res[:, 0], res[:, 1:]


# ══════════════════════════════════════════════════════════════════
#  Group lasso with non-negative coefficients
# ══════════════════════════════════════════════════════════════════
#  min_{beta >= 0}  1/(2n)||y - X beta||^2 + lam * sum_g sqrt(K) * ||beta_g||_2
#
#  A GROUP = one food, that is, its K lag coefficients. They enter or leave
#  the model together: a real effect spreads over several adjacent kernels,
#  and a plain lasso would pay the penalty on each of them separately —
#  hence a loss of power.
#
#  Solved by block descent on the Gram matrix (glmnet's "covariance updates"
#  trick), with the per-group subproblem handled by FISTA. The proximal
#  operator of  lam||.||_2 + indicator(. >= 0)  is the group thresholding
#  applied to the positive part.

def _prox_group(v, threshold):
    v = np.maximum(v, 0.0)
    n = np.sqrt(v @ v)
    return v * max(0.0, 1.0 - threshold / n) if n > 0 else v


def _solve_group(Gg, u, lam_g, b0, pas, n_iter=80, tol=1e-9):
    """min_{b>=0} 1/2 b'Gg b - u'b + lam_g||b||_2, solved by FISTA."""
    b = b0.copy()
    z = b.copy()
    t = 1.0
    for _it in range(n_iter):
        b_new = _prox_group(z - pas * (Gg @ z - u), pas * lam_g)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = b_new + ((t - 1.0) / t_new) * (b_new - b)
        done = np.abs(b_new - b).max() < tol
        b, t = b_new, t_new
        if done:
            break
    return b


def nn_group_lasso(G, c, groups, lam, weights, group_blocks, beta=None,
                    max_iter=150, tol=1e-6):
    """Fit a non-negative group Lasso via block coordinate descent.

    Solves, for beta >= 0 group-wise:
        min_beta (1/2n) ||y - X beta||^2
                  + lam * sum_g weights[g] * ||beta_g||_2

    Each group is updated in turn using its precomputed Gram
    submatrix and step size (see `prepare_groups`); a group is set
    entirely to zero when its gradient fails the group-wise KKT
    threshold test, otherwise it is solved via `_solve_group`.
    G @ beta is maintained incrementally across updates rather than
    recomputed from scratch each iteration.

    Args:
        G: (p, p) ndarray, Gram matrix X.T @ X / n.
        c: (p,) ndarray, X.T @ y / n.
        groups: List of 1-D index arrays, one per group (e.g. one
            array of lag-kernel columns per food).
        lam: float, overall regularization strength.
        weights: (len(groups),) array-like of per-group penalty
            weights, multiplying `lam` for each group.
        group_blocks: List of (Gg, step) tuples as returned by
            `prepare_groups`, in the same order as `groups`.
        beta: (p,) initial coefficient vector. Defaults to all zeros.
            If provided, it is copied, not modified in place.
        max_iter: int, maximum number of full passes over all groups.
        tol: float, convergence tolerance on the largest coefficient
            change (across all groups) within a pass.

    Returns:
        (p,) ndarray, the fitted non-negative coefficient vector.
    """
    p = len(c)
    beta = np.zeros(p) if beta is None else beta.copy()
    Gbeta = G @ beta

    for _it in range(max_iter):
        delta_max = 0.0
        for gi, idx in enumerate(groups):
            Gg, step = group_blocks[gi]
            bg = beta[idx]
            # X_g'(y - X beta_{-g}) / n
            u = c[idx] - Gbeta[idx] + Gg @ bg
            lam_g = lam * weights[gi]
            if np.linalg.norm(np.maximum(u, 0.0)) <= lam_g:
                new = np.zeros_like(bg)
            else:
                new = _solve_group(Gg, u, lam_g, bg, step)
            d = new - bg
            dmax = np.abs(d).max()
            if dmax > 0.0:
                beta[idx] = new
                Gbeta += G[:, idx] @ d
                delta_max = max(delta_max, dmax)
        if delta_max < tol:
            break
    return beta


def prepare_groups(G, groups):
    """Precompute per-group Gram submatrices and gradient step sizes.

    For block coordinate descent (e.g. group Lasso), each group of
    variables is updated using its own local Gram submatrix and a
    step size safe for gradient (or proximal-gradient) updates
    restricted to that group. The step size is set to 1/L, where L is
    the largest eigenvalue of the group's Gram submatrix — the
    Lipschitz constant of the gradient restricted to that group.

    Args:
        G: (p, p) ndarray, the full Gram matrix (e.g. X.T @ X).
        groups: List of 1-D index arrays, each listing the columns of
            G belonging to one group. Groups need not be the same
            size and may overlap or leave columns unassigned.

    Returns:
        List of (Gg, step) tuples, one per group, in the same order
        as `groups`:
          Gg: (k, k) ndarray, the Gram submatrix restricted to that
              group's k columns.
          step: float, the gradient step size 1/L for that group.
              Set to 0.0 if L is numerically negligible (<= 1e-12),
              e.g. for a group of all-zero or fully collinear columns.
    """
    ready = []
    for idx in groups:
        Gg = G[np.ix_(idx, idx)]
        L = float(np.linalg.eigvalsh(Gg)[-1])
        ready.append((Gg, 1.0 / L if L > 1e-12 else 0.0))
    return ready


def lambda_max(c, groups, weights):
    """Smallest penalty that zeroes out the whole solution."""
    vals = [np.linalg.norm(np.maximum(c[idx], 0.0)) / weights[gi]
            for gi, idx in enumerate(groups)]
    return max(max(vals), 1e-9)


def lambda_grid(G, c, groups, weights, group_blocks, q_max, n_lam=N_LAMBDAS):
    """
    Penalty grid restricted to the SPARSE regime: only the lambdas for which
    at most `q_max` foods enter the model are kept (Meinshausen & Bühlmann's
    rule). Without that bound, the selection frequency maximized along the
    path is ~1 for everyone and the selection no longer discriminates.
    """
    path = lambda_max(c, groups, weights) * np.logspace(0, -2.0, 50)
    beta, kept = None, []
    for lam in path:
        beta = nn_group_lasso(G, c, groups, lam, weights, group_blocks, beta=beta)
        active = sum(1 for idx in groups if beta[idx].max() > 0)
        if active == 0:
            continue
        if active > q_max:
            break
        kept.append(lam)
    if not kept:
        kept = [path[min(4, len(path) - 1)]]
    if len(kept) > n_lam:
        idx = np.linspace(0, len(kept) - 1, n_lam).round().astype(int)
        kept = [kept[i] for i in idx]
    return np.array(kept)


def orthonormalize(Xs, groups):
    """
    Replace the columns of each group by an orthonormal basis of the same
    subspace (X_g'X_g/n = I). The group lasso then becomes invariant to the
    correlation INTERNAL to the group — which is its correct formulation:
    without it the penalty is an ellipsoid and arbitrarily favours some lag
    kernels. Bonus: the block update has a closed form.
    """
    n = Xs.shape[0]
    Xq = np.zeros_like(Xs)
    for idx in groups:
        U, s, _vt = np.linalg.svd(Xs[:, idx], full_matrices=False)
        if s.size == 0 or s[0] <= 0:
            continue
        keep = s > s[0] * 1e-8
        Xq[:, idx[:int(keep.sum())]] = U[:, keep] * np.sqrt(n)
    return Xq


def group_lasso_ortho(G, c, groups, lam, weights, beta=None,
                      max_iter=200, tol=1e-7):
    """Group lasso on orthonormal groups: block descent, closed form."""
    beta = np.zeros(len(c)) if beta is None else beta.copy()
    Gbeta = G @ beta
    for _it in range(max_iter):
        delta_max = 0.0
        for gi, idx in enumerate(groups):
            bg = beta[idx]
            u = c[idx] - Gbeta[idx] + bg          # G_gg = I
            norm = np.sqrt(u @ u)
            threshold = lam * weights[gi]
            new = u * \
                (1.0 - threshold / norm) if norm > threshold else np.zeros_like(u)
            d = new - bg
            dmax = np.abs(d).max()
            if dmax > 0.0:
                beta[idx] = new
                Gbeta += G[:, idx] @ d
                delta_max = max(delta_max, dmax)
        if delta_max < tol:
            break
    return beta


def lambda_max_ortho(c, groups, weights):
    return max(max(np.linalg.norm(c[idx]) / weights[gi]
                   for gi, idx in enumerate(groups)), 1e-9)


def lambda_grid_ortho(G, c, groups, weights, q_max, n_lam=N_LAMBDAS):
    path = lambda_max_ortho(c, groups, weights) * np.logspace(0, -2.0, 40)
    beta, kept = None, []
    for lam in path:
        beta = group_lasso_ortho(G, c, groups, lam, weights, beta=beta)
        active = sum(1 for idx in groups if np.abs(beta[idx]).max() > 0)
        if active == 0:
            continue
        if active > q_max:
            break
        kept.append(lam)
    if not kept:
        kept = [path[min(4, len(path) - 1)]]
    if len(kept) > n_lam:
        i = np.linspace(0, len(kept) - 1, n_lam).round().astype(int)
        kept = [kept[j] for j in i]
    return np.array(kept)


# ══════════════════════════════════════════════════════════════════
#  Inseparable foods
# ══════════════════════════════════════════════════════════════════

def fuse_inseparable(meals, foods, threshold=JACCARD_THRESHOLD):
    """
    Two foods that are almost always eaten together cannot be separated by
    observational data: we merge them into an explicit block rather than
    letting the lasso pick one of them at random.
    → (blocks, members): blocks = block names, members = {block: [foods]}
    """
    presence = {a: set() for a in foods}
    for i, (_t, items) in enumerate(meals):
        for a in items:
            if a in presence:
                presence[a].add(i)

    parent = {a: a for a in foods}

    def root_of(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(foods):
        for b in foods[i + 1:]:
            inter = len(presence[a] & presence[b])
            if not inter:
                continue
            if inter / len(presence[a] | presence[b]) >= threshold:
                ra, rb = root_of(a), root_of(b)
                if ra != rb:
                    parent[rb] = ra

    members = defaultdict(list)
    for a in foods:
        members[root_of(a)].append(a)
    blocks, out = [], {}
    for ms in members.values():
        name = "+".join(sorted(ms))
        blocks.append(name)
        out[name] = sorted(ms)
    blocks.sort()
    return blocks, out


# ══════════════════════════════════════════════════════════════════
#  Full analysis
# ══════════════════════════════════════════════════════════════════

def analyze(observations, meals, min_occurrences=MIN_OCCURRENCES,
            n_replicates=N_REPLICATES, threshold=STABILITY_THRESHOLD, q_max=None,
            whiten=True, positive=True, seed=0):
    """
    Args:
        observations : [(time_hours, hour_of_day, pain_score)]
        meals        : [(time_hours, [normalized foods])]

    Returns:
        dict of results
    """
    rng = np.random.default_rng(seed)

    # We check the number of occurrences of each food in the meals and keep only
    # those that meet the minimum occurrence threshold. Then, we merge foods
    # that are almost always consumed together into a single block.
    counts = defaultdict(int)
    for _key, items in meals:
        for a in set(items):
            counts[a] += 1
    frequent = sorted(a for a, c in counts.items() if c >= min_occurrences)
    if not frequent:
        return {"error": _("no food reaches the minimum number of occurrences"),
                "counts": dict(counts)}

    blocks, members = fuse_inseparable(meals, frequent)
    to_block = {a: b for b, ms in members.items() for a in ms}
    block_meals = [(t, sorted({to_block[a] for a in items if a in to_block}))
                   for t, items in meals]

    # -- matrices -----------------------------------------------------
    t_obs = np.array([o[0] for o in observations], dtype=float)
    hours = np.array([o[1] for o in observations], dtype=float)
    y = np.array([o[2] for o in observations], dtype=float)

    X = build_exposition(t_obs, block_meals, blocks)
    Z = build_controls(t_obs, hours)
    rho = 0.0
    if whiten and len(y) > 4:
        y_b, X_b, Z_b, rho = whiten_ar1(y, X, Z, t_obs)
        y_res, X_res = residualize(y_b, X_b, Z_b)
        X, t_obs_eff = X[1:], t_obs[1:]
    else:
        y_res, X_res = residualize(y, X, Z)
        t_obs_eff = t_obs

    n, K = len(y_res), len(LAG_CENTERS)

    # -- structurally undetectable foods ------------------------------
    #  If the exposure to a food is almost entirely absorbed by the controls
    #  (a food eaten every day at a fixed time), no variation is left to
    #  exploit: its residual column is pure noise, which standardization
    #  would amplify until it stands out. We discard it explicitly rather
    #  than produce a misleading result.
    raw_norm = np.sqrt((X ** 2).sum(axis=0))
    res_norm = np.sqrt((X_res ** 2).sum(axis=0))
    free_share = np.where(raw_norm > 1e-12, res_norm /
                          np.maximum(raw_norm, 1e-12), 0.0)
    informative = free_share.reshape(len(blocks), K).max(axis=1) >= 0.10
    undetectable = [blocks[b] for b in np.flatnonzero(~informative)]
    if not informative.any():
        return {"error": _("no food varies enough to be tested "
                           "(diet too regular)")}
    if not informative.all():
        kept_idx = np.flatnonzero(informative)
        cols = np.concatenate([np.arange(b * K, (b + 1) * K) for b in kept_idx])
        X, X_res = X[:, cols], X_res[:, cols]
        blocks = [blocks[b] for b in kept_idx]

    n_blocks = len(blocks)
    scale = np.sqrt((X_res ** 2).sum(axis=0) / n)
    scale[scale < 1e-12] = 1.0
    Xs = X_res / scale
    groups = [np.arange(b * K, (b + 1) * K) for b in range(n_blocks)]
    weights = np.full(n_blocks, np.sqrt(K))

    if q_max is None:
        q_max = max(4, round(np.sqrt(0.8 * n_blocks)))

    if positive:
        G_tot = Xs.T @ Xs / n
        c_tot = Xs.T @ y_res / n
        lambdas = lambda_grid(G_tot, c_tot, groups, weights,
                                prepare_groups(G_tot, groups), q_max)
    else:
        Xo = orthonormalize(Xs, groups)
        G_tot = Xo.T @ Xo / n
        c_tot = Xo.T @ y_res / n
        lambdas = lambda_grid_ortho(G_tot, c_tot, groups, weights, q_max)

    # -- stability selection, contiguous day blocks -------------------
    obs_day = (t_obs_eff // 24).astype(int)
    days = np.unique(obs_day)
    starts = list(range(0, len(days), DAY_BLOCK_SIZE))
    n_drawn = max(1, round(len(starts) * SUBSAMPLE_FRACTION))

    sel_count = np.zeros((len(lambdas), n_blocks))
    valid_replicates = 0

    for _rep in range(n_replicates):
        chosen = rng.choice(len(starts), size=n_drawn, replace=False)
        kept_days = np.concatenate(
            [days[d:d + DAY_BLOCK_SIZE] for d in (starts[c] for c in chosen)])
        mask = np.isin(obs_day, kept_days)
        nb = int(mask.sum())
        if nb < 8:
            continue
        valid_replicates += 1
        Xb, yb = Xs[mask], y_res[mask]
        if not positive:
            Xb = orthonormalize(Xb, groups)
        G = Xb.T @ Xb / nb
        c = Xb.T @ yb / nb
        group_blocks = prepare_groups(G, groups) if positive else None
        beta = None
        for li, lam in enumerate(lambdas):
            beta = (nn_group_lasso(G, c, groups, lam, weights, group_blocks, beta=beta)
                    if positive else
                    group_lasso_ortho(G, c, groups, lam, weights, beta=beta))
            per_block = np.abs(beta.reshape(n_blocks, K))
            active = per_block.max(axis=1) > 0
            sel_count[li, active] += 1

    if valid_replicates == 0:
        return {"error": _("diary too short for resampling")}

    frequencies = sel_count.max(axis=0) / valid_replicates

    # -- unpenalized refit on the candidates --------------------------
    #  The group lasso spreads the coefficients inside a group (L2 norm):
    #  its profile does not locate the lag. We therefore refit without
    #  penalty (NNLS) on the candidate foods only, and it is from THAT
    #  profile that the lag and the effect are read, in pain points.
    from scipy.optimize import nnls

    lags = np.full(n_blocks, np.nan)
    effects = np.zeros(n_blocks)
    peaks = np.zeros(n_blocks)
    #  The refit must stay SPARSE: pouring the near-selected ones in dilutes
    #  the contributions across correlated columns and shifts the estimated
    #  lag. We restrict it to the selected foods (at least the first two).
    candidates = np.flatnonzero(frequencies >= threshold)
    if len(candidates) < 2:
        candidates = np.argsort(-frequencies)[:2]
    cols = np.concatenate([np.arange(b * K, (b + 1) * K) for b in candidates])
    coef, _residual = nnls(X_res[:, cols], y_res)
    #  The lag is read off the reconstructed RESPONSE FUNCTION
    #  h(d) = sum_k coef_k * kernel_k(d), not off the mass of the
    #  contributions: wide kernels aggregate more meals, so their columns
    #  have a larger mean, which would bias the centroid towards long
    #  delays whatever the true lag.
    d_grid = np.linspace(0.0, MAX_SPAN, 400)
    base_d = kernels_weights(d_grid)                              # (400, K)
    for pos, b in enumerate(candidates):
        tr = slice(pos * K, (pos + 1) * K)
        response = base_d @ coef[tr]                              # (400,)
        if response.max() > 1e-9:
            lags[b] = float(d_grid[int(np.argmax(response))])
            peaks[b] = float(response.max())
        # coefficients estimated on the residualized data (exact, FWL),
        # applied to the RAW exposure → pain points actually added
        effects[b] = float((X[:, cols[tr]] * coef[tr]).mean(axis=0).sum())

    selected = np.flatnonzero(frequencies >= threshold)

    return {
        "blocks": blocks,
        "members": members,
        "frequencies": frequencies,
        "lags": lags,
        "effects": effects,
        "peaks": peaks,
        "occurrences": np.array([
            sum(1 for _, al in block_meals if b in al) for b in blocks]),
        "selected": selected,
        "n_observations": n,
        "rho_ar1": rho,
        "n_days": len(days),
        "n_replicates": valid_replicates,
        "lambdas": lambdas,
        "q_max": q_max,
        "threshold": threshold,
        "discarded": sorted(a for a, c in counts.items() if c < min_occurrences),
        "undetectable": undetectable,
    }
