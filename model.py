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

# Lag basis: raised-cosine kernels, 0-84 h. Centers and half-widths in hours.
LAG_CENTERS = np.array([2.0, 5.0, 9.0, 15.0, 24.0, 40.0, 60.0])
LAG_WIDTHS = np.array([3.0, 4.0, 5.0, 8.0, 12.0, 18.0, 24.0])
MIN_DELAY = 0.25          # A meal never acts on a simultaneous pain
MAX_SPAN = float(LAG_CENTERS[-1] + LAG_WIDTHS[-1])

# Selection by stability (Meinshausen & Bühlmann 2010)
TAILLE_BLOC_JOURS = 3     # blocs contigus : préserve l'autocorrélation
FRACTION_SOUS_ECH = 0.5
N_REPLICATS = 200
N_LAMBDAS = 6
SEUIL_STABILITE = 0.70

MIN_OCCURRENCES = 3       # nb minimal de repas contenant l'aliment
SEUIL_JACCARD = 0.85      # au-delà : aliments indissociables, fusionnés


def normalize(name):
    """ Normalize a food name to a canonical form: lowercase, no accents, unified separators.

    Examples:
        >>> normalize("Pain au chocolat")
        'pain_au_chocolat'
        >>> normalize("Crème brûlée")
        'creme_brulee'
        >>> normalize("Salade de fruits - frais")
        'salade_de_fruits_frais'

    Args:
        name (str): The food name to normalize.

    Returns:
        str: The normalized food name.
    """
    name = unicodedata.normalize("NFKD", name.strip().lower())
    name = "".join(c for c in name if not unicodedata.combining(c))
    return "_".join(name.replace("-", " ").replace("_", " ").split())


# ══════════════════════════════════════════════════════════════════
#  Matrice d'exposition
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
        pertinents = (delays >= MIN_DELAY) & (delays <= MAX_SPAN)
        if not pertinents.any():
            continue
        w = kernels_weights(delays[pertinents])          # (m, K)
        lines = np.flatnonzero(pertinents)
        for i in targets:
            j = idx[norm_foods[i]] * K
            X[np.ix_(lines, np.arange(j, j + K))] += w
    return X


import subprocess
import tempfile

def plot_exposition_gnuplot(X, t_obs, foods, K, lag_centers,
                             columns=None, outfile=None, title=None,
                             xlabel="Temps (h)", ylabel="Dose cumulée",
                             terminal="qt"):
    """
    Trace une ou plusieurs colonnes de la matrice retournée par
    build_exposition, avec gnuplot.

    Args:
        X: (n, F*K) matrice de design (sortie de build_exposition).
        t_obs: (n,) temps des observations (axe des abscisses).
        foods: liste ordonnée des aliments (même ordre que build_exposition).
        K: nombre de noyaux de lag (len(LAG_CENTERS)).
        lag_centers: centres des noyaux (LAG_CENTERS), pour légender.
        columns: liste des séries à tracer. Chaque élément peut être :
            - un int : index de colonne direct dans X
            - (aliment, k) : aliment + index de noyau (0 <= k < K)
            - (aliment, None) : somme sur tous les noyaux pour cet aliment
              (exposition totale, tous lags confondus)
            Si None : trace le total (tous lags) de chaque aliment de `foods`.
        outfile: chemin d'image (.png/.svg/.pdf) ; si None, fenêtre interactive.
        title, xlabel, ylabel: habillage du graphe.
        terminal: terminal gnuplot interactif (qt, wxt, x11...).

    Returns:
        Le chemin du fichier produit (si outfile fourni) ou None.
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
            aliment, k = item
            if aliment not in idx:
                raise ValueError(f"Aliment inconnu: {aliment}")
            base = idx[aliment] * K
            if k is None:
                values = X[:, base:base + K].sum(axis=1)
                label = f"{aliment} (total)"
            else:
                if not (0 <= k < K):
                    raise ValueError(f"Index de noyau invalide: {k}")
                values = X[:, base + k]
                lag_lbl = lag_centers[k] if lag_centers is not None else k
                label = f"{aliment} (lag={lag_lbl})"
        series.append((label, values))

    # Tri par temps pour un tracé propre
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


def blanchir_ar1(y, X, Z, t_obs):
    """
    Transformation de Cochrane-Orcutt : la douleur est autocorrélée d'un repas
    au suivant. Sans blanchiment, le modèle « explique » cette inertie en
    attribuant à des aliments innocents la lente dérive du niveau de douleur.
    Le coefficient est atténué en puissance de l'écart de temps réel, les
    observations n'étant pas régulièrement espacées. Suppose `t_obs` trié.
    → (y, X, Z, rho) — la première observation est consommée par la transformation.
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
#  Group-lasso à coefficients positifs
# ══════════════════════════════════════════════════════════════════
#  min_{beta >= 0}  1/(2n)||y - X beta||^2 + lam * sum_g sqrt(K) * ||beta_g||_2
#
#  Un GROUPE = un aliment, c'est-à-dire ses K coefficients de décalage. Ils
#  entrent ou sortent du modèle ensemble : un vrai effet s'étale sur plusieurs
#  noyaux adjacents, et un lasso ordinaire paierait la pénalité sur chacun
#  séparément — d'où une perte de puissance.
#
#  Résolution par descente par blocs sur la matrice de Gram (astuce
#  « covariance updates » de glmnet), sous-problème de groupe par FISTA.
#  L'opérateur proximal de  lam||.||_2 + indicatrice(. >= 0)  est le
#  seuillage de groupe appliqué à la partie positive.

def _prox_groupe(v, seuil):
    v = np.maximum(v, 0.0)
    n = np.sqrt(v @ v)
    return v * max(0.0, 1.0 - seuil / n) if n > 0 else v


def _solve_group(Gg, u, lam_g, b0, pas, n_iter=80, tol=1e-9):
    """min_{b>=0} 1/2 b'Gg b - u'b + lam_g||b||_2, par FISTA."""
    b = b0.copy()
    z = b.copy()
    t = 1.0
    for _ in range(n_iter):
        b_new = _prox_groupe(z - pas * (Gg @ z - u), pas * lam_g)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = b_new + ((t - 1.0) / t_new) * (b_new - b)
        fini = np.abs(b_new - b).max() < tol
        b, t = b_new, t_new
        if fini:
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

    for _ in range(max_iter):
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


def lambda_max(c, groupes, poids):
    """Plus petite pénalité annulant toute la solution."""
    vals = [np.linalg.norm(np.maximum(c[idx], 0.0)) / poids[gi]
            for gi, idx in enumerate(groupes)]
    return max(max(vals), 1e-9)


def grille_lambda(G, c, groupes, poids, blocs_g, q_max, n_lam=N_LAMBDAS):
    """
    Grille de pénalités restreinte au régime PARCIMONIEUX : on ne garde que
    les lambda pour lesquels au plus `q_max` aliments entrent dans le modèle
    (règle de Meinshausen & Bühlmann). Sans cette borne, la fréquence de
    sélection maximisée sur le chemin vaut ~1 pour tout le monde et la
    sélection ne discrimine plus rien.
    """
    chemin = lambda_max(c, groupes, poids) * np.logspace(0, -2.0, 50)
    beta, retenues = None, []
    for lam in chemin:
        beta = nn_group_lasso(G, c, groupes, lam, poids, blocs_g, beta=beta)
        actifs = sum(1 for idx in groupes if beta[idx].max() > 0)
        if actifs == 0:
            continue
        if actifs > q_max:
            break
        retenues.append(lam)
    if not retenues:
        retenues = [chemin[min(4, len(chemin) - 1)]]
    if len(retenues) > n_lam:
        idx = np.linspace(0, len(retenues) - 1, n_lam).round().astype(int)
        retenues = [retenues[i] for i in idx]
    return np.array(retenues)


def orthonormaliser(Xs, groupes):
    """
    Remplace les colonnes de chaque groupe par une base orthonormée du même
    sous-espace (X_g'X_g/n = I). Le group-lasso devient alors invariant à la
    corrélation INTERNE au groupe — c'est sa formulation correcte : sans cela
    la pénalité est un ellipsoïde et favorise arbitrairement certains noyaux
    de décalage. Bonus : la mise à jour de bloc a une forme fermée.
    """
    n = Xs.shape[0]
    Xq = np.zeros_like(Xs)
    for idx in groupes:
        U, s, _ = np.linalg.svd(Xs[:, idx], full_matrices=False)
        if s.size == 0 or s[0] <= 0:
            continue
        garde = s > s[0] * 1e-8
        Xq[:, idx[:int(garde.sum())]] = U[:, garde] * np.sqrt(n)
    return Xq


def group_lasso_ortho(G, c, groupes, lam, poids, beta=None,
                      max_iter=200, tol=1e-7):
    """Group-lasso sur groupes orthonormés : descente par blocs, forme fermée."""
    beta = np.zeros(len(c)) if beta is None else beta.copy()
    Gbeta = G @ beta
    for _ in range(max_iter):
        delta_max = 0.0
        for gi, idx in enumerate(groupes):
            bg = beta[idx]
            u = c[idx] - Gbeta[idx] + bg          # G_gg = I
            norme = np.sqrt(u @ u)
            seuil = lam * poids[gi]
            nouveau = u * \
                (1.0 - seuil / norme) if norme > seuil else np.zeros_like(u)
            d = nouveau - bg
            dmax = np.abs(d).max()
            if dmax > 0.0:
                beta[idx] = nouveau
                Gbeta += G[:, idx] @ d
                delta_max = max(delta_max, dmax)
        if delta_max < tol:
            break
    return beta


def lambda_max_ortho(c, groupes, poids):
    return max(max(np.linalg.norm(c[idx]) / poids[gi]
                   for gi, idx in enumerate(groupes)), 1e-9)


def grille_lambda_ortho(G, c, groupes, poids, q_max, n_lam=N_LAMBDAS):
    chemin = lambda_max_ortho(c, groupes, poids) * np.logspace(0, -2.0, 40)
    beta, retenues = None, []
    for lam in chemin:
        beta = group_lasso_ortho(G, c, groupes, lam, poids, beta=beta)
        actifs = sum(1 for idx in groupes if np.abs(beta[idx]).max() > 0)
        if actifs == 0:
            continue
        if actifs > q_max:
            break
        retenues.append(lam)
    if not retenues:
        retenues = [chemin[min(4, len(chemin) - 1)]]
    if len(retenues) > n_lam:
        i = np.linspace(0, len(retenues) - 1, n_lam).round().astype(int)
        retenues = [retenues[j] for j in i]
    return np.array(retenues)


# ══════════════════════════════════════════════════════════════════
#  Aliments indissociables
# ══════════════════════════════════════════════════════════════════

def fuse_inseparable(repas, aliments, seuil=SEUIL_JACCARD):
    """
    Deux aliments presque toujours consommés ensemble ne sont pas séparables
    par des données d'observation : on les fusionne en un bloc explicite plutôt
    que de laisser le lasso en désigner un au hasard.
    → (blocs, membres) : blocs = noms de blocs, membres = {bloc: [aliments]}
    """
    presence = {a: set() for a in aliments}
    for i, (_, alims) in enumerate(repas):
        for a in alims:
            if a in presence:
                presence[a].add(i)

    parent = {a: a for a in aliments}

    def racine(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(aliments):
        for b in aliments[i + 1:]:
            inter = len(presence[a] & presence[b])
            if not inter:
                continue
            if inter / len(presence[a] | presence[b]) >= seuil:
                ra, rb = racine(a), racine(b)
                if ra != rb:
                    parent[rb] = ra

    membres = defaultdict(list)
    for a in aliments:
        membres[racine(a)].append(a)
    blocs, sortie = [], {}
    for ms in membres.values():
        nom = "+".join(sorted(ms))
        blocs.append(nom)
        sortie[nom] = sorted(ms)
    blocs.sort()
    return blocs, sortie


# ══════════════════════════════════════════════════════════════════
#  Analyse complète
# ══════════════════════════════════════════════════════════════════

def analyze(observations, repas, min_occurrences=MIN_OCCURRENCES,
            n_replicats=N_REPLICATS, seuil=SEUIL_STABILITE, q_max=None,
            blanchir=True, positif=True, graine=0):
    """
    Args:
        observations : [(instant_heures, heure_du_jour, score_douleur)]
        repas        : [(instant_heures, [aliments normalisés])]

    Returns:
        dict de résultats
    """
    rng = np.random.default_rng(graine)

    # We check the number of occurrences of each food in the meals and keep only
    # those that meet the minimum occurrence threshold. Then, we merge foods
    # that are almost always consumed together into a single block.
    compte = defaultdict(int)
    for _, alims in repas:
        for a in set(alims):
            compte[a] += 1
    frequents = sorted(a for a, c in compte.items() if c >= min_occurrences)
    if not frequents:
        return {"error": _("no food reaches the minimum number of occurrences"),
                "compte": dict(compte)}

    blocs, members = fuse_inseparable(repas, frequents)
    vers_bloc = {a: b for b, ms in members.items() for a in ms}
    repas_blocs = [(t, sorted({vers_bloc[a] for a in alims if a in vers_bloc}))
                   for t, alims in repas]

    # -- matrices -----------------------------------------------------
    t_obs = np.array([o[0] for o in observations], dtype=float)
    heures = np.array([o[1] for o in observations], dtype=float)
    y = np.array([o[2] for o in observations], dtype=float)

    X = build_exposition(t_obs, repas_blocs, blocs)
    Z = build_controls(t_obs, heures)
    rho = 0.0
    if blanchir and len(y) > 4:
        y_b, X_b, Z_b, rho = blanchir_ar1(y, X, Z, t_obs)
        y_res, X_res = residualize(y_b, X_b, Z_b)
        X, t_obs_eff = X[1:], t_obs[1:]
    else:
        y_res, X_res = residualize(y, X, Z)
        t_obs_eff = t_obs

    n, K = len(y_res), len(LAG_CENTERS)

    # -- aliments structurellement indétectables ---------------------
    #  Si l'exposition à un aliment est presque entièrement absorbée par les
    #  contrôles (aliment consommé tous les jours à heure fixe), il ne reste
    #  aucune variation à exploiter : sa colonne résiduelle n'est que du bruit,
    #  que la standardisation amplifierait jusqu'à le faire ressortir. On
    #  l'écarte explicitement plutôt que de produire un résultat trompeur.
    norme_brute = np.sqrt((X ** 2).sum(axis=0))
    norme_res = np.sqrt((X_res ** 2).sum(axis=0))
    part_libre = np.where(norme_brute > 1e-12, norme_res /
                          np.maximum(norme_brute, 1e-12), 0.0)
    informatif = part_libre.reshape(len(blocs), K).max(axis=1) >= 0.10
    indetectables = [blocs[b] for b in np.flatnonzero(~informatif)]
    if not informatif.any():
        return {"error": _("no food varies enough to be tested "
                           "(diet too regular)")}
    if not informatif.all():
        gardes = np.flatnonzero(informatif)
        cols = np.concatenate([np.arange(b * K, (b + 1) * K) for b in gardes])
        X, X_res = X[:, cols], X_res[:, cols]
        blocs = [blocs[b] for b in gardes]

    n_blocs = len(blocs)
    echelle = np.sqrt((X_res ** 2).sum(axis=0) / n)
    echelle[echelle < 1e-12] = 1.0
    Xs = X_res / echelle
    groupes = [np.arange(b * K, (b + 1) * K) for b in range(n_blocs)]
    poids = np.full(n_blocs, np.sqrt(K))

    if q_max is None:
        q_max = max(4, round(np.sqrt(0.8 * n_blocs)))

    if positif:
        G_tot = Xs.T @ Xs / n
        c_tot = Xs.T @ y_res / n
        lambdas = grille_lambda(G_tot, c_tot, groupes, poids,
                                prepare_groups(G_tot, groupes), q_max)
    else:
        Xo = orthonormaliser(Xs, groupes)
        G_tot = Xo.T @ Xo / n
        c_tot = Xo.T @ y_res / n
        lambdas = grille_lambda_ortho(G_tot, c_tot, groupes, poids, q_max)

    # -- sélection par stabilité, blocs de jours contigus -------------
    jour_obs = (t_obs_eff // 24).astype(int)
    jours = np.unique(jour_obs)
    debuts = list(range(0, len(jours), TAILLE_BLOC_JOURS))
    n_tire = max(1, round(len(debuts) * FRACTION_SOUS_ECH))

    compte_sel = np.zeros((len(lambdas), n_blocs))
    replicats_valides = 0

    for _ in range(n_replicats):
        choisis = rng.choice(len(debuts), size=n_tire, replace=False)
        jours_gardes = np.concatenate(
            [jours[d:d + TAILLE_BLOC_JOURS] for d in (debuts[c] for c in choisis)])
        masque = np.isin(jour_obs, jours_gardes)
        nb = int(masque.sum())
        if nb < 8:
            continue
        replicats_valides += 1
        Xb, yb = Xs[masque], y_res[masque]
        if not positif:
            Xb = orthonormaliser(Xb, groupes)
        G = Xb.T @ Xb / nb
        c = Xb.T @ yb / nb
        blocs_g = prepare_groups(G, groupes) if positif else None
        beta = None
        for li, lam in enumerate(lambdas):
            beta = (nn_group_lasso(G, c, groupes, lam, poids, blocs_g, beta=beta)
                    if positif else
                    group_lasso_ortho(G, c, groupes, lam, poids, beta=beta))
            par_bloc = np.abs(beta.reshape(n_blocs, K))
            actifs = par_bloc.max(axis=1) > 0
            compte_sel[li, actifs] += 1

    if replicats_valides == 0:
        return {"error": _("log too short for resampling")}

    frequences = compte_sel.max(axis=0) / replicats_valides

    # -- ré-ajustement non pénalisé sur les candidats ----------------
    #  Le group-lasso étale les coefficients à l'intérieur d'un groupe (norme
    #  L2) : son profil ne localise pas le décalage. On ré-ajuste donc sans
    #  pénalité (NNLS) sur les seuls aliments candidats, et c'est de CE profil
    #  qu'on tire décalage et effet, exprimés en points de douleur.
    from scipy.optimize import nnls

    decalages = np.full(n_blocs, np.nan)
    effets = np.zeros(n_blocs)
    pics = np.zeros(n_blocs)
    #  Le ré-ajustement doit rester PARCIMONIEUX : y verser les quasi-retenus
    #  dilue les contributions entre colonnes corrélées et déplace le décalage
    #  estimé. On se limite aux aliments retenus (au minimum les 2 premiers).
    candidats = np.flatnonzero(frequences >= seuil)
    if len(candidats) < 2:
        candidats = np.argsort(-frequences)[:2]
    cols = np.concatenate([np.arange(b * K, (b + 1) * K) for b in candidats])
    coef, _ = nnls(X_res[:, cols], y_res)
    #  Le décalage se lit sur la FONCTION DE RÉPONSE reconstruite
    #  h(d) = somme_k coef_k * noyau_k(d), et non sur la masse des
    #  contributions : les noyaux larges agrègent plus de repas, donc leurs
    #  colonnes ont une moyenne plus grande, ce qui biaiserait le centroïde
    #  vers les longs délais quel que soit le vrai décalage.
    grille_d = np.linspace(0.0, MAX_SPAN, 400)
    base_d = kernels_weights(grille_d)                              # (400, K)
    for pos, b in enumerate(candidats):
        tr = slice(pos * K, (pos + 1) * K)
        reponse = base_d @ coef[tr]                              # (400,)
        if reponse.max() > 1e-9:
            decalages[b] = float(grille_d[int(np.argmax(reponse))])
            pics[b] = float(reponse.max())
        # coefficients estimés sur les données résiduelles (exacts, FWL),
        # appliqués à l'exposition BRUTE → points de douleur réellement ajoutés
        effets[b] = float((X[:, cols[tr]] * coef[tr]).mean(axis=0).sum())

    retenus = np.flatnonzero(frequences >= seuil)

    return {
        "blocs": blocs,
        "membres": membres,
        "frequences": frequences,
        "decalages": decalages,
        "effets": effets,
        "pics": pics,
        "occurrences": np.array([
            sum(1 for _, al in repas_blocs if b in al) for b in blocs]),
        "retenus": retenus,
        "n_observations": n,
        "rho_ar1": rho,
        "n_jours": len(jours),
        "n_replicats": replicats_valides,
        "lambdas": lambdas,
        "q_max": q_max,
        "seuil": seuil,
        "ecartes": sorted(a for a, c in compte.items() if c < min_occurrences),
        "indetectables": indetectables,
    }
