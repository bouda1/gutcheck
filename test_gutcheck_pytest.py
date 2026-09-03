#!/usr/bin/env python3
# gutcheck - pytest test suite
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

import numpy as np

from model import (
    MAX_SPAN,
    MIN_DELAY,
    LAG_CENTERS,
    kernels_weights,
    normalize,
    build_exposition,
    residualize,
    build_controls,
    nn_group_lasso,
    prepare_groups,
)


def test_normalize():
    assert normalize(
        "  Café  ") == "cafe", "Accents and spaces should be removed"
    assert normalize(
        "Pomme-de-terre") == "pomme_de_terre", "Separators should be replaced with underscores"
    assert normalize(
        "RIZ") == "riz", "Uppercase letters should be converted to lowercase"


def test_kernels_weights():
    d = np.linspace(0, MAX_SPAN, 500)
    w = kernels_weights(d)
    assert (w >= 0).all(), "All weights should be non-negative"
    assert (w <= 1).all(), "All weights should be less than or equal to 1"
    assert (kernels_weights([0.0, MIN_DELAY / 2]) ==
            0).all(), "Weights should be zero for distances less than MIN_DELAY"
    inside = d[(d >= MIN_DELAY) & (d <= LAG_CENTERS[-1])]
    # In inside, the sum of weights should be greater than 0.2 for each kernel.
    # The meaning is that there should be no gaps in the ckernel coverages.
    # 0.2 is an arbitrary threshold to ensure that the coverage is continuous.
    assert kernels_weights(inside).sum(axis=1).min(
    ) > 0.2, "Continuous coverage: we should have no gaps in the kernel coverage"
    assert kernels_weights(
        [MAX_SPAN * 2]).sum() == 0, "We should have zero everywhere outside the kernel coverage"


def test_exposition_matrix():
    # In the context of this test, the exposition matrix is built with 14 columns,
    # for each food and each kernel (2 * K).
    # The first K columns correspond to the first food (here "a"), one for each
    # lag kernel.
    # build_exposition is called with two observation hours (10.0 and 100.0).
    # two foods (a and b) with their respective consumption times (8.0 and 95.0).
    # Even if each column is a raised cosine kernel, we only keep the observation
    # hours values. So, here, X has only 2 rows.
    # The third argument is the list of foods, which is ["a", "b"].
    X = build_exposition(
        [10.0, 100.0], [(8.0, ["a"]), (95.0, ["b"])], ["a", "b"])
    K = len(LAG_CENTERS)
    # We look at the first observation time, 2 hours after the food "a" was consumed.
    # We look at the first K columns, which correspond to the food "a".
    # The sum of these columns should be greater than 0, and even greater than 1 because
    # the first kernel reaches its maximum at 2 hours after the food was consumed.
    assert X[0, :K].sum() > 1, "The meal 2 hours before should expose the food a"

    # We still look at the first observation time, but at the second K columns,
    # which correspond to the food "b".
    assert X[0, K:].sum() == 0, "The meal 85 hours later should expose nothing"
    # Now, we look at the second observation time, 5 hours after the food "b" was consumed.
    assert X[1, K:].sum() > 0, "The meal 5 hours before should expose the food b"


def test_residualize():
    rng = np.random.default_rng(0)
    # We build a design matrix Z with 40 rows and 6 columns, corresponding to
    # 6 control variables.
    Z = build_controls(np.arange(40) * 6.0, (np.arange(40) * 6.0) % 24)
    Xr = rng.normal(size=(40, 5))
    # We build a response variable y as a linear combination of the columns of Z and Xr,
    # plus some noise. The true coefficients for Xr are [2.0, 0, 0, 0, 0], and the
    # coefficients for Z are arbitrary.
    yr = Z @ np.arange(1, 7) + \
        Xr @ np.array([2.0, 0, 0, 0, 0]) + rng.normal(0, .1, 40)
    # We remove from yr and Xr the part that is explained by Z, to get the residuals y_res and X_res.
    y_res, X_res = residualize(yr, Xr, Z)
    # We check that the residuals are orthogonal to the controls Z, which means
    # that the dot product of Z.T with y_res and X_res should be close to zero.
    assert np.abs(Z.T @ y_res).max() < 1e-8, "y_res should be orthogonal to Z"
    # We check that the residuals X_res are also orthogonal to the controls Z.
    assert np.abs(Z.T @ X_res).max() < 1e-8, "X_res should be orthogonal to Z"
    # We check that the residuals y_res can be explained by X_res with the true
    # coefficients [2.0, 0, 0, 0, 0].
    beta_hat, *_ = np.linalg.lstsq(X_res, y_res, rcond=None)
    assert np.allclose(beta_hat, [
                       2.0, 0, 0, 0, 0], atol=0.1), "y_res should be explained by X_res with the true coefficients"

def test_group_lasso():
    n, G_blocs = 200, 6
    rng = np.random.default_rng(0)
    K = len(LAG_CENTERS)
    Xg = rng.normal(size=(n, G_blocs * K))
    beta_vrai = np.zeros(G_blocs * K)
    beta_vrai[K:2 * K] = 1.5          # seul le groupe 1 est actif
    yg = Xg @ beta_vrai + rng.normal(0, 0.5, n)
    groups = [np.arange(b * K, (b + 1) * K) for b in range(G_blocs)]
    Gm, cm = Xg.T @ Xg / n, Xg.T @ yg / n
    b = nn_group_lasso(Gm, cm, groups, 0.15, np.full(G_blocs, np.sqrt(K)),
                       prepare_groups(Gm, groups))
    actifs = [g for g in range(G_blocs) if b[groups[g]].max() > 0]
    assert actifs == [1], f"seul le groupe planté est sélectionné (obtenu {actifs})"
    assert (b >= 0).all(), "contrainte de positivité respectée"

