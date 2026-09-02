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
        )

def test_normalize():
    assert normalize("  Café  ") == "cafe", "Accents and spaces should be removed"
    assert normalize("Pomme-de-terre") == "pomme_de_terre", "Separators should be replaced with underscores"
    assert normalize("RIZ") == "riz", "Uppercase letters should be converted to lowercase"

def test_kernels_weights():
    d = np.linspace(0, MAX_SPAN, 500)
    w = kernels_weights(d)
    assert (w >= 0).all(), "All weights should be non-negative"
    assert (w <= 1).all(), "All weights should be less than or equal to 1"
    assert (kernels_weights([0.0, MIN_DELAY / 2]) == 0).all(), "Weights should be zero for distances less than MIN_DELAY"
    inside = d[(d >= MIN_DELAY) & (d <= LAG_CENTERS[-1])]
    # In inside, the sum of weights should be greater than 0.2 for each kernel. The meaning is that there should be no gaps in the ckernel coverages.
    # 0.2 is an arbitrary threshold to ensure that the coverage is continuous.
    assert kernels_weights(inside).sum(axis=1).min() > 0.2, "Continuous coverage: we should have no gaps in the kernel coverage"
    assert kernels_weights([MAX_SPAN * 2]).sum() == 0, "We should have zero everywhere outside the kernel coverage"

def test_exposition_matrix():
    X = build_exposition([10.0, 100.0], [(8.0, ["a"]), (95.0, ["b"])], ["a", "b"])
    K = len(LAG_CENTERS)
    assert X[0, :K].sum() > 0, "le repas 2 h avant expose l'aliment a"
    assert X[0, K:].sum() == 0, "un repas postérieur n'expose rien"
    assert X[1, K:].sum() > 0, "le repas 5 h avant expose l'aliment b"

