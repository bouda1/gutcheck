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

import os
import re
import tempfile
import zipfile

import numpy as np
import pytest

from i18n import _
from gutcheck import load_diary
from model import (
    MAX_SPAN,
    MIN_DELAY,
    LAG_CENTERS,
    analyze,
    kernels_weights,
    normalize,
    build_exposition,
    residualize,
    build_controls,
    nn_group_lasso,
    prepare_groups,
    fuse_inseparable,
)
from simu import generate
from spreadsheet import normalize_header, read_ods, read_sheet, write_ods

#  One sample diary, shared by every .ods test: four data rows covering a
#  typed date, a typed time, a numeric cell and an empty trailing cell.
SAMPLE_ROWS = [
    [_("date"), _("time"), _("meal"), _("foods"), "Pain (0-10)"],
    ["2026-03-01", "08:00", "breakfast", "Coffee; bread", 2],
    ["2026-03-01", "11:00", "", "", 4],
    ["2026-03-01", "12:30", "lunch", "rice; chicken", ""],
    ["2026-03-02", "19:00", "dinner", "soup", 5],
]


@pytest.fixture
def ods_diary(tmp_path):
    """Path to a freshly written .ods diary holding SAMPLE_ROWS."""
    path = str(tmp_path / "diary.ods")
    write_ods(path, SAMPLE_ROWS)
    return path


def test_normalize():
    assert normalize(
        "  Café  ") == "cafe", "Accents and spaces should be removed"
    assert normalize(
        "Sweet-potato") == "sweet_potato", "Separators should be replaced with underscores"
    assert normalize(
        "RICE") == "rice", "Uppercase letters should be converted to lowercase"


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
    # We simulate a linear regression problem with 200 observations and 6 groups of features.
    n, G_blocks = 200, 6
    rng = np.random.default_rng(0)
    K = len(LAG_CENTERS)
    # We build a design matrix Xg with n rows and G_blocks * K columns,
    # where each group of K columns corresponds to a different feature group.
    Xg = rng.normal(size=(n, G_blocks * K))
    # We plant a true coefficient vector beta_true, where only the second group
    # (group 1) has non-zero coefficients.
    beta_true = np.zeros(G_blocks * K)
    beta_true[K:2 * K] = 1.5          # Only the group 1 is active
    # We generate a response variable yg as a linear combination of Xg and beta_true,
    # plus some noise.
    yg = Xg @ beta_true + rng.normal(0, 0.5, n)
    # We define the groups of features for the group lasso, where each group
    # consists of K consecutive columns.
    groups = [np.arange(b * K, (b + 1) * K) for b in range(G_blocks)]
    # We compute the Gram matrix Gm and the vector cm for the group lasso problem.
    Gm, cm = Xg.T @ Xg / n, Xg.T @ yg / n
    b = nn_group_lasso(Gm, cm, groups, 0.15, np.full(G_blocks, np.sqrt(K)),
                       prepare_groups(Gm, groups))
    active = [g for g in range(G_blocks) if b[groups[g]].max() > 0]
    assert active == [1], f"only the planted group should be selected (got {active})"
    assert (b >= 0).all(), "we should have non negative coefficients"

def test_fusion():
    meal = [(float(i), ["garlic", "onion"] + (["rice"] if i % 2 else []))
            for i in range(10)]
    blocks, _members = fuse_inseparable(meal, ["garlic", "onion", "rice"])
    assert "garlic+onion" in blocks, "foods that are always consumed together should be fused"
    assert "rice" in blocks, "independent foods should not be fused"

def test_read_csv():
    content = """{date},{time},{meal},{foods},{pain}
2026-03-02,19:00,dinner,soup,5
2026-03-01,,breakfast,Coffee; bread,2
2026-03-01,12:30,lunch,rice,
2026-03-01,15:00,,,4
2026-03-01,16:00,,,abc
not-a-date,08:00,,,3
""".format(
           date=_("date"),
           time=_("time"),
           meal=_("meal"),
           foods=_("foods"),
           pain=_("pain"),
       )

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                     encoding="utf-8") as f:
        f.write(content)
        path = f.name
    obs, rep, warnings = load_diary(path)
    os.unlink(path)
    assert [o[0] for o in obs] == sorted(o[0] for o in obs), "observation times should be sorted"
    assert len(rep) == 3, f"3 meals read (got {len(rep)})"
    assert len(obs) == 3, f"3 pain observations read (got {len(obs)})"
    assert any("without a time" in a for a in warnings), "empty time should be reported"
    assert abs((rep[1][0] - rep[0][0]) - 0.5) < 1e-9, "empty time should be translated into 12:00 (30 min before the lunch at 12:30)"
    assert any("abc" in a for a in warnings), "unreadable pain value should be reported"
    assert any("not-a-date" in a for a in warnings), "unreadable date should be reported"
    assert ["bread", "coffee"] == rep[0][1], f"first meal foods should be normalized (got {rep[0][1]})"

def test_spreadsheet(ods_diary):
    assert normalize_header("  Pain (0-10) ") == "pain", "a decorated header is normalized"
    assert normalize_header("Foods") == "foods", "header case is ignored"

    d = read_ods(ods_diary)
    assert len(d) == 4, f"4 data rows should be read (got {len(d)})"
    assert d[0]["date"] == "2026-03-01", \
        "a typed DATE cell is read back as ISO, not as the displayed text"
    assert d[0]["time"] == "08:00", \
        f"a typed TIME cell is read back as HH:MM (got {d[0]['time']!r})"
    assert d[0]["pain"] == "2", "a numeric cell has no spurious decimal part"
    assert d[2]["pain"] == "", "an empty trailing cell is an empty string"

    obs_o, rep_o, _dumb = load_diary(ods_diary)
    assert len(rep_o) == 3 and len(obs_o) == 3, \
        "an .ods diary loads like its CSV equivalent"
    assert rep_o[0][1] == ["bread", "coffee"], \
        "names are normalized when read from the workbook"


def test_ods_package(ods_diary):
    #  Without a data style, Calc would display the internal value of a date
    #  (number of days since the epoch) instead of the date itself.
    with zipfile.ZipFile(ods_diary) as z:
        content = z.read("content.xml").decode()
        entries = z.namelist()

    assert "<number:date-style" in content, "the date display format is declared"
    assert "<number:time-style" in content, "the time display format is declared"

    declared = set(re.findall(r'<style:style style:name="([^"]+)"', content))
    used = set(re.findall(r'table:style-name="([^"]+)"', content))
    assert used, "some cells reference a style"
    assert used <= declared, f"every referenced style is declared (used {sorted(used)})"

    typed_cells = re.findall(
        r'<table:table-cell[^>]*office:value-type="(?:date|time)"[^>]*>', content)
    assert typed_cells, "the workbook holds typed date/time cells"
    assert all("table:style-name=" in c for c in typed_cells), \
        "every date/time cell carries a display style"

    assert entries[0] == "mimetype", "mimetype is the first entry of the archive"
    assert "styles.xml" in entries, "styles.xml is present in the package"


def test_ods_compressed_cells(tmp_path):
    #  Calc compresses empty cells and rows at the end of a sheet, commonly
    #  with counts of 16384 columns and 1048576 rows. They must be expanded,
    #  but bounded, or a million empty rows get materialized.
    content = ('<?xml version="1.0" encoding="UTF-8"?>'
               '<office:document-content'
               ' xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"'
               ' xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"'
               ' xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
               '<office:body><office:spreadsheet><table:table table:name="f">'
               '<table:table-row><table:table-cell office:value-type="string">'
               '<text:p>date</text:p></table:table-cell>'
               '<table:table-cell table:number-columns-repeated="3"/>'
               '<table:table-cell office:value-type="string">'
               '<text:p>pain</text:p></table:table-cell></table:table-row>'
               '<table:table-row table:number-rows-repeated="1048576">'
               '<table:table-cell table:number-columns-repeated="16384"/>'
               '</table:table-row>'
               '</table:table></office:spreadsheet></office:body>'
               '</office:document-content>')
    path = str(tmp_path / "compressed.ods")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("content.xml", content)

    rows = read_sheet(path)
    assert len(rows) == 1, f"repeated empty rows are not materialized (got {len(rows)})"
    assert rows[0] == ["date", "", "", "", "pain"], \
        f"interleaved empty columns are expanded (got {rows[0]})"


@pytest.mark.slow
def test_end_to_end_detection():
    #  56 days under the recommended protocol (6 readings a day away from
    #  meals): the three planted culprits must come out of the analysis.
    obs, meals, truth = generate(n_days=56, seed=0,
                                 reading_hours=[7, 10, 13, 16, 19, 22])
    res = analyze(obs, meals, n_replicates=80, seed=0)
    selected = {res["blocks"][i] for i in res["selected"]}

    assert set(truth) <= selected, \
        f"the 3 culprits are selected (selected: {sorted(selected)})"
    assert len(selected - set(truth)) <= 1, \
        f"at most 1 false positive (got {sorted(selected - set(truth))})"
    for name, (true_lag, _amp) in truth.items():
        est = res["lags"][res["blocks"].index(name)]
        assert abs(est - true_lag) <= 8, \
            f"lag of {name}: {est:.0f} h estimated vs {true_lag:.0f} h actual"
