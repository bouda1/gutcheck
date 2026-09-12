# gutcheck - synthetic journal generator with known culprits
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
Generator of synthetic diaries with KNOWN culprits, and evaluation of the
algorithm on them. This is the only way to know whether it works: on real
data, the truth is not observable.

The generator deliberately reproduces the traps of the real problem:
  - circadian rhythm (stronger pain in the evening);
  - foods tied to a time of day (coffee is at breakfast) — hence a possible
    confusion between "food" and "time";
  - pain autocorrelated from one meal to the next;
  - foods always eaten together (inseparable);
  - several simultaneous culprits, with different lags.

Food names are source (English) strings translated through gettext, so a
generated diary is written in the user's language.
"""

import numpy as np
from babel.dates import format_date

from i18n import LANG_CODE, N_, _
from model import analyze, normalize

MOMENTS = [(8.0, _("breakfast")), (12.5, _("lunch")), (19.0, _("dinner"))]

CATALOGUE = [
    # (name, P(breakfast), P(lunch), P(dinner))
    (N_("coffee"),     0.90, 0.20, 0.02), (N_("bread"),      0.75, 0.35, 0.30),
    (N_("butter"),     0.60, 0.05, 0.10), (N_("jam"),        0.40, 0.00, 0.00),
    (N_("milk"),       0.45, 0.02, 0.05), (N_("cereal"),     0.30, 0.00, 0.00),
    (N_("eggs"),       0.25, 0.10, 0.10), (N_("banana"),     0.20, 0.15, 0.05),
    (N_("yoghurt"),    0.20, 0.15, 0.20), (N_("orange juice"), 0.25, 0.05, 0.02),
    (N_("rice"),       0.00, 0.30, 0.25), (N_("pasta"),      0.00, 0.30, 0.20),
    (N_("chicken"),    0.00, 0.30, 0.20), (N_("beef"),       0.00, 0.20, 0.15),
    (N_("fish"),       0.00, 0.15, 0.20), (N_("vegetables"), 0.02, 0.55, 0.55),
    (N_("salad"),      0.00, 0.35, 0.30), (N_("tomato"),     0.00, 0.30, 0.25),
    (N_("cheese"),     0.05, 0.35, 0.30), (N_("potato"),     0.00, 0.25, 0.25),
    (N_("oil"),        0.02, 0.45, 0.45), (N_("wine"),       0.00, 0.15, 0.30),
    (N_("chocolate"),  0.10, 0.15, 0.20), (N_("apple"),      0.10, 0.20, 0.15),
    (N_("soup"),       0.00, 0.05, 0.35), (N_("lentils"),    0.00, 0.15, 0.12),
    (N_("onion"),      0.00, 0.30, 0.30), (N_("garlic"),     0.00, 0.25, 0.25),
    (N_("chilli"),     0.00, 0.10, 0.12), (N_("prawns"),     0.00, 0.08, 0.10),
    (N_("mushroom"),   0.00, 0.12, 0.15), (N_("courgette"),  0.00, 0.15, 0.18),
]

# foods always eaten together: the inseparability trap
INSEPARABLES = [(N_("garlic"), N_("onion"))]

# (name, lag in hours, amplitude in pain points)
DEFAULT_CULPRITS = [(N_("cheese"), 26.0, 2.2), (N_("wine"), 6.0, 2.0),
                    (N_("chilli"), 3.0, 2.4)]


def _food(name):
    """Translated, normalized name of a catalogue food."""
    return normalize(_(name))


def response(delays, center, width=7.0):
    """Pain profile triggered by a meal, as a function of the elapsed delay."""
    u = (np.asarray(delays, float) - center) / width
    r = 0.5 * (1.0 + np.cos(np.pi * u))
    r[np.abs(u) >= 1.0] = 0.0
    r[np.asarray(delays, float) < 0] = 0.0
    return r


def generate(n_days=28, culprits=None, seed=0, noise=1.0,
            circadian_amplitude=1.2, autocorr=0.45, base=2.5,
            reading_hours=None, pain_at_meals=True):
    """→ (observations, meals, truth) in the format expected by model.analyze.

    `culprits` holds source (English) food names, translated like CATALOGUE.
    """
    rng = np.random.default_rng(seed)
    if culprits is None:
        culprits = DEFAULT_CULPRITS
    effects = {_food(name): (lag, amp) for name, lag, amp in culprits}

    # -- meals ---------------------------------------------------------
    meals = []
    for j in range(n_days):
        for m, (h, _slot) in enumerate(MOMENTS):
            t = j * 24 + h + rng.normal(0, 0.4)
            chosen = [_food(name) for name, *p in CATALOGUE
                      if rng.random() < p[m]]
            for a, b in INSEPARABLES:
                a, b = _food(a), _food(b)
                if a in chosen or b in chosen:
                    chosen = sorted(set(chosen) | {a, b})
            if not chosen:
                chosen = [_food(N_("bread"))]
            meals.append((t, sorted(chosen)))
    meals.sort(key=lambda r: r[0])

    # -- pain reading times --------------------------------------------
    #  `reading_hours`: readings at fixed hours, INDEPENDENT of the meals.
    #  This is what makes it possible to tell "12 h after breakfast" from
    #  "in the evening"; without them, the two are the same column.
    instants = [t for t, _ in meals] if pain_at_meals else []
    for h in (reading_hours or []):
        instants += [j * 24 + h + rng.normal(0, 0.3) for j in range(n_days)]
    t_obs = np.sort(np.array(instants, dtype=float))
    hours = t_obs % 24.0

    signal = np.full(len(t_obs), base, dtype=float)
    signal += circadian_amplitude * np.sin(2 * np.pi * (hours - 10) / 24)

    for name, (lag, amp) in effects.items():
        for t_r, items in meals:
            if name in items:
                signal += amp * response(t_obs - t_r, lag)

    e = rng.normal(0, noise, len(t_obs))
    for i in range(1, len(e)):
        e[i] += autocorr * e[i - 1]
    y = np.clip(np.round(signal + e), 0, 10)

    observations = [(t_obs[i], hours[i], float(y[i])) for i in range(len(t_obs))]
    truth = {_food(n): (lag, amp) for n, lag, amp in culprits}
    return observations, meals, truth


# ══════════════════════════════════════════════════════════════════

def _one_diary(params):
    """One replicate (duration, protocol, seed) → raw metrics. Must live at
    module level to be picklable by ProcessPoolExecutor."""
    n_days, hours, threshold, n_replicates, seed = params
    obs, meals, truth = generate(n_days=n_days, seed=seed,
                                 reading_hours=hours)
    res = analyze(obs, meals, n_replicates=n_replicates, threshold=threshold,
                   seed=seed)
    if "error" in res:
        return None
    blocks, members, true_foods = res["blocks"], res["members"], set(truth)
    selected = [blocks[b] for b in res["selected"]]
    found, vp, errs = set(), 0, []
    for name in selected:
        inter = set(members[name]) & true_foods
        if inter:
            vp += 1
            found |= inter
            lag = res["lags"][blocks.index(name)]
            if not np.isnan(lag):
                errs.append(abs(lag - np.mean([truth[a][0] for a in inter])))
    return (len(found) / len(true_foods),
            vp / len(selected) if selected else 1.0,
            errs, len(obs),
            [n for n in selected if not set(members[n]) & true_foods])


def evaluate_parallel(n_seeds=12, n_days=28, reading_hours=None,
                      n_replicates=120, threshold=None, procs=None):
    """Same measure as `evaluate`, spread over the available cores."""
    from concurrent.futures import ProcessPoolExecutor

    from model import STABILITY_THRESHOLD
    threshold = STABILITY_THRESHOLD if threshold is None else threshold
    tasks = [(n_days, reading_hours, threshold, n_replicates, s)
              for s in range(n_seeds)]
    with ProcessPoolExecutor(max_workers=procs) as ex:
        outputs = [r for r in ex.map(_one_diary, tasks) if r]
    if not outputs:
        return {"recall": 0.0, "precision": 0.0, "err_lag_h": float("nan"),
                "n_obs": 0, "n_seeds": 0, "false_positives": {}}
    errs = [e for s in outputs for e in s[2]]
    false_pos = {}
    for s in outputs:
        for n in s[4]:
            false_pos[n] = false_pos.get(n, 0) + 1
    return {"recall": float(np.mean([s[0] for s in outputs])),
            "precision": float(np.mean([s[1] for s in outputs])),
            "err_lag_h": float(np.median(errs)) if errs else float("nan"),
            "n_obs": outputs[0][3], "n_seeds": len(outputs),
            "false_positives": dict(sorted(false_pos.items(), key=lambda x: -x[1]))}


def evaluate(n_seeds=12, n_days=28, n_replicates=120, threshold=None,
            reading_hours=None, **kw):
    """Precision / recall / lag error over n_seeds diaries."""
    from model import STABILITY_THRESHOLD
    threshold = STABILITY_THRESHOLD if threshold is None else threshold
    recalls, precisions, lag_err, false_pos, n_obs = [], [], [], {}, 0
    for s in range(n_seeds):
        obs, meals, truth = generate(n_days=n_days, seed=s,
                                     reading_hours=reading_hours, **kw)
        n_obs = len(obs)
        res = analyze(obs, meals, n_replicates=n_replicates, threshold=threshold, seed=s)
        if "error" in res:
            continue
        blocks, members, true_foods = res["blocks"], res["members"], set(truth)
        selected = [blocks[b] for b in res["selected"]]
        found, vp = set(), 0
        for name in selected:
            inter = set(members[name]) & true_foods
            if inter:
                vp += 1
                found |= inter
                lag = res["lags"][blocks.index(name)]
                if not np.isnan(lag):
                    lag_err.append(abs(lag - np.mean([truth[a][0] for a in inter])))
            else:
                false_pos[name] = false_pos.get(name, 0) + 1
        recalls.append(len(found) / len(true_foods))
        precisions.append(vp / len(selected) if selected else 1.0)
    return {
        "recall": float(np.mean(recalls)), "precision": float(np.mean(precisions)),
        "err_lag_h": float(np.median(lag_err)) if lag_err else float("nan"),
        "false_positives": dict(sorted(false_pos.items(), key=lambda x: -x[1])),
        "n_seeds": len(recalls), "n_obs": n_obs,
    }


if __name__ == "__main__":
    import sys
    n_days = int(sys.argv[1]) if len(sys.argv) > 1 else 28
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    r = evaluate(n_seeds=n_seeds, n_days=n_days)
    print(f"{n_days:3d} days | recall {r['recall']:.0%} | "
          f"precision {r['precision']:.0%} | lag error "
          f"{r['err_lag_h']:.1f} h | {r['n_seeds']} draws")
    if r["false_positives"]:
        print("   false positives:", r["false_positives"])


def write_diary(path, n_days=42, seed=0,
                   reading_hours=(7, 10, 13, 16, 19, 22)):
    """Write a synthetic diary. The format follows the extension: .ods for a
    LibreOffice Calc workbook, CSV otherwise."""
    from datetime import datetime, timedelta, timezone

    obs, meals, truth = generate(n_days=n_days, seed=seed,
                                 reading_hours=list(reading_hours))
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    by_time = {round(t, 4): a for t, a in meals}
    names = {8.0: _("breakfast"), 12.5: _("lunch"), 19.0: _("dinner")}

    rows = []
    for t, _h, d in obs:
        items = by_time.get(round(t, 4), [])
        slot = min(names, key=lambda k: abs((t % 24) - k))
        rows.append((start + timedelta(hours=float(t)),
                       names[slot] if items else "",
                       "; ".join(items), int(d)))
    rows.sort(key=lambda x: x[0])

    table = [[_("date"), _("time"), _("meal"), _("foods"), _("pain")]]
    table += [[format_date(ts, format='short', locale=LANG_CODE), ts.strftime("%H:%M"), slot, items, d]
              for ts, slot, items, d in rows]

    if path.lower().endswith(".ods"):
        from spreadsheet import write_ods
        write_ods(path, table, sheet=_("diary"))
    else:
        import csv as _csv
        with open(path, "w", encoding="utf-8", newline="") as f:
            _csv.writer(f).writerows(table)
    return truth


write_csv = write_diary          # backwards compatibility
