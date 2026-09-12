#!/usr/bin/env python3
# gutcheck - detection of pain-triggering foods, with unknown time lag
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
Detection of pain-triggering foods, with unknown time lag
and several possible culprits.

  python gutcheck.py journal.csv        analyze a journal (CSV)
  python gutcheck.py journal.ods [name] analyze a LibreOffice Calc workbook
                                    ([name] = sheet, defaults to the 1st)
  python gutcheck.py --validate         measure performance on synthetic
                                    data with known culprits
  python gutcheck.py --example f.csv    write a synthetic test journal
  python gutcheck.py --example f.ods    same, in LibreOffice Calc format
  python gutcheck.py --format-help      expected file format

CSV format: date,time,meal,foods,pain
  - `foods`: separated by ";"; leave blank for a row that only records
    pain (strongly recommended: recording pain ALSO between meals is
    what makes it possible to separate the lag from the meal time);
  - `pain`:  0-10; leave blank for a meal with no pain reading;
  - `time`:  HH:MM, defaults to 12:00.
"""

import csv
import sys
from datetime import datetime

import numpy as np
from babel.dates import parse_date, parse_time

from i18n import LANG_CODE, _, n_
from model import MIN_OCCURRENCES, STABILITY_THRESHOLD, analyze, normalize
from spreadsheet import normalize_header, read_ods

COLUMNS = (_("date"), _("time"), _("meal"), _("foods"), _("pain"))


def read_lines(file, sheet=None):
    """
    Read a diary, whatever its format, and return dictionaries
    column → text. Headers are normalized ("Pain (0-10)" →
    "pain") to tolerate the formatting of a real spreadsheet.

    Args:
        file (str): path to the diary file (CSV or ODS)
        sheet (str, optional): sheet name for ODS files; defaults to the first sheet

    Returns:
        list: list of dictionaries representing each row in the diary
    """
    if file.lower().endswith((".ods", ".fods")):
        return read_ods(file, sheet)
    with open(file, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            headers = [normalize_header(x) for x in next(reader)]
            replacements = {
                normalize_header(_("date")): "date",
                normalize_header(_("time")): "time",
                normalize_header(_("foods")): "foods",
                normalize_header(_("meal")): "meal",
                normalize_header(_("pain")): "pain",
            }

            headers = [replacements.get(x, x) for x in headers]
        except StopIteration:
            return []
        return [dict(zip(headers, line + [""] * (len(headers) - len(line))))
                for line in reader if any(x.strip() for x in line)]


def load_diary(file, sheet=None):
    """ Load a diary from CSV or LibreOffice Calc, and return
    observations (pain readings), meals (foods eaten), and warnings.
    Each observation is a tuple (hours_since_first, hour_of_day, pain_value).
    Each meal is a tuple (hours_since_first, list_of_foods).

    Args:
        file (str): path to the diary file (CSV or ODS)
        sheet (str, optional): sheet name for ODS files; defaults to the first sheet
    Returns:
        tuple: (observations, meals, warnings)
            observations: list of tuples (hours_since_first, hour_of_day, pain_value)
            meals: list of tuples (hours_since_first, list_of_foods)
            warnings: list of warning messages
    """
    lines, warnings, no_time = [], [], 0
    raw = read_lines(file, sheet)
    missing = [c for c in ("date", "foods", "pain")
                  if raw and c not in raw[0]]
    if missing:
        warnings.append(_("missing column(s): %(missing)s — expected: %(expected)s")
                        % {"missing": ', '.join(missing),
                           "expected": ', '.join(COLUMNS)})
    for num, line in enumerate(raw, start=2):
        txt_date = (line.get("date") or "").strip()
        if not txt_date:
            continue
        try:
            d = parse_date(txt_date, locale=LANG_CODE)
        except ValueError:
            warnings.append(_("line {}: date '{}' ignored").format(num, txt_date))
            continue

        txt_time = (line.get("time") or "").strip()
        if not txt_time:
            no_time += 1
            txt_time = "12:00"
        try:
            h = parse_time(txt_time, locale=LANG_CODE)
        except ValueError:
            warnings.append(_("line {}: time '{}' unreadable, "
                              "12:00 assumed").format(num, txt_time))
            h = parse_time("12:00", locale=LANG_CODE)
        ts = datetime.combine(d, h)

        foods = [normalize(a) for a in (line.get("foods") or "").split(";")]
        foods = sorted({a for a in foods if a})

        txt_pain = (line.get("pain") or "").strip()
        pain_value = None
        if txt_pain:
            try:
                pain_value = float(txt_pain.replace(",", "."))
            except ValueError:
                warnings.append(_("line {}: pain '{}' ignored").format(
                    num, txt_pain))
        lines.append((ts, foods, pain_value))

    if no_time:
        warnings.append(n_("%(n)s line without a time → 12:00 assumed; "
                           "a wrong time degrades the lag estimate",
                           "%(n)s lines without a time → 12:00 assumed; "
                           "a wrong time degrades the lag estimate",
                           no_time) % {"n": no_time})
    if not lines:
        return [], [], warnings + [_("no usable line")]

    #  Chronological sort: the AR(1) whitening and the splitting into blocks of
    #  days assume ordered readings, and nothing guarantees the CSV is sorted.
    lines.sort(key=lambda x: x[0])
    origin = lines[0][0]
    to_hours = lambda t: (t - origin).total_seconds() / 3600.0

    meals = [(to_hours(t), a) for t, a, _p in lines if a]
    observations = [(to_hours(t), t.hour + t.minute / 60.0, d)
                    for t, _f, d in lines if d is not None]
    return observations, meals, warnings + suspicious_names(meals)


def suspicious_names(meals, max_distance=1):
    """Report food names that are nearly identical (likely typos)."""
    names = sorted({a for _, items in meals for a in items})
    suspects = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if abs(len(a) - len(b)) > max_distance or a[0] != b[0]:
                continue
            if _distance(a, b) <= max_distance:
                suspects.append(_("'{}' and '{}': the same food?").format(a, b))
    return suspects


def _distance(a, b):
    """ Levenshtein distance (edit distance) between two strings.

    Args:
        a (str): first string
        b (str): second string

    Returns:
        int: Levenshtein distance between a and b
    """
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(prev[j] + 1, current[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = current
    return prev[-1]


def display(res, warnings):
    if warnings:
        print(_("\n  Warning:"))
        for a in warnings:
            print(f"    · {a}")

    if "error" in res:
        print(_("\n  Analysis impossible: %(error)s\n") % {"error": res["error"]})
        return

    print(f"\n{'='*72}")
    print(_("  %(days)s days · %(obs)s pain entries "
            "· %(nb)s foods analyzed "
            "(autocorrelation removed: rho = %(rho).2f)") % {
                "days": res['n_days'],
                "obs": res['n_observations'],
                "nb": len(res['blocks']),
                "rho": res['rho_ar1'],
    })
    print(f"{'='*72}\n")

    order = np.argsort(-res["frequencies"])
    threshold = res["threshold"]
    selected = set(res["selected"].tolist())

    print(f"  {_('Food'):<24} {_('Stability'):>10} {_('Lag'):>9} "
                f"{_('Peak'):>7} {_('Effect'):>7}  {_('n meals'):>7}")
    print(f"  {'-'*24} {'-'*10} {'-'*9} {'-'*7} {'-'*7}  {'-'*7}")
    for i in order[:12]:
        f = res["frequencies"][i]
        if f < 0.15:
            break
        lag = res["lags"][i]
        mark = "◆" if i in selected else " "
        lag_txt = f"{lag:4.0f} h" if not np.isnan(lag) else "   –"
        peak_txt = f"+{res['peaks'][i]:.1f}" if res["peaks"][i] > 0 else "   –"
        effect_txt = f"+{res['effects'][i]:.2f}" if res["effects"][i] > 0 else "   –"
        print(f" {mark}{res['blocks'][i]:<24} {f:>9.0%} {lag_txt:>9} "
              f"{peak_txt:>7} {effect_txt:>7}  {res['occurrences'][i]:>7}")

    print(_("\n  ◆ = selected (stability ≥ %(threshold).0f%%)") % {
        "threshold": threshold * 100})
    print(_("  Stability : fraction of the day-block resamplings in which the"))
    print(_("              food is selected, all foods competing."))
    print(_("  Lag       : delay of the pain PEAK after ingestion."))
    print(_("  Peak      : pain points added at the top, for one intake."))
    print(_("  Effect    : average attributable pain points over the whole"))
    print(_("              diary (≈ expected gain if the food is removed)."))

    multiple = [b for b in res["blocks"] if "+" in b]
    if multiple:
        print(_("\n  INSEPARABLE foods (always eaten together — no observational"))
        print(_("  data can tell them apart):"))
        for b in multiple:
            print(f"    · {b.replace('+', ' + ')}")

    if res["undetectable"]:
        print(_("\n  UNDETECTABLE — eaten too regularly: there is no variation"))
        print(_("  that could either incriminate or clear them."))
        print(f"    {', '.join(res['undetectable'])}")
        print(_("  The only way to test them is to remove them temporarily."))

    if res["discarded"]:
        print(_("\n  Discarded (< %(min)s occurrences): %(list)s") % {
            "min": MIN_OCCURRENCES,
            "list": ', '.join(res['discarded'][:15])
                    + ('…' if len(res['discarded']) > 15 else ''),
        })

    top = [res["blocks"][i] for i in order if res["frequencies"][i] >= threshold][:3]
    print(f"\n{'-'*72}")
    if top:
        print(_("  Next step — observation alone doesn't prove causality."))
        print(_("  Remove %(food)s for 2 weeks while keeping the journal,") % {
                    "food": top[0],
        })
        print(_("  then reintroduce it. One food at a time."))
        if len(top) > 1:
            print(_("  Next candidates: %(list)s.") % {
                      "list": ', '.join(top[1:]),
            })
    else:
        print(_("  No food stands out consistently."))
        print(_("  Possible causes: journal too short, effect genuinely weak, or"))
        print(_("  food eaten almost every day (no \"without\" days"))
        print(_("  to compare against). Extend the journal or vary your diet."))
    print(f"{'-'*72}\n")


FORMAT_HELP = _("""
Two formats accepted: CSV, or LibreOffice Calc spreadsheet (.ods).

Expected columns — date,time,meal,foods,pain

  date,time,meal,foods,pain
  2026-08-31,08:00,breakfast,eggs; bread; coffee,2
  2026-08-31,11:00,,,4              ← pain reading alone
  2026-08-31,12:30,lunch,rice; chicken; vegetables,4
  2026-08-31,19:00,dinner,soup; cheese,

Input tolerances:
  - headers  : case, accents and suffixes ignored ("Pain (0-10)" works too);
  - dates    : 2026-08-31, 08/31/2026, 08.31.2026;
  - times    : 08:00, 8:00, 08:00:00, 8:30am;
  - in an .ods file, date/time cells typed by Calc are read at their
    actual value, not their local display.

Two tips that matter more than the algorithm:

  1. Record pain OUTSIDE of meals (empty foods columns), every
     3-4 hours. If pain is only noted at meals, "12 h offset
     after breakfast" and "in the evening" are the same thing: nothing
     lets you tell them apart.

  2. VARY your diet. A food eaten every day without exception is
     undetectable no matter its effect: there is no comparison
     day available.
""")


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if args else 1)

    if args[0] == "--format-help":
        print(FORMAT_HELP)
        return

    if args[0] == "--example":
        from simu import write_diary
        path = args[1] if len(args) > 1 else "example_diary.csv"
        n_days = int(args[2]) if len(args) > 2 else 42
        truth = write_diary(path, n_days=n_days)
        print(_("\n  Synthetic diary written to %(path)s (%(days)s days).") % {
            "path": path, "days": n_days})
        print(_("  Culprits actually injected (to be found again):"))
        for name, (lag, amp) in truth.items():
            print(_("    · %(name)-12s lag %(lag)4.0f h, amplitude %(amp).1f") % {
                "name": name, "lag": lag, "amp": amp})
        print(_("\n  Try:  python gutcheck.py %(path)s\n") % {"path": path})
        return

    if args[0] == "--validate":
        from simu import evaluate_parallel
        durations = [int(a) for a in args[1:]] or [21, 28, 42, 56, 84]
        print(_("\n  Validation on synthetic diaries with KNOWN culprits"))
        print(_("  3 culprits among ~32 foods, lags 3 h / 6 h / 26 h,"))
        print(_("  noisy and autocorrelated pain, foods tied to the time of"))
        print(_("  the meal, one pair of inseparable foods.\n"))
        protocols = [(_("pain recorded at meals only"), None),
                     (_("+ 6 readings/day outside meals"),
                      [7, 10, 13, 16, 19, 22])]
        for title, hours in protocols:
            print(f"  {title}")
            print(f"    {_('days'):>6} {_('readings'):>8} {_('recall'):>8} "
                  f"{_('precision'):>10} {_('lag err.'):>14}")
            print(f"    {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*14}")
            for j in durations:
                r = evaluate_parallel(n_seeds=12, n_days=j,
                                      reading_hours=hours)
                print(f"    {j:>6} {r['n_obs']:>8} {r['recall']:>7.0%} "
                      f"{r['precision']:>10.0%} {r['err_lag_h']:>12.1f} h",
                      flush=True)
            print()
        print(_("  recall    : share of the true culprits that were selected"))
        print(_("  precision : share of the selected foods that really are "
                "culprits\n"))
        return

    sheet = args[1] if len(args) > 1 else None
    observations, meals, warnings = load_diary(args[0], sheet)
    if len(observations) < 10 or len(meals) < 5:
        print(_("\n  Diary too short: %(obs)s pain entries, "
                "%(meals)s meals.") % {
                    "obs": len(observations),
                    "meals": len(meals),
        })
        print(_("  You need at least about ten days. See --format-help.\n"))
        sys.exit(1)

    res = analyze(observations, meals, threshold=STABILITY_THRESHOLD)
    display(res, warnings)


if __name__ == "__main__":
    main()
