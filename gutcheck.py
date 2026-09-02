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

from i18n import _
from model import MIN_OCCURRENCES, SEUIL_STABILITE, analyze, normalize
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
                _("time"): "time",
                _("foods"): "foods",
                _("meal"): "meal",
                _("pain"): "pain",
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
    lines, avert, no_time = [], [], 0
    raw = read_lines(file, sheet)
    missings = [c for c in ("date", "foods", "pain")
                  if raw and c not in raw[0]]
    if missings:
        avert.append(_("missing column(s): %(missing)s  — expected: %(await)s") % {"missing": ', '.join(missings), "await": ', '.join(COLUMNS)})
    for num, line in enumerate(raw, start=2):
        texte_date = (line.get("date") or "").strip()
        if not texte_date:
            continue
        try:
            d = parse_date(texte_date)
        except ValueError:
            avert.append(f"ligne {num} : date « {texte_date} » ignorée")
            continue

        texte_heure = (line.get("time") or "").strip()
        if not texte_heure:
            no_time += 1
            texte_heure = "12:00"
        try:
            h = parse_time(texte_heure)
        except ValueError:
            avert.append(f"ligne {num} : heure « {texte_heure} » → 12:00")
            h = parse_time("12:00")
        ts = datetime.combine(d, h)

        aliments = [normalize(a) for a in (line.get("foods") or "").split(";")]
        aliments = sorted({a for a in aliments if a})

        texte_douleur = (line.get("pain") or "").strip()
        douleur = None
        if texte_douleur:
            try:
                douleur = float(texte_douleur.replace(",", "."))
            except ValueError:
                avert.append(f"ligne {num} : douleur « {texte_douleur} » ignorée")
        lines.append((ts, aliments, douleur))

    if no_time:
        avert.append(f"{no_time} ligne(s) sans heure → 12:00 supposé ; "
                     "une heure fausse dégrade l'estimation du décalage")
    if not lines:
        return [], [], avert + ["aucune ligne exploitable"]

    #  Tri chronologique : le blanchiment AR(1) et le découpage en blocs de
    #  jours supposent des relevés ordonnés, or rien ne garantit que le CSV l'est.
    lines.sort(key=lambda x: x[0])
    origine = lines[0][0]
    en_heures = lambda t: (t - origine).total_seconds() / 3600.0

    repas = [(en_heures(t), a) for t, a, _ in lines if a]
    observations = [(en_heures(t), t.hour + t.minute / 60.0, d)
                    for t, _, d in lines if d is not None]
    return observations, repas, avert + noms_suspects(repas)


def noms_suspects(repas, distance_max=1):
    """Signale les noms d'aliments quasi identiques (fautes de frappe)."""
    noms = sorted({a for _, alims in repas for a in alims})
    suspects = []
    for i, a in enumerate(noms):
        for b in noms[i + 1:]:
            if abs(len(a) - len(b)) > distance_max or a[0] != b[0]:
                continue
            if _distance(a, b) <= distance_max:
                suspects.append(f"« {a} » et « {b} » : même aliment ?")
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


def display(res, avert):
    if avert:
        print(_("\n  Warning:"))
        for a in avert:
            print(f"    · {a}")

    if "error" in res:
        print(f"\n  Analyse impossible : {res['error']}\n")
        return

    print(f"\n{'='*72}")
    print(_("  %(days)s days · %(obs)s pain entries "
            "· %(nb)s foods analyzed "
            "(autocorrelation removed: rho = %(rho).2f)") % {
                "days": res['n_jours'],
                "obs": res['n_observations'],
                "nb": len(res['blocs']),
                "rho": res['rho_ar1'],
    })
    print(f"{'='*72}\n")

    ordre = np.argsort(-res["frequences"])
    seuil = res["seuil"]
    retenus = set(res["retenus"].tolist())

    print(f"  {_('Food'):<24} {_('Stability'):>10} {_('Lag'):>9} "
                f"{_('Peak'):>7} {_('Effect'):>7}  {_('n meals'):>7}")
    print(f"  {'-'*24} {'-'*10} {'-'*9} {'-'*7} {'-'*7}  {'-'*7}")
    for i in ordre[:12]:
        f = res["frequences"][i]
        if f < 0.15:
            break
        lag = res["decalages"][i]
        marque = "◆" if i in retenus else " "
        lag_txt = f"{lag:4.0f} h" if not np.isnan(lag) else "   –"
        pic_txt = f"+{res['pics'][i]:.1f}" if res["pics"][i] > 0 else "   –"
        eff_txt = f"+{res['effets'][i]:.2f}" if res["effets"][i] > 0 else "   –"
        print(f" {marque}{res['blocs'][i]:<24} {f:>9.0%} {lag_txt:>9} "
              f"{pic_txt:>7} {eff_txt:>7}  {res['occurrences'][i]:>7}")

    print(f"\n  ◆ = retenu (stabilité ≥ {seuil:.0%})")
    print("  Stabilité : fraction des ré-échantillonnages par blocs de jours où")
    print("              l'aliment est sélectionné, tous aliments en concurrence.")
    print("  Décalage  : délai du PIC de douleur après l'ingestion.")
    print("  Pic       : points de douleur ajoutés au sommet, pour une prise.")
    print("  Effet     : points de douleur moyens attribuables sur l'ensemble du")
    print("              journal (≈ gain attendu si l'aliment est supprimé).")

    multiples = [b for b in res["blocs"] if "+" in b]
    if multiples:
        print("\n  Aliments INDISSOCIABLES (toujours consommés ensemble — aucune")
        print("  donnée d'observation ne peut les départager) :")
        for b in multiples:
            print(f"    · {b.replace('+', ' + ')}")

    if res["indetectables"]:
        print("\n  INDÉTECTABLES — consommés de façon trop régulière : il n'existe")
        print("  aucune variation permettant de les mettre en cause ou hors de cause.")
        print(f"    {', '.join(res['indetectables'])}")
        print("  Le seul moyen de les tester est de les supprimer temporairement.")

    if res["ecartes"]:
        print(f"\n  Écartés (< {MIN_OCCURRENCES} occurrences) : "
              f"{', '.join(res['ecartes'][:15])}"
              f"{'…' if len(res['ecartes']) > 15 else ''}")

    tete = [res["blocs"][i] for i in ordre if res["frequences"][i] >= seuil][:3]
    print(f"\n{'-'*72}")
    if tete:
        print(_("  Next step — observation alone doesn't prove causality."))
        print(_("  Remove %(aliment)s for 2 weeks while keeping the journal,") % {
                    "aliment": tete[0],
        })
        print(_("  then reintroduce it. One food at a time."))
        if len(tete) > 1:
            print(_("  Next candidates: %(liste)s.") % {
                      "liste": ', '.join(tete[1:]),
            })
    else:
        print(_("  No food stands out consistently."))
        print(_("  Possible causes: journal too short, effect genuinely weak, or"))
        print(_("  food eaten almost every day (no \"without\" days"))
        print(_("  to compare against). Extend the journal or vary your diet."))
    print(f"{'-'*72}\n")


AIDE_FORMAT = _("""
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

    if not args or args[0] in ("-h", "--help", "--aide"):
        print(__doc__)
        sys.exit(0 if args else 1)

    if args[0] in ("--aide-format", "--format-help"):
        print(AIDE_FORMAT)
        return

    if args[0] in ("--example", "--exemple"):
        from simu import write_diary
        chemin = args[1] if len(args) > 1 else "journal_exemple.csv"
        n_jours = int(args[2]) if len(args) > 2 else 42
        verite = write_diary(chemin, n_jours=n_jours)
        print(f"\n  Journal synthétique écrit dans {chemin} ({n_jours} jours).")
        print("  Coupables réellement injectés (à retrouver) :")
        for nom, (lag, amp) in verite.items():
            print(f"    · {nom:<12} décalage {lag:>4.0f} h, amplitude {amp:.1f}")
        print(f"\n  Essayez :  python gutcheck.py {chemin}\n")
        return

    if args[0] == "--valider":
        from simu import evaluer_parallele
        durees = [int(a) for a in args[1:]] or [21, 28, 42, 56, 84]
        print("\n  Validation sur journaux synthétiques à coupables CONNUS")
        print("  3 coupables parmi ~32 aliments, décalages 3 h / 6 h / 26 h,")
        print("  douleur bruitée et autocorrélée, aliments liés au moment du")
        print("  repas, une paire d'aliments indissociables.\n")
        protocoles = [("douleur relevée aux repas seuls", None),
                      ("+ 6 relevés/jour hors repas", [7, 10, 13, 16, 19, 22])]
        for titre, heures in protocoles:
            print(f"  {titre}")
            print(f"    {'jours':>6} {'relevés':>8} {'rappel':>8} "
                  f"{'précision':>10} {'err. décalage':>14}")
            print(f"    {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*14}")
            for j in durees:
                r = evaluer_parallele(n_seeds=12, n_jours=j,
                                      heures_releve=heures)
                print(f"    {j:>6} {r['n_obs']:>8} {r['rappel']:>7.0%} "
                      f"{r['precision']:>10.0%} {r['err_lag_h']:>12.1f} h",
                      flush=True)
            print()
        print("  rappel    : part des vrais coupables retenus")
        print("  précision : part des aliments retenus qui sont vraiment coupables\n")
        return

    sheet = args[1] if len(args) > 1 else None
    observations, repas, avert = load_diary(args[0], sheet)
    if len(observations) < 10 or len(repas) < 5:
        print(_("\n  Diary too short: %(obs)s pain entries, "
                "%(meals)s meals.") % {
                    "obs": len(observations),
                    "meals": len(repas),
        })
        print(_("  You need at least about ten days. See --aide-format.\n"))
        sys.exit(1)

    res = analyze(observations, repas, seuil=SEUIL_STABILITE)
    display(res, avert)


if __name__ == "__main__":
    main()
