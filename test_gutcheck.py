#!/usr/bin/env python3
# gutcheck - regression tests
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

"""Regression tests. Run: python test_gutcheck.py"""
import os
import sys
import tempfile

import numpy as np

from i18n import _

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gutcheck import load_diary
from model import (
    LAG_CENTERS,
    MIN_DELAY,
    MAX_SPAN,
    analyze,
    build_controls,
    build_exposition,
    fusionner_indissociables,
    nn_group_lasso,
    normalize,
    kernels_weights,
    prepare_groups,
    residualize,
)
from simu import generer

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
fails = []


def check(condition, message):
    """ Check a condition and print the result. If the condition is False, add the message to the fails list."""
    status = f"{GREEN}OK  {RESET}" if condition else f"{RED}FAIL{RESET}"
    print(f"  {status}  {message}")
    if not condition:
        fails.append(message)


print("\nnormalisation des noms")
check(normalize("  Café  ") == "cafe", "accents et espaces retirés")
check(normalize("Pomme-de-terre") == "pomme_de_terre", "séparateurs unifiés")
check(normalize("RIZ") == normalize("riz"), "casse ignorée")

print("\nnoyaux de décalage")
d = np.linspace(0, MAX_SPAN, 500)
w = kernels_weights(d)
check((w >= 0).all(), "poids toujours positifs")
check((kernels_weights([0.0, MIN_DELAY / 2]) == 0).all(),
         "aucune contribution à délai nul (pas de causalité inverse)")
interieur = d[(d >= MIN_DELAY) & (d <= LAG_CENTERS[-1])]
check(kernels_weights(interieur).sum(axis=1).min() > 0.2,
         "couverture continue : aucun délai sans noyau")
check(kernels_weights([MAX_SPAN * 2]).sum() == 0, "nul au-delà de la portée")

print("\nmatrice d'exposition")
X = build_exposition([10.0, 100.0], [(8.0, ["a"]), (95.0, ["b"])], ["a", "b"])
K = len(LAG_CENTERS)
check(X[0, :K].sum() > 0, "le repas 2 h avant expose l'aliment a")
check(X[0, K:].sum() == 0, "un repas postérieur n'expose rien")
check(X[1, K:].sum() > 0, "le repas 5 h avant expose l'aliment b")

print("\nrésidualisation (Frisch-Waugh-Lovell)")
rng = np.random.default_rng(0)
Z = build_controls(np.arange(40) * 6.0, (np.arange(40) * 6.0) % 24)
Xr = rng.normal(size=(40, 5))
yr = Z @ np.arange(1, 7) + Xr @ np.array([2.0, 0, 0, 0, 0]) + rng.normal(0, .1, 40)
y_res, X_res = residualize(yr, Xr, Z)
check(np.abs(Z.T @ y_res).max() < 1e-8, "y résiduel orthogonal aux contrôles")
check(np.abs(Z.T @ X_res).max() < 1e-8, "X résiduel orthogonal aux contrôles")

print("\ngroup-lasso positif")
n, G_blocs = 200, 6
Xg = rng.normal(size=(n, G_blocs * K))
beta_vrai = np.zeros(G_blocs * K)
beta_vrai[K:2 * K] = 1.5          # seul le groupe 1 est actif
yg = Xg @ beta_vrai + rng.normal(0, 0.5, n)
groupes = [np.arange(b * K, (b + 1) * K) for b in range(G_blocs)]
Gm, cm = Xg.T @ Xg / n, Xg.T @ yg / n
b = nn_group_lasso(Gm, cm, groupes, 0.15, np.full(G_blocs, np.sqrt(K)),
                   prepare_groups(Gm, groupes))
actifs = [g for g in range(G_blocs) if b[groupes[g]].max() > 0]
check(actifs == [1], f"seul le groupe planté est sélectionné (obtenu {actifs})")
check((b >= 0).all(), "contrainte de positivité respectée")

print("\nfusion des aliments indissociables")
repas = [(float(i), ["ail", "oignon"] + (["riz"] if i % 2 else [])) for i in range(10)]
blocs, membres = fusionner_indissociables(repas, ["ail", "oignon", "riz"])
check("ail+oignon" in blocs, "aliments toujours ensemble fusionnés")
check("riz" in blocs, "aliment indépendant non fusionné")

print("\nlecture CSV robuste")
contenu = """{date},{heure},{repas},{aliments},{douleur}
2026-03-02,19:00,diner,soupe,5
2026-03-01,,petit_dejeuner,Café; pain,2
2026-03-01,12:30,dejeuner,riz,
2026-03-01,15:00,,,4
2026-03-01,16:00,,,abc
pas-une-date,08:00,,,3
""".format(
       date=_("Date"),
       heure=_("Time"),
       repas=_("Meal"),
       aliments=_("Foods"),
       douleur=_("Pain (0-10)"),
   )

with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                 encoding="utf-8") as f:
    f.write(contenu)
    path = f.name
obs, rep, avert = load_diary(path)
os.unlink(path)
check([o[0] for o in obs] == sorted(o[0] for o in obs), "relevés triés")
check(len(rep) == 3, f"3 repas lus (obtenu {len(rep)})")
check(len(obs) == 3, f"3 relevés de douleur valides (obtenu {len(obs)})")
check(any("sans heure" in a for a in avert), "heure vide signalée en bloc")
check(abs((rep[1][0] - rep[0][0]) - 0.5) < 1e-9,
         "heure vide → 12:00 (30 min avant le déjeuner de 12:30)")
check(any("abc" in a for a in avert), "douleur illisible signalée")
check(any("pas-une-date" in a for a in avert), "date illisible signalée")
check(["cafe", "pain"] == rep[0][1], f"noms normalisés (obtenu {rep[0][1]})")

print("\nclasseur LibreOffice Calc (.ods)")
from spreadsheet import ecrire_ods, lire_feuille, normalize_header, read_ods

check(normalize_header("  Douleur (0-10) ") == "douleur",
         "en-tête décoré normalisé")
check(normalize_header("Aliments") == "aliments", "en-tête casse ignorée")

chemin_ods = os.path.join(tempfile.gettempdir(), "alim_test.ods")
ecrire_ods(chemin_ods, [
    ["Date", _("Time"), _("Meal"), _("Foods"), _("Pain (0-10)")],
    ["2026-03-01", "08:00", "petit_dejeuner", "Café; pain", 2],
    ["2026-03-01", "11:00", "", "", 4],
    ["2026-03-01", "12:30", "dejeuner", "riz; poulet", ""],
    ["2026-03-02", "19:00", "diner", "soupe", 5],
])
d = read_ods(chemin_ods)
check(len(d) == 4, f"4 lignes de données lues (obtenu {len(d)})")
check(d[0]["date"] == "2026-03-01",
         "cellule DATE typée relue en ISO (pas le texte affiché)")
check(d[0]["time"] == "08:00",
         f"cellule HEURE typée relue en HH:MM (obtenu {d[0]['time']!r})")
check(d[0]["pain"] == "2", "cellule numérique sans décimale parasite")
check(d[2]["pain"] == "", "cellule vide en fin de ligne = chaîne vide")

obs_o, rep_o, _ = load_diary(chemin_ods)
check(len(rep_o) == 3 and len(obs_o) == 3,
         "journal .ods chargé comme son équivalent CSV")
check(rep_o[0][1] == ["cafe", "pain"], "noms normalisés depuis le classeur")

#  Without a data style, Calc would display the internal value of a date
#  (number of days since the epoch) instead of the date.
import re as _re
import zipfile as _zip0

with _zip0.ZipFile(chemin_ods) as z:
    contenu_xml = z.read("content.xml").decode()
    entrees = z.namelist()
check("<number:date-style" in contenu_xml, "format d'affichage des dates déclaré")
check("<number:time-style" in contenu_xml, "format d'affichage des heures déclaré")
declares = set(_re.findall(r'<style:style style:name="([^"]+)"', contenu_xml))
utilises = set(_re.findall(r'table:style-name="([^"]+)"', contenu_xml))
check(utilises and utilises <= declares,
         f"tout style référencé est déclaré (utilisés {sorted(utilises)})")
check(_re.search(r'office:value-type="date"[^>]*>', contenu_xml) and
         all('table:style-name=' in c for c in
             _re.findall(r'<table:table-cell[^>]*office:value-type="(?:date|time)"[^>]*>',
                         contenu_xml)),
         "chaque cellule date/heure porte un style d'affichage")
check(entrees[0] == "mimetype", "mimetype en première entrée de l'archive")
check("styles.xml" in entrees, "styles.xml présent dans le paquet")

# compressed empty cells, as Calc actually does at the end of a sheet
import zipfile as _zip

contenu = ('<?xml version="1.0" encoding="UTF-8"?>'
           '<office:document-content'
           ' xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"'
           ' xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"'
           ' xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
           '<office:body><office:spreadsheet><table:table table:name="f">'
           '<table:table-row><table:table-cell office:value-type="string">'
           '<text:p>date</text:p></table:table-cell>'
           '<table:table-cell table:number-columns-repeated="3"/>'
           '<table:table-cell office:value-type="string">'
           '<text:p>douleur</text:p></table:table-cell></table:table-row>'
           '<table:table-row table:number-rows-repeated="1048576">'
           '<table:table-cell table:number-columns-repeated="16384"/>'
           '</table:table-row>'
           '</table:table></office:spreadsheet></office:body>'
           '</office:document-content>')
with _zip.ZipFile(chemin_ods, "w") as z:
    z.writestr("content.xml", contenu)
brut = lire_feuille(chemin_ods)
check(len(brut) == 1, f"lignes vides répétées non matérialisées "
                         f"(obtenu {len(brut)})")
check(brut[0] == ["date", "", "", "", "douleur"],
         f"colonnes vides intercalées développées (obtenu {brut[0]})")
os.unlink(chemin_ods)

print("\nformats de date et d'heure tolérés")
#from gutcheck import analyze_date, analyser_heure
#check(analyze_date("2026-08-31") is not None, "ISO")
#check(analyze_date("31/08/2026").month == 8, "jour/mois/année (français)")
#check(analyze_date("31.08.2026").day == 31, "séparateur point")
#check(analyze_date("pas-une-date") is None, "date invalide rejetée")
#check(analyser_heure("8:00") == (8, 0), "heure sans zéro initial")
#check(analyser_heure("08:00:00") == (8, 0), "heure avec secondes")
#check(analyser_heure("08h30") == (8, 30), "notation 08h30")
#check(analyser_heure("25:00") is None, "heure hors bornes rejetée")

print("\ndétection de bout en bout (56 jours, protocole recommandé)")
o, r, verite = generer(n_jours=56, graine=0,
                       heures_releve=[7, 10, 13, 16, 19, 22])
res = analyze(o, r, n_replicats=80, graine=0)
retenus = {res["blocs"][i] for i in res["retenus"]}
check(set(verite) <= retenus,
         f"les 3 coupables sont retenus (retenus : {sorted(retenus)})")
check(len(retenus - set(verite)) <= 1,
         f"au plus 1 faux positif (obtenu {sorted(retenus - set(verite))})")
for nom, (lag_vrai, _) in verite.items():
    est = res["decalages"][res["blocs"].index(nom)]
    check(abs(est - lag_vrai) <= 8,
             f"décalage de {nom} : {est:.0f} h estimé vs {lag_vrai:.0f} h réel")

print(f"\n{'-'*60}")
if fails:
    print(f"  {len(fails)} ÉCHEC(S)")
    sys.exit(1)
print("  tous les tests passent")
