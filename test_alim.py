#!/usr/bin/env python3
"""Tests de non-régression.  Exécution :  python test_alim.py"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import modele
from modele import (CENTRES_LAG, DELAI_MIN, PORTEE_MAX, analyser,
                    construire_controles, construire_exposition,
                    fusionner_indissociables, nn_group_lasso, normaliser,
                    poids_noyaux, preparer_blocs, residualiser)
from alim import charger_journal
from simu import generer

echecs = []


def verifier(condition, message):
    print(f"  {'ok  ' if condition else 'ÉCHEC'}  {message}")
    if not condition:
        echecs.append(message)


print("\nnormalisation des noms")
verifier(normaliser("  Café  ") == "cafe", "accents et espaces retirés")
verifier(normaliser("Pomme-de-terre") == "pomme_de_terre", "séparateurs unifiés")
verifier(normaliser("RIZ") == normaliser("riz"), "casse ignorée")

print("\nnoyaux de décalage")
d = np.linspace(0, PORTEE_MAX, 500)
w = poids_noyaux(d)
verifier((w >= 0).all(), "poids toujours positifs")
verifier((poids_noyaux([0.0, DELAI_MIN / 2]) == 0).all(),
         "aucune contribution à délai nul (pas de causalité inverse)")
interieur = d[(d >= DELAI_MIN) & (d <= CENTRES_LAG[-1])]
verifier(poids_noyaux(interieur).sum(axis=1).min() > 0.2,
         "couverture continue : aucun délai sans noyau")
verifier(poids_noyaux([PORTEE_MAX * 2]).sum() == 0, "nul au-delà de la portée")

print("\nmatrice d'exposition")
X = construire_exposition([10.0, 100.0], [(8.0, ["a"]), (95.0, ["b"])], ["a", "b"])
K = len(CENTRES_LAG)
verifier(X[0, :K].sum() > 0, "le repas 2 h avant expose l'aliment a")
verifier(X[0, K:].sum() == 0, "un repas postérieur n'expose rien")
verifier(X[1, K:].sum() > 0, "le repas 5 h avant expose l'aliment b")

print("\nrésidualisation (Frisch-Waugh-Lovell)")
rng = np.random.default_rng(0)
Z = construire_controles(np.arange(40) * 6.0, (np.arange(40) * 6.0) % 24)
Xr = rng.normal(size=(40, 5))
yr = Z @ np.arange(1, 7) + Xr @ np.array([2.0, 0, 0, 0, 0]) + rng.normal(0, .1, 40)
y_res, X_res = residualiser(yr, Xr, Z)
verifier(np.abs(Z.T @ y_res).max() < 1e-8, "y résiduel orthogonal aux contrôles")
verifier(np.abs(Z.T @ X_res).max() < 1e-8, "X résiduel orthogonal aux contrôles")

print("\ngroup-lasso positif")
n, G_blocs = 200, 6
Xg = rng.normal(size=(n, G_blocs * K))
beta_vrai = np.zeros(G_blocs * K)
beta_vrai[K:2 * K] = 1.5          # seul le groupe 1 est actif
yg = Xg @ beta_vrai + rng.normal(0, 0.5, n)
groupes = [np.arange(b * K, (b + 1) * K) for b in range(G_blocs)]
Gm, cm = Xg.T @ Xg / n, Xg.T @ yg / n
b = nn_group_lasso(Gm, cm, groupes, 0.15, np.full(G_blocs, np.sqrt(K)),
                   preparer_blocs(Gm, groupes))
actifs = [g for g in range(G_blocs) if b[groupes[g]].max() > 0]
verifier(actifs == [1], f"seul le groupe planté est sélectionné (obtenu {actifs})")
verifier((b >= 0).all(), "contrainte de positivité respectée")

print("\nfusion des aliments indissociables")
repas = [(float(i), ["ail", "oignon"] + (["riz"] if i % 2 else [])) for i in range(10)]
blocs, membres = fusionner_indissociables(repas, ["ail", "oignon", "riz"])
verifier("ail+oignon" in blocs, "aliments toujours ensemble fusionnés")
verifier("riz" in blocs, "aliment indépendant non fusionné")

print("\nlecture CSV robuste")
contenu = """date,heure,repas,aliments,douleur
2026-03-02,19:00,diner,soupe,5
2026-03-01,,petit_dejeuner,Café; pain,2
2026-03-01,12:30,dejeuner,riz,
2026-03-01,15:00,,,4
2026-03-01,16:00,,,abc
pas-une-date,08:00,,,3
"""
with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                 encoding="utf-8") as f:
    f.write(contenu)
    chemin = f.name
obs, rep, avert = charger_journal(chemin)
os.unlink(chemin)
verifier([o[0] for o in obs] == sorted(o[0] for o in obs), "relevés triés")
verifier(len(rep) == 3, f"3 repas lus (obtenu {len(rep)})")
verifier(len(obs) == 3, f"3 relevés de douleur valides (obtenu {len(obs)})")
verifier(any("sans heure" in a for a in avert), "heure vide signalée en bloc")
verifier(abs((rep[1][0] - rep[0][0]) - 0.5) < 1e-9,
         "heure vide → 12:00 (30 min avant le déjeuner de 12:30)")
verifier(any("abc" in a for a in avert), "douleur illisible signalée")
verifier(any("pas-une-date" in a for a in avert), "date illisible signalée")
verifier(["cafe", "pain"] == rep[0][1], f"noms normalisés (obtenu {rep[0][1]})")

print("\nclasseur LibreOffice Calc (.ods)")
from tableur import ecrire_ods, lire_feuille, lire_ods, normaliser_entete
verifier(normaliser_entete("  Douleur (0-10) ") == "douleur",
         "en-tête décoré normalisé")
verifier(normaliser_entete("Aliments") == "aliments", "en-tête casse ignorée")

chemin_ods = os.path.join(tempfile.gettempdir(), "alim_test.ods")
ecrire_ods(chemin_ods, [
    ["Date", "Heure", "Repas", "Aliments", "Douleur (0-10)"],
    ["2026-03-01", "08:00", "petit_dejeuner", "Café; pain", 2],
    ["2026-03-01", "11:00", "", "", 4],
    ["2026-03-01", "12:30", "dejeuner", "riz; poulet", ""],
    ["2026-03-02", "19:00", "diner", "soupe", 5],
])
d = lire_ods(chemin_ods)
verifier(len(d) == 4, f"4 lignes de données lues (obtenu {len(d)})")
verifier(d[0]["date"] == "2026-03-01",
         "cellule DATE typée relue en ISO (pas le texte affiché)")
verifier(d[0]["heure"] == "08:00",
         f"cellule HEURE typée relue en HH:MM (obtenu {d[0]['heure']!r})")
verifier(d[0]["douleur"] == "2", "cellule numérique sans décimale parasite")
verifier(d[2]["douleur"] == "", "cellule vide en fin de ligne = chaîne vide")

obs_o, rep_o, _ = charger_journal(chemin_ods)
verifier(len(rep_o) == 3 and len(obs_o) == 3,
         "journal .ods chargé comme son équivalent CSV")
verifier(rep_o[0][1] == ["cafe", "pain"], "noms normalisés depuis le classeur")

#  Sans style de données, Calc afficherait la valeur interne d'une date
#  (nombre de jours depuis l'époque) au lieu de la date.
import re as _re
import zipfile as _zip0
with _zip0.ZipFile(chemin_ods) as z:
    contenu_xml = z.read("content.xml").decode()
    entrees = z.namelist()
verifier("<number:date-style" in contenu_xml, "format d'affichage des dates déclaré")
verifier("<number:time-style" in contenu_xml, "format d'affichage des heures déclaré")
declares = set(_re.findall(r'<style:style style:name="([^"]+)"', contenu_xml))
utilises = set(_re.findall(r'table:style-name="([^"]+)"', contenu_xml))
verifier(utilises and utilises <= declares,
         f"tout style référencé est déclaré (utilisés {sorted(utilises)})")
verifier(_re.search(r'office:value-type="date"[^>]*>', contenu_xml) and
         all('table:style-name=' in c for c in
             _re.findall(r'<table:table-cell[^>]*office:value-type="(?:date|time)"[^>]*>',
                         contenu_xml)),
         "chaque cellule date/heure porte un style d'affichage")
verifier(entrees[0] == "mimetype", "mimetype en première entrée de l'archive")
verifier("styles.xml" in entrees, "styles.xml présent dans le paquet")

# cellules vides compressées, comme le fait réellement Calc en fin de feuille
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
verifier(len(brut) == 1, f"lignes vides répétées non matérialisées "
                         f"(obtenu {len(brut)})")
verifier(brut[0] == ["date", "", "", "", "douleur"],
         f"colonnes vides intercalées développées (obtenu {brut[0]})")
os.unlink(chemin_ods)

print("\nformats de date et d'heure tolérés")
from alim import analyser_date, analyser_heure
verifier(analyser_date("2026-08-31") is not None, "ISO")
verifier(analyser_date("31/08/2026").month == 8, "jour/mois/année (français)")
verifier(analyser_date("31.08.2026").day == 31, "séparateur point")
verifier(analyser_date("pas-une-date") is None, "date invalide rejetée")
verifier(analyser_heure("8:00") == (8, 0), "heure sans zéro initial")
verifier(analyser_heure("08:00:00") == (8, 0), "heure avec secondes")
verifier(analyser_heure("08h30") == (8, 30), "notation 08h30")
verifier(analyser_heure("25:00") is None, "heure hors bornes rejetée")

print("\ndétection de bout en bout (56 jours, protocole recommandé)")
o, r, verite = generer(n_jours=56, graine=0,
                       heures_releve=[7, 10, 13, 16, 19, 22])
res = analyser(o, r, n_replicats=80, graine=0)
retenus = {res["blocs"][i] for i in res["retenus"]}
verifier(set(verite) <= retenus,
         f"les 3 coupables sont retenus (retenus : {sorted(retenus)})")
verifier(len(retenus - set(verite)) <= 1,
         f"au plus 1 faux positif (obtenu {sorted(retenus - set(verite))})")
for nom, (lag_vrai, _) in verite.items():
    est = res["decalages"][res["blocs"].index(nom)]
    verifier(abs(est - lag_vrai) <= 8,
             f"décalage de {nom} : {est:.0f} h estimé vs {lag_vrai:.0f} h réel")

print(f"\n{'-'*60}")
if echecs:
    print(f"  {len(echecs)} ÉCHEC(S)")
    sys.exit(1)
print("  tous les tests passent")
