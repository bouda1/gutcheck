#!/usr/bin/env python3
"""
Détection d'aliments déclencheurs de douleur, avec décalage temporel inconnu
et plusieurs coupables possibles.

  python alim.py journal.csv        analyse un journal (CSV)
  python alim.py journal.ods [nom]  analyse un classeur LibreOffice Calc
                                    ([nom] = feuille, la 1re par défaut)
  python alim.py --valider          mesure les performances sur données
                                    synthétiques à coupables connus
  python alim.py --exemple f.csv    écrit un journal synthétique de test
  python alim.py --exemple f.ods    idem, au format LibreOffice Calc
  python alim.py --aide-format      format de fichier attendu

Format CSV : date,heure,repas,aliments,douleur
  - `aliments` : séparés par « ; » ; laisser vide pour une ligne qui ne relève
    que la douleur (fortement recommandé : relever la douleur AUSSI en dehors
    des repas est ce qui permet de séparer le décalage de l'heure du repas) ;
  - `douleur`  : 0-10 ; laisser vide pour un repas sans relevé de douleur ;
  - `heure`    : HH:MM, 12:00 par défaut.
"""

import csv
import re
import sys
from datetime import datetime

import numpy as np

from modele import analyser, normaliser, SEUIL_STABILITE, MIN_OCCURRENCES
from tableur import lire_ods, normaliser_entete

COLONNES_ATTENDUES = ("date", "heure", "repas", "aliments", "douleur")


def lire_lignes(fichier, feuille=None):
    """
    Lit un journal, quel que soit son format, et renvoie des dictionnaires
    colonne → texte. Les en-têtes sont normalisés (« Douleur (0-10) » →
    « douleur ») pour tolérer la mise en forme d'un vrai classeur.
    """
    if fichier.lower().endswith((".ods", ".fods")):
        return lire_ods(fichier, feuille)
    with open(fichier, encoding="utf-8-sig", newline="") as f:
        lecteur = csv.reader(f)
        try:
            entetes = [normaliser_entete(x) for x in next(lecteur)]
        except StopIteration:
            return []
        return [dict(zip(entetes, ligne + [""] * (len(entetes) - len(ligne))))
                for ligne in lecteur if any(x.strip() for x in ligne)]


FORMATS_DATE = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d.%m.%Y")


def analyser_date(texte):
    """Tolère les formats usuels : un classeur français écrit 31/08/2026."""
    for f in FORMATS_DATE:
        try:
            return datetime.strptime(texte, f)
        except ValueError:
            continue
    return None


def analyser_heure(texte):
    """→ (heure, minute) ; accepte 8:00, 08:00, 08:00:00, 08h00."""
    m = re.fullmatch(r"\s*(\d{1,2})\s*[:hH]\s*(\d{1,2})(?:\s*[:.]\s*\d{1,2})?\s*",
                     texte)
    if not m:
        return None
    h, mn = int(m.group(1)), int(m.group(2))
    return (h, mn) if 0 <= h < 24 and 0 <= mn < 60 else None


def charger_journal(fichier, feuille=None):
    """→ (observations, repas, avertissements) en heures depuis la 1re ligne."""
    lignes, avert, sans_heure = [], [], 0
    brutes = lire_lignes(fichier, feuille)
    manquantes = [c for c in ("date", "aliments", "douleur")
                  if brutes and c not in brutes[0]]
    if manquantes:
        avert.append(f"colonne(s) absente(s) : {', '.join(manquantes)} — "
                     f"attendu : {', '.join(COLONNES_ATTENDUES)}")
    for num, ligne in enumerate(brutes, start=2):
        texte_date = (ligne.get("date") or "").strip()
        if not texte_date:
            continue
        try:
            d = datetime.strptime(texte_date, "%Y-%m-%d")
        except ValueError:
            avert.append(f"ligne {num} : date « {texte_date} » ignorée")
            continue

        texte_heure = (ligne.get("heure") or "").strip()
        if not texte_heure:
            sans_heure += 1
            texte_heure = "12:00"
        try:
            h = datetime.strptime(texte_heure, "%H:%M")
        except ValueError:
            avert.append(f"ligne {num} : heure « {texte_heure} » → 12:00")
            h = datetime.strptime("12:00", "%H:%M")
        ts = d.replace(hour=h.hour, minute=h.minute)

        aliments = [normaliser(a) for a in (ligne.get("aliments") or "").split(";")]
        aliments = sorted({a for a in aliments if a})

        texte_douleur = (ligne.get("douleur") or "").strip()
        douleur = None
        if texte_douleur:
            try:
                douleur = float(texte_douleur.replace(",", "."))
            except ValueError:
                avert.append(f"ligne {num} : douleur « {texte_douleur} » ignorée")
        lignes.append((ts, aliments, douleur))

    if sans_heure:
        avert.append(f"{sans_heure} ligne(s) sans heure → 12:00 supposé ; "
                     "une heure fausse dégrade l'estimation du décalage")
    if not lignes:
        return [], [], avert + ["aucune ligne exploitable"]

    #  Tri chronologique : le blanchiment AR(1) et le découpage en blocs de
    #  jours supposent des relevés ordonnés, or rien ne garantit que le CSV l'est.
    lignes.sort(key=lambda x: x[0])
    origine = lignes[0][0]
    en_heures = lambda t: (t - origine).total_seconds() / 3600.0

    repas = [(en_heures(t), a) for t, a, _ in lignes if a]
    observations = [(en_heures(t), t.hour + t.minute / 60.0, d)
                    for t, _, d in lignes if d is not None]
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
    prec = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cour = [i]
        for j, cb in enumerate(b, 1):
            cour.append(min(prec[j] + 1, cour[j - 1] + 1, prec[j - 1] + (ca != cb)))
        prec = cour
    return prec[-1]


# ══════════════════════════════════════════════════════════════════

def afficher(res, avert):
    if avert:
        print("\n  Avertissements :")
        for a in avert:
            print(f"    · {a}")

    if "erreur" in res:
        print(f"\n  Analyse impossible : {res['erreur']}\n")
        return

    print(f"\n{'='*72}")
    print(f"  {res['n_jours']} jours · {res['n_observations']} relevés de douleur "
          f"· {len(res['blocs'])} aliments analysés "
          f"(autocorrélation retirée : rho = {res['rho_ar1']:.2f})")
    print(f"{'='*72}\n")

    ordre = np.argsort(-res["frequences"])
    seuil = res["seuil"]
    retenus = set(res["retenus"].tolist())

    print(f"  {'Aliment':<24} {'Stabilité':>10} {'Décalage':>9} "
          f"{'Pic':>7} {'Effet':>7}  {'n repas':>7}")
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
        print("  Étape suivante — l'observation seule ne prouve pas la causalité.")
        print(f"  Supprimez {tete[0]} pendant 2 semaines en gardant le journal,")
        print("  puis réintroduisez-le. Un seul aliment à la fois.")
        if len(tete) > 1:
            print(f"  Candidats suivants : {', '.join(tete[1:])}.")
    else:
        print("  Aucun aliment ne ressort de façon stable.")
        print("  Causes possibles : journal trop court, effet réel faible, ou")
        print("  aliment consommé presque tous les jours (pas de jours « sans »")
        print("  pour le comparer). Allongez le journal ou variez l'alimentation.")
    print(f"{'-'*72}\n")


AIDE_FORMAT = """
Deux formats acceptés : CSV, ou classeur LibreOffice Calc (.ods).

Colonnes attendues — date,heure,repas,aliments,douleur

  date,heure,repas,aliments,douleur
  2026-08-31,08:00,petit_dejeuner,oeufs; pain; cafe,2
  2026-08-31,11:00,,,4              ← relevé de douleur seul
  2026-08-31,12:30,dejeuner,riz; poulet; legumes,4
  2026-08-31,19:00,diner,soupe; fromage,

Tolérances de saisie :
  - en-têtes : casse, accents et suffixes ignorés (« Douleur (0-10) » convient) ;
  - dates    : 2026-08-31, 31/08/2026, 31.08.2026 ;
  - heures   : 08:00, 8:00, 08:00:00, 08h30 ;
  - dans un .ods, les cellules date/heure typées par Calc sont relues à leur
    valeur réelle, pas à leur affichage local.

Deux conseils qui pèsent plus que l'algorithme :

  1. Relevez la douleur EN DEHORS des repas (colonnes aliments vides), toutes
     les 3-4 h. Si la douleur n'est notée qu'aux repas, « décalage de 12 h
     après le petit-déjeuner » et « le soir » sont la même chose : rien ne
     permet de les distinguer.

  2. VARIEZ. Un aliment consommé tous les jours sans exception est
     indétectable, quel que soit son effet : il n'existe aucun jour de
     comparaison.
"""


def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "--aide"):
        print(__doc__)
        sys.exit(0 if args else 1)

    if args[0] == "--aide-format":
        print(AIDE_FORMAT)
        return

    if args[0] == "--exemple":
        from simu import ecrire_journal
        chemin = args[1] if len(args) > 1 else "journal_exemple.csv"
        n_jours = int(args[2]) if len(args) > 2 else 42
        verite = ecrire_journal(chemin, n_jours=n_jours)
        print(f"\n  Journal synthétique écrit dans {chemin} ({n_jours} jours).")
        print("  Coupables réellement injectés (à retrouver) :")
        for nom, (lag, amp) in verite.items():
            print(f"    · {nom:<12} décalage {lag:>4.0f} h, amplitude {amp:.1f}")
        print(f"\n  Essayez :  python alim.py {chemin}\n")
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

    feuille = args[1] if len(args) > 1 else None
    observations, repas, avert = charger_journal(args[0], feuille)
    if len(observations) < 10 or len(repas) < 5:
        print(f"\n  Journal trop court : {len(observations)} relevés de douleur, "
              f"{len(repas)} repas.")
        print("  Il en faut au moins une dizaine de jours. Voir --aide-format.\n")
        sys.exit(1)

    res = analyser(observations, repas, seuil=SEUIL_STABILITE)
    afficher(res, avert)


if __name__ == "__main__":
    main()
