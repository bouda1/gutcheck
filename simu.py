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
Générateur de journaux synthétiques à coupables CONNUS, et évaluation de
l'algorithme dessus. C'est la seule manière de savoir s'il fonctionne : sur
données réelles, la vérité n'est pas observable.

Le générateur reproduit exprès les pièges du problème réel :
  - rythme circadien (douleur plus forte le soir) ;
  - aliments liés à un moment de la journée (le café est au petit-déjeuner) —
    donc confusion possible entre « aliment » et « heure » ;
  - douleur autocorrélée d'un repas au suivant ;
  - aliments toujours consommés ensemble (indissociables) ;
  - plusieurs coupables simultanés, à décalages différents.
"""

import numpy as np
from babel.dates import format_date

from i18n import _
from model import analyze, normalize

MOMENTS = [(8.0, _("breakfast")), (12.5, _("lunch")), (19.0, _("dinner"))]

CATALOGUE = [
    # (nom, proba au p-dej, proba au dej, proba au diner)
    ("cafe",       0.90, 0.20, 0.02), ("pain",       0.75, 0.35, 0.30),
    ("beurre",     0.60, 0.05, 0.10), ("confiture",  0.40, 0.00, 0.00),
    ("lait",       0.45, 0.02, 0.05), ("cereales",   0.30, 0.00, 0.00),
    ("oeufs",      0.25, 0.10, 0.10), ("banane",     0.20, 0.15, 0.05),
    ("yaourt",     0.20, 0.15, 0.20), ("jus_orange", 0.25, 0.05, 0.02),
    ("riz",        0.00, 0.30, 0.25), ("pates",      0.00, 0.30, 0.20),
    ("poulet",     0.00, 0.30, 0.20), ("boeuf",      0.00, 0.20, 0.15),
    ("poisson",    0.00, 0.15, 0.20), ("legumes",    0.02, 0.55, 0.55),
    ("salade",     0.00, 0.35, 0.30), ("tomate",     0.00, 0.30, 0.25),
    ("fromage",    0.05, 0.35, 0.30), ("pomme_terre",0.00, 0.25, 0.25),
    ("huile",      0.02, 0.45, 0.45), ("vin",        0.00, 0.15, 0.30),
    ("chocolat",   0.10, 0.15, 0.20), ("pomme",      0.10, 0.20, 0.15),
    ("soupe",      0.00, 0.05, 0.35), ("lentilles",  0.00, 0.15, 0.12),
    ("oignon",     0.00, 0.30, 0.30), ("ail",        0.00, 0.25, 0.25),
    ("piment",     0.00, 0.10, 0.12), ("crevettes",  0.00, 0.08, 0.10),
    ("champignon", 0.00, 0.12, 0.15), ("courgette",  0.00, 0.15, 0.18),
]

# aliments toujours ensemble : piège d'indissociabilité
INSEPARABLES = [("ail", "oignon")]


def reponse(delais, centre, largeur=7.0):
    """Profil de douleur déclenché par un repas, en fonction du délai écoulé."""
    u = (np.asarray(delais, float) - centre) / largeur
    r = 0.5 * (1.0 + np.cos(np.pi * u))
    r[np.abs(u) >= 1.0] = 0.0
    r[np.asarray(delais, float) < 0] = 0.0
    return r


def generer(n_jours=28, coupables=None, graine=0, bruit=1.0,
            amplitude_circadienne=1.2, autocorr=0.45, base=2.5,
            heures_releve=None, douleur_aux_repas=True):
    """→ (observations, repas, verite) au format attendu par model.analyze."""
    rng = np.random.default_rng(graine)
    if coupables is None:
        coupables = [("fromage", 26.0, 2.2), ("vin", 6.0, 2.0),
                     ("piment", 3.0, 2.4)]
    effets = {nom: (lag, amp) for nom, lag, amp in coupables}

    # -- repas ---------------------------------------------------------
    repas = []
    for j in range(n_jours):
        for m, (h, _nom) in enumerate(MOMENTS):
            t = j * 24 + h + rng.normal(0, 0.4)
            choisis = [nom for nom, *p in CATALOGUE if rng.random() < p[m]]
            for a, b in INSEPARABLES:
                if a in choisis or b in choisis:
                    choisis = sorted(set(choisis) | {a, b})
            if not choisis:
                choisis = ["pain"]
            repas.append((t, sorted(normalize(c) for c in choisis)))
    repas.sort(key=lambda r: r[0])

    # -- instants de relevé de la douleur ------------------------------
    #  `heures_releve` : relevés à heures fixes, INDÉPENDANTS des repas.
    #  C'est ce qui permet de distinguer « 12 h après le petit-déjeuner » de
    #  « le soir » ; sans eux, les deux sont la même colonne.
    instants = [t for t, _ in repas] if douleur_aux_repas else []
    for h in (heures_releve or []):
        instants += [j * 24 + h + rng.normal(0, 0.3) for j in range(n_jours)]
    t_obs = np.sort(np.array(instants, dtype=float))
    heures = t_obs % 24.0

    signal = np.full(len(t_obs), base, dtype=float)
    signal += amplitude_circadienne * np.sin(2 * np.pi * (heures - 10) / 24)

    for nom, (lag, amp) in effets.items():
        n = normalize(nom)
        for t_r, alims in repas:
            if n in alims:
                signal += amp * reponse(t_obs - t_r, lag)

    e = rng.normal(0, bruit, len(t_obs))
    for i in range(1, len(e)):
        e[i] += autocorr * e[i - 1]
    y = np.clip(np.round(signal + e), 0, 10)

    observations = [(t_obs[i], heures[i], float(y[i])) for i in range(len(t_obs))]
    verite = {normalize(n): (lag, amp) for n, lag, amp in coupables}
    return observations, repas, verite


# ══════════════════════════════════════════════════════════════════

def _un_journal(params):
    """Une réplique (durée, protocole, graine) → métriques brutes. Doit être
    au niveau module pour être picklable par ProcessPoolExecutor."""
    n_jours, heures, seuil, n_replicats, graine = params
    obs, repas, verite = generer(n_jours=n_jours, graine=graine,
                                 heures_releve=heures)
    res = analyze(obs, repas, n_replicats=n_replicats, seuil=seuil,
                   graine=graine)
    if "error" in res:
        return None
    blocs, membres, vrais = res["blocs"], res["membres"], set(verite)
    retenus = [blocs[b] for b in res["retenus"]]
    trouves, vp, errs = set(), 0, []
    for nom in retenus:
        inter = set(membres[nom]) & vrais
        if inter:
            vp += 1
            trouves |= inter
            lag = res["decalages"][blocs.index(nom)]
            if not np.isnan(lag):
                errs.append(abs(lag - np.mean([verite[a][0] for a in inter])))
    return (len(trouves) / len(vrais),
            vp / len(retenus) if retenus else 1.0,
            errs, len(obs),
            [n for n in retenus if not set(membres[n]) & vrais])


def evaluer_parallele(n_seeds=12, n_jours=28, heures_releve=None,
                      n_replicats=120, seuil=None, procs=None):
    """Même mesure que `evaluer`, répartie sur les coeurs disponibles."""
    from concurrent.futures import ProcessPoolExecutor

    from model import SEUIL_STABILITE
    seuil = SEUIL_STABILITE if seuil is None else seuil
    taches = [(n_jours, heures_releve, seuil, n_replicats, s)
              for s in range(n_seeds)]
    with ProcessPoolExecutor(max_workers=procs) as ex:
        sorties = [r for r in ex.map(_un_journal, taches) if r]
    if not sorties:
        return {"rappel": 0.0, "precision": 0.0, "err_lag_h": float("nan"),
                "n_obs": 0, "n_seeds": 0, "faux_positifs": {}}
    errs = [e for s in sorties for e in s[2]]
    faux = {}
    for s in sorties:
        for n in s[4]:
            faux[n] = faux.get(n, 0) + 1
    return {"rappel": float(np.mean([s[0] for s in sorties])),
            "precision": float(np.mean([s[1] for s in sorties])),
            "err_lag_h": float(np.median(errs)) if errs else float("nan"),
            "n_obs": sorties[0][3], "n_seeds": len(sorties),
            "faux_positifs": dict(sorted(faux.items(), key=lambda x: -x[1]))}


def evaluer(n_seeds=12, n_jours=28, n_replicats=120, seuil=None,
            heures_releve=None, **kw):
    """Précision / rappel / erreur de décalage sur n_seeds journaux."""
    from model import SEUIL_STABILITE
    seuil = SEUIL_STABILITE if seuil is None else seuil
    rappels, precisions, err_lag, faux, n_obs = [], [], [], {}, 0
    for s in range(n_seeds):
        obs, repas, verite = generer(n_jours=n_jours, graine=s,
                                     heures_releve=heures_releve, **kw)
        n_obs = len(obs)
        res = analyze(obs, repas, n_replicats=n_replicats, seuil=seuil, graine=s)
        if "error" in res:
            continue
        blocs, membres, vrais = res["blocs"], res["membres"], set(verite)
        retenus = [blocs[b] for b in res["retenus"]]
        trouves, vp = set(), 0
        for nom in retenus:
            inter = set(membres[nom]) & vrais
            if inter:
                vp += 1
                trouves |= inter
                lag = res["decalages"][blocs.index(nom)]
                if not np.isnan(lag):
                    err_lag.append(abs(lag - np.mean([verite[a][0] for a in inter])))
            else:
                faux[nom] = faux.get(nom, 0) + 1
        rappels.append(len(trouves) / len(vrais))
        precisions.append(vp / len(retenus) if retenus else 1.0)
    return {
        "rappel": float(np.mean(rappels)), "precision": float(np.mean(precisions)),
        "err_lag_h": float(np.median(err_lag)) if err_lag else float("nan"),
        "faux_positifs": dict(sorted(faux.items(), key=lambda x: -x[1])),
        "n_seeds": len(rappels), "n_obs": n_obs,
    }


if __name__ == "__main__":
    import sys
    n_jours = int(sys.argv[1]) if len(sys.argv) > 1 else 28
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    r = evaluer(n_seeds=n_seeds, n_jours=n_jours)
    print(f"{n_jours:3d} jours | rappel {r['rappel']:.0%} | "
          f"precision {r['precision']:.0%} | erreur decalage "
          f"{r['err_lag_h']:.1f} h | {r['n_seeds']} tirages")
    if r["faux_positifs"]:
        print("   faux positifs :", r["faux_positifs"])


def write_diary(chemin, n_jours=42, graine=0,
                   heures_releve=(7, 10, 13, 16, 19, 22)):
    """Écrit un journal synthétique. Le format suit l'extension : .ods pour un
    classeur LibreOffice Calc, CSV sinon."""
    from datetime import datetime, timedelta, timezone

    obs, repas, verite = generer(n_jours=n_jours, graine=graine,
                                 heures_releve=list(heures_releve))
    depart = datetime(2026, 1, 1, tzinfo=timezone.utc)
    par_instant = {round(t, 4): a for t, a in repas}
    noms = {8.0: _("breakfast"), 12.5: _("lunch"), 19.0: _("dinner")}

    lignes = []
    for t, _h, d in obs:
        alims = par_instant.get(round(t, 4), [])
        moment = min(noms, key=lambda k: abs((t % 24) - k))
        lignes.append((depart + timedelta(hours=float(t)),
                       noms[moment] if alims else "",
                       "; ".join(alims), int(d)))
    lignes.sort(key=lambda x: x[0])

    table = [[_("date"), _("time"), _("meal"), _("foods"), _("pain")]]
    table += [[format_date(ts, format='short'), ts.strftime("%H:%M"), moment, alims, d]
              for ts, moment, alims, d in lignes]

    if chemin.lower().endswith(".ods"):
        from spreadsheet import ecrire_ods
        ecrire_ods(chemin, table, feuille="journal")
    else:
        import csv as _csv
        with open(chemin, "w", encoding="utf-8", newline="") as f:
            _csv.writer(f).writerows(table)
    return verite


ecrire_csv = write_diary          # rétrocompatibilité
