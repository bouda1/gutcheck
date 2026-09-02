#!/usr/bin/env python3
"""
Modèle à décalages distribués + sélection par stabilité.

Principe
--------
La douleur observée à l'instant t est la somme des contributions de TOUS les
repas passés. Chaque aliment agit avec un profil de décalage qui lui est propre,
décrit par une base de noyaux lisses (raised-cosine) couvrant 0-84 h.

Tous les aliments sont estimés SIMULTANÉMENT (donc plusieurs coupables
possibles, et chacun est jugé « à aliments concurrents constants »), sous deux
contraintes :
  - positivité : on cherche des déclencheurs, pas des protecteurs ;
  - parcimonie (lasso) : peu d'aliments sont réellement en cause.

La question « lesquels ? » n'est pas tranchée par une p-value (invalide ici :
tests multiples, décalage choisi a posteriori, douleur autocorrélée) mais par la
FRÉQUENCE DE SÉLECTION sous ré-échantillonnage par blocs de jours
(stability selection, Meinshausen & Bühlmann 2010).
"""

import unicodedata
from collections import defaultdict

import numpy as np

# ─── Base de décalages : centres et demi-largeurs, en heures ──────
CENTRES_LAG = np.array([2.0, 5.0, 9.0, 15.0, 24.0, 40.0, 60.0])
LARGEURS_LAG = np.array([3.0, 4.0, 5.0, 8.0, 12.0, 18.0, 24.0])
DELAI_MIN = 0.25          # un repas n'agit jamais sur une douleur simultanée
PORTEE_MAX = float(CENTRES_LAG[-1] + LARGEURS_LAG[-1])

# ─── Sélection par stabilité ──────────────────────────────────────
TAILLE_BLOC_JOURS = 3     # blocs contigus : préserve l'autocorrélation
FRACTION_SOUS_ECH = 0.5
N_REPLICATS = 200
N_LAMBDAS = 6
SEUIL_STABILITE = 0.70

MIN_OCCURRENCES = 3       # nb minimal de repas contenant l'aliment
SEUIL_JACCARD = 0.85      # au-delà : aliments indissociables, fusionnés


# ══════════════════════════════════════════════════════════════════
#  Normalisation des noms d'aliments
# ══════════════════════════════════════════════════════════════════

def normalize(nom):
    """ Normalize a food name to a canonical form: lowercase, no accents, unified separators.

    Args:
        nom (str): The food name to normalize.

    Returns:
        str: The normalized food name.
    """
    nom = unicodedata.normalize("NFKD", nom.strip().lower())
    nom = "".join(c for c in nom if not unicodedata.combining(c))
    return "_".join(nom.replace("-", " ").replace("_", " ").split())


# ══════════════════════════════════════════════════════════════════
#  Matrice d'exposition
# ══════════════════════════════════════════════════════════════════

def poids_noyaux(delais):
    """(n,) délais en heures → (n, K) poids des noyaux raised-cosine."""
    d = np.asarray(delais, dtype=float)[:, None]
    u = (d - CENTRES_LAG[None, :]) / LARGEURS_LAG[None, :]
    w = 0.5 * (1.0 + np.cos(np.pi * u))
    w[np.abs(u) >= 1.0] = 0.0
    w[(d < DELAI_MIN).ravel(), :] = 0.0
    return w


def construire_exposition(t_obs, repas, aliments):
    """
    t_obs   : (n,) instants d'observation de la douleur, en heures
    repas   : liste de (instant_heures, [aliments normalisés])
    aliments: liste ordonnée des aliments retenus
    → X (n, F*K) : colonne (f,k) = dose cumulée de l'aliment f vue par le
      noyau de décalage k.
    """
    n, F, K = len(t_obs), len(aliments), len(CENTRES_LAG)
    idx = {a: i for i, a in enumerate(aliments)}
    X = np.zeros((n, F * K))
    t_obs = np.asarray(t_obs, dtype=float)

    for t_repas, alims in repas:
        cibles = [i for i, a in enumerate(alims) if a in idx]
        if not cibles:
            continue
        delais = t_obs - t_repas
        pertinents = (delais >= DELAI_MIN) & (delais <= PORTEE_MAX)
        if not pertinents.any():
            continue
        w = poids_noyaux(delais[pertinents])          # (m, K)
        lignes = np.flatnonzero(pertinents)
        for i in cibles:
            j = idx[alims[i]] * K
            X[np.ix_(lignes, np.arange(j, j + K))] += w
    return X


def construire_controles(t_obs, heure_du_jour):
    """
    Facteurs de confusion retirés sans pénalité :
      - constante
      - rythme circadien (harmoniques 24 h et 12 h) : sinon un aliment du
        petit-déjeuner hérite du décalage qui tombe sur le pic du soir ;
      - dérive linéaire sur la durée du journal.
    """
    t = np.asarray(t_obs, dtype=float)
    h = np.asarray(heure_du_jour, dtype=float)
    jours = (t - t.mean()) / 24.0
    return np.column_stack([
        np.ones_like(t),
        np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
        np.sin(4 * np.pi * h / 24), np.cos(4 * np.pi * h / 24),
        jours,
    ])


def blanchir_ar1(y, X, Z, t_obs):
    """
    Transformation de Cochrane-Orcutt : la douleur est autocorrélée d'un repas
    au suivant. Sans blanchiment, le modèle « explique » cette inertie en
    attribuant à des aliments innocents la lente dérive du niveau de douleur.
    Le coefficient est atténué en puissance de l'écart de temps réel, les
    observations n'étant pas régulièrement espacées. Suppose `t_obs` trié.
    → (y, X, Z, rho) — la première observation est consommée par la transformation.
    """
    r = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    rho = float(np.clip((r[:-1] @ r[1:]) / max(r[:-1] @ r[:-1], 1e-9), 0.0, 0.9))
    dt = np.diff(np.asarray(t_obs, dtype=float))
    w = rho ** (dt / max(np.median(dt), 1e-6))

    def tr(M):
        M = np.asarray(M)
        return M[1:] - w[:, None] * M[:-1] if M.ndim == 2 else M[1:] - w * M[:-1]

    return tr(y), tr(X), tr(Z), rho


def residualiser(y, X, Z):
    """Frisch-Waugh-Lovell : retire l'espace de Z de y et de X.
    Exact pour le problème pénalisé, les colonnes de Z n'étant pas pénalisées."""
    coef, *_ = np.linalg.lstsq(Z, np.column_stack([y, X]), rcond=None)
    res = np.column_stack([y, X]) - Z @ coef
    return res[:, 0], res[:, 1:]


# ══════════════════════════════════════════════════════════════════
#  Group-lasso à coefficients positifs
# ══════════════════════════════════════════════════════════════════
#  min_{beta >= 0}  1/(2n)||y - X beta||^2 + lam * sum_g sqrt(K) * ||beta_g||_2
#
#  Un GROUPE = un aliment, c'est-à-dire ses K coefficients de décalage. Ils
#  entrent ou sortent du modèle ensemble : un vrai effet s'étale sur plusieurs
#  noyaux adjacents, et un lasso ordinaire paierait la pénalité sur chacun
#  séparément — d'où une perte de puissance.
#
#  Résolution par descente par blocs sur la matrice de Gram (astuce
#  « covariance updates » de glmnet), sous-problème de groupe par FISTA.
#  L'opérateur proximal de  lam||.||_2 + indicatrice(. >= 0)  est le
#  seuillage de groupe appliqué à la partie positive.

def _prox_groupe(v, seuil):
    v = np.maximum(v, 0.0)
    n = np.sqrt(v @ v)
    return v * max(0.0, 1.0 - seuil / n) if n > 0 else v


def _resoudre_groupe(Gg, u, lam_g, b0, pas, n_iter=80, tol=1e-9):
    """min_{b>=0} 1/2 b'Gg b - u'b + lam_g||b||_2, par FISTA."""
    b = b0.copy()
    z = b.copy()
    t = 1.0
    for _ in range(n_iter):
        b_new = _prox_groupe(z - pas * (Gg @ z - u), pas * lam_g)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = b_new + ((t - 1.0) / t_new) * (b_new - b)
        fini = np.abs(b_new - b).max() < tol
        b, t = b_new, t_new
        if fini:
            break
    return b


def nn_group_lasso(G, c, groupes, lam, poids, blocs_g, beta=None,
                   max_iter=150, tol=1e-6):
    """
    G, c    : X'X/n et X'y/n
    groupes : liste d'indices de colonnes (un tableau par aliment)
    blocs_g : liste de (Gg, pas) préparés par `preparer_blocs`
    """
    p = len(c)
    beta = np.zeros(p) if beta is None else beta.copy()
    Gbeta = G @ beta

    for _ in range(max_iter):
        delta_max = 0.0
        for gi, idx in enumerate(groupes):
            Gg, pas = blocs_g[gi]
            bg = beta[idx]
            # X_g'(y - X beta_{-g}) / n
            u = c[idx] - Gbeta[idx] + Gg @ bg
            lam_g = lam * poids[gi]
            if np.linalg.norm(np.maximum(u, 0.0)) <= lam_g:
                nouveau = np.zeros_like(bg)
            else:
                nouveau = _resoudre_groupe(Gg, u, lam_g, bg, pas)
            d = nouveau - bg
            dmax = np.abs(d).max()
            if dmax > 0.0:
                beta[idx] = nouveau
                Gbeta += G[:, idx] @ d
                delta_max = max(delta_max, dmax)
        if delta_max < tol:
            break
    return beta


def preparer_blocs(G, groupes):
    """Sous-matrices de Gram par groupe et pas de gradient associé."""
    prets = []
    for idx in groupes:
        Gg = G[np.ix_(idx, idx)]
        L = float(np.linalg.eigvalsh(Gg)[-1])
        prets.append((Gg, 1.0 / L if L > 1e-12 else 0.0))
    return prets


def lambda_max(c, groupes, poids):
    """Plus petite pénalité annulant toute la solution."""
    vals = [np.linalg.norm(np.maximum(c[idx], 0.0)) / poids[gi]
            for gi, idx in enumerate(groupes)]
    return max(max(vals), 1e-9)


def grille_lambda(G, c, groupes, poids, blocs_g, q_max, n_lam=N_LAMBDAS):
    """
    Grille de pénalités restreinte au régime PARCIMONIEUX : on ne garde que
    les lambda pour lesquels au plus `q_max` aliments entrent dans le modèle
    (règle de Meinshausen & Bühlmann). Sans cette borne, la fréquence de
    sélection maximisée sur le chemin vaut ~1 pour tout le monde et la
    sélection ne discrimine plus rien.
    """
    chemin = lambda_max(c, groupes, poids) * np.logspace(0, -2.0, 50)
    beta, retenues = None, []
    for lam in chemin:
        beta = nn_group_lasso(G, c, groupes, lam, poids, blocs_g, beta=beta)
        actifs = sum(1 for idx in groupes if beta[idx].max() > 0)
        if actifs == 0:
            continue
        if actifs > q_max:
            break
        retenues.append(lam)
    if not retenues:
        retenues = [chemin[min(4, len(chemin) - 1)]]
    if len(retenues) > n_lam:
        idx = np.linspace(0, len(retenues) - 1, n_lam).round().astype(int)
        retenues = [retenues[i] for i in idx]
    return np.array(retenues)


def orthonormaliser(Xs, groupes):
    """
    Remplace les colonnes de chaque groupe par une base orthonormée du même
    sous-espace (X_g'X_g/n = I). Le group-lasso devient alors invariant à la
    corrélation INTERNE au groupe — c'est sa formulation correcte : sans cela
    la pénalité est un ellipsoïde et favorise arbitrairement certains noyaux
    de décalage. Bonus : la mise à jour de bloc a une forme fermée.
    """
    n = Xs.shape[0]
    Xq = np.zeros_like(Xs)
    for idx in groupes:
        U, s, _ = np.linalg.svd(Xs[:, idx], full_matrices=False)
        if s.size == 0 or s[0] <= 0:
            continue
        garde = s > s[0] * 1e-8
        Xq[:, idx[:int(garde.sum())]] = U[:, garde] * np.sqrt(n)
    return Xq


def group_lasso_ortho(G, c, groupes, lam, poids, beta=None,
                      max_iter=200, tol=1e-7):
    """Group-lasso sur groupes orthonormés : descente par blocs, forme fermée."""
    beta = np.zeros(len(c)) if beta is None else beta.copy()
    Gbeta = G @ beta
    for _ in range(max_iter):
        delta_max = 0.0
        for gi, idx in enumerate(groupes):
            bg = beta[idx]
            u = c[idx] - Gbeta[idx] + bg          # G_gg = I
            norme = np.sqrt(u @ u)
            seuil = lam * poids[gi]
            nouveau = u * (1.0 - seuil / norme) if norme > seuil else np.zeros_like(u)
            d = nouveau - bg
            dmax = np.abs(d).max()
            if dmax > 0.0:
                beta[idx] = nouveau
                Gbeta += G[:, idx] @ d
                delta_max = max(delta_max, dmax)
        if delta_max < tol:
            break
    return beta


def lambda_max_ortho(c, groupes, poids):
    return max(max(np.linalg.norm(c[idx]) / poids[gi]
                   for gi, idx in enumerate(groupes)), 1e-9)


def grille_lambda_ortho(G, c, groupes, poids, q_max, n_lam=N_LAMBDAS):
    chemin = lambda_max_ortho(c, groupes, poids) * np.logspace(0, -2.0, 40)
    beta, retenues = None, []
    for lam in chemin:
        beta = group_lasso_ortho(G, c, groupes, lam, poids, beta=beta)
        actifs = sum(1 for idx in groupes if np.abs(beta[idx]).max() > 0)
        if actifs == 0:
            continue
        if actifs > q_max:
            break
        retenues.append(lam)
    if not retenues:
        retenues = [chemin[min(4, len(chemin) - 1)]]
    if len(retenues) > n_lam:
        i = np.linspace(0, len(retenues) - 1, n_lam).round().astype(int)
        retenues = [retenues[j] for j in i]
    return np.array(retenues)


# ══════════════════════════════════════════════════════════════════
#  Aliments indissociables
# ══════════════════════════════════════════════════════════════════

def fusionner_indissociables(repas, aliments, seuil=SEUIL_JACCARD):
    """
    Deux aliments presque toujours consommés ensemble ne sont pas séparables
    par des données d'observation : on les fusionne en un bloc explicite plutôt
    que de laisser le lasso en désigner un au hasard.
    → (blocs, membres) : blocs = noms de blocs, membres = {bloc: [aliments]}
    """
    presence = {a: set() for a in aliments}
    for i, (_, alims) in enumerate(repas):
        for a in alims:
            if a in presence:
                presence[a].add(i)

    parent = {a: a for a in aliments}

    def racine(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(aliments):
        for b in aliments[i + 1:]:
            inter = len(presence[a] & presence[b])
            if not inter:
                continue
            if inter / len(presence[a] | presence[b]) >= seuil:
                ra, rb = racine(a), racine(b)
                if ra != rb:
                    parent[rb] = ra

    membres = defaultdict(list)
    for a in aliments:
        membres[racine(a)].append(a)
    blocs, sortie = [], {}
    for r, ms in membres.items():
        nom = "+".join(sorted(ms))
        blocs.append(nom)
        sortie[nom] = sorted(ms)
    blocs.sort()
    return blocs, sortie


# ══════════════════════════════════════════════════════════════════
#  Analyse complète
# ══════════════════════════════════════════════════════════════════

def analyser(observations, repas, min_occurrences=MIN_OCCURRENCES,
             n_replicats=N_REPLICATS, seuil=SEUIL_STABILITE, q_max=None,
             blanchir=True, positif=True, graine=0):
    """
    observations : [(instant_heures, heure_du_jour, score_douleur)]
    repas        : [(instant_heures, [aliments normalisés])]
    → dict de résultats
    """
    rng = np.random.default_rng(graine)

    # -- aliments assez fréquents, puis fusion des indissociables ----
    compte = defaultdict(int)
    for _, alims in repas:
        for a in set(alims):
            compte[a] += 1
    frequents = sorted(a for a, c in compte.items() if c >= min_occurrences)
    if not frequents:
        return {"erreur": "aucun aliment n'atteint le minimum d'occurrences",
                "compte": dict(compte)}

    blocs, membres = fusionner_indissociables(repas, frequents)
    vers_bloc = {a: b for b, ms in membres.items() for a in ms}
    repas_blocs = [(t, sorted({vers_bloc[a] for a in alims if a in vers_bloc}))
                   for t, alims in repas]

    # -- matrices -----------------------------------------------------
    t_obs = np.array([o[0] for o in observations], dtype=float)
    heures = np.array([o[1] for o in observations], dtype=float)
    y = np.array([o[2] for o in observations], dtype=float)

    X = construire_exposition(t_obs, repas_blocs, blocs)
    Z = construire_controles(t_obs, heures)
    rho = 0.0
    if blanchir and len(y) > 4:
        y_b, X_b, Z_b, rho = blanchir_ar1(y, X, Z, t_obs)
        y_res, X_res = residualiser(y_b, X_b, Z_b)
        X, t_obs_eff = X[1:], t_obs[1:]
    else:
        y_res, X_res = residualiser(y, X, Z)
        t_obs_eff = t_obs

    n, K = len(y_res), len(CENTRES_LAG)

    # -- aliments structurellement indétectables ---------------------
    #  Si l'exposition à un aliment est presque entièrement absorbée par les
    #  contrôles (aliment consommé tous les jours à heure fixe), il ne reste
    #  aucune variation à exploiter : sa colonne résiduelle n'est que du bruit,
    #  que la standardisation amplifierait jusqu'à le faire ressortir. On
    #  l'écarte explicitement plutôt que de produire un résultat trompeur.
    norme_brute = np.sqrt((X ** 2).sum(axis=0))
    norme_res = np.sqrt((X_res ** 2).sum(axis=0))
    part_libre = np.where(norme_brute > 1e-12, norme_res / np.maximum(norme_brute, 1e-12), 0.0)
    informatif = part_libre.reshape(len(blocs), K).max(axis=1) >= 0.10
    indetectables = [blocs[b] for b in np.flatnonzero(~informatif)]
    if not informatif.any():
        return {"erreur": "aucun aliment ne varie assez pour être testé "
                          "(alimentation trop régulière)"}
    if not informatif.all():
        gardes = np.flatnonzero(informatif)
        cols = np.concatenate([np.arange(b * K, (b + 1) * K) for b in gardes])
        X, X_res = X[:, cols], X_res[:, cols]
        blocs = [blocs[b] for b in gardes]

    n_blocs = len(blocs)
    echelle = np.sqrt((X_res ** 2).sum(axis=0) / n)
    echelle[echelle < 1e-12] = 1.0
    Xs = X_res / echelle
    groupes = [np.arange(b * K, (b + 1) * K) for b in range(n_blocs)]
    poids = np.full(n_blocs, np.sqrt(K))

    if q_max is None:
        q_max = max(4, int(round(np.sqrt(0.8 * n_blocs))))

    if positif:
        G_tot = Xs.T @ Xs / n
        c_tot = Xs.T @ y_res / n
        lambdas = grille_lambda(G_tot, c_tot, groupes, poids,
                                preparer_blocs(G_tot, groupes), q_max)
    else:
        Xo = orthonormaliser(Xs, groupes)
        G_tot = Xo.T @ Xo / n
        c_tot = Xo.T @ y_res / n
        lambdas = grille_lambda_ortho(G_tot, c_tot, groupes, poids, q_max)

    # -- sélection par stabilité, blocs de jours contigus -------------
    jour_obs = (t_obs_eff // 24).astype(int)
    jours = np.unique(jour_obs)
    debuts = list(range(0, len(jours), TAILLE_BLOC_JOURS))
    n_tire = max(1, int(round(len(debuts) * FRACTION_SOUS_ECH)))

    compte_sel = np.zeros((len(lambdas), n_blocs))
    replicats_valides = 0

    for _ in range(n_replicats):
        choisis = rng.choice(len(debuts), size=n_tire, replace=False)
        jours_gardes = np.concatenate(
            [jours[d:d + TAILLE_BLOC_JOURS] for d in (debuts[c] for c in choisis)])
        masque = np.isin(jour_obs, jours_gardes)
        nb = int(masque.sum())
        if nb < 8:
            continue
        replicats_valides += 1
        Xb, yb = Xs[masque], y_res[masque]
        if not positif:
            Xb = orthonormaliser(Xb, groupes)
        G = Xb.T @ Xb / nb
        c = Xb.T @ yb / nb
        blocs_g = preparer_blocs(G, groupes) if positif else None
        beta = None
        for li, lam in enumerate(lambdas):
            beta = (nn_group_lasso(G, c, groupes, lam, poids, blocs_g, beta=beta)
                    if positif else
                    group_lasso_ortho(G, c, groupes, lam, poids, beta=beta))
            par_bloc = np.abs(beta.reshape(n_blocs, K))
            actifs = par_bloc.max(axis=1) > 0
            compte_sel[li, actifs] += 1

    if replicats_valides == 0:
        return {"erreur": "journal trop court pour le ré-échantillonnage"}

    frequences = compte_sel.max(axis=0) / replicats_valides

    # -- ré-ajustement non pénalisé sur les candidats ----------------
    #  Le group-lasso étale les coefficients à l'intérieur d'un groupe (norme
    #  L2) : son profil ne localise pas le décalage. On ré-ajuste donc sans
    #  pénalité (NNLS) sur les seuls aliments candidats, et c'est de CE profil
    #  qu'on tire décalage et effet, exprimés en points de douleur.
    from scipy.optimize import nnls

    decalages = np.full(n_blocs, np.nan)
    effets = np.zeros(n_blocs)
    pics = np.zeros(n_blocs)
    #  Le ré-ajustement doit rester PARCIMONIEUX : y verser les quasi-retenus
    #  dilue les contributions entre colonnes corrélées et déplace le décalage
    #  estimé. On se limite aux aliments retenus (au minimum les 2 premiers).
    candidats = np.flatnonzero(frequences >= seuil)
    if len(candidats) < 2:
        candidats = np.argsort(-frequences)[:2]
    cols = np.concatenate([np.arange(b * K, (b + 1) * K) for b in candidats])
    coef, _ = nnls(X_res[:, cols], y_res)
    #  Le décalage se lit sur la FONCTION DE RÉPONSE reconstruite
    #  h(d) = somme_k coef_k * noyau_k(d), et non sur la masse des
    #  contributions : les noyaux larges agrègent plus de repas, donc leurs
    #  colonnes ont une moyenne plus grande, ce qui biaiserait le centroïde
    #  vers les longs délais quel que soit le vrai décalage.
    grille_d = np.linspace(0.0, PORTEE_MAX, 400)
    base_d = poids_noyaux(grille_d)                              # (400, K)
    for pos, b in enumerate(candidats):
        tr = slice(pos * K, (pos + 1) * K)
        reponse = base_d @ coef[tr]                              # (400,)
        if reponse.max() > 1e-9:
            decalages[b] = float(grille_d[int(np.argmax(reponse))])
            pics[b] = float(reponse.max())
        # coefficients estimés sur les données résiduelles (exacts, FWL),
        # appliqués à l'exposition BRUTE → points de douleur réellement ajoutés
        effets[b] = float((X[:, cols[tr]] * coef[tr]).mean(axis=0).sum())

    retenus = np.flatnonzero(frequences >= seuil)

    return {
        "blocs": blocs,
        "membres": membres,
        "frequences": frequences,
        "decalages": decalages,
        "effets": effets,
        "pics": pics,
        "occurrences": np.array([
            sum(1 for _, al in repas_blocs if b in al) for b in blocs]),
        "retenus": retenus,
        "n_observations": n,
        "rho_ar1": rho,
        "n_jours": len(jours),
        "n_replicats": replicats_valides,
        "lambdas": lambdas,
        "q_max": q_max,
        "seuil": seuil,
        "ecartes": sorted(a for a, c in compte.items() if c < min_occurrences),
        "indetectables": indetectables,
    }
