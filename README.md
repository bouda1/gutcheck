# Installation

Pour installer les dépendances, exécuter :

```bash
uv --version                                      # Déjà installé ?
curl -LsSf https://astral.sh/uv/install.sh | sh   # installation de uv and uvx dans ~/.local/bin
. ~/.local/bin/env                                # On les ajoute au PATH pour ce shell
uv --version                                      # 0.9 ou plus est bien.
```

Ensuite, installer les dépendances Python :

```bash
uv venv alim
. alim/bin/activate
uv pip install scipy
```

# alim — détection d'aliments déclencheurs de douleur

Trois contraintes définissent le problème et déterminent tout l'algorithme :

1. **Le décalage est inconnu.** La douleur peut suivre l'ingestion de quelques
   heures à plus de deux jours, et le délai diffère d'un aliment à l'autre.
2. **Il y a probablement plusieurs coupables.** Tester les aliments un par un
   attribue au premier ce qui revient au second dès qu'ils sont corrélés — et
   ils le sont toujours (les repas se ressemblent).
3. **Les données sont rares et bruitées.** Un journal de quatre semaines, c'est
   ~30 aliments à départager sur ~85 relevés de douleur.

## L'algorithme

**1 — Exposition à décalage distribué.** La douleur relevée à l'instant *t* est
modélisée comme la somme des contributions de *tous* les repas passés. Chaque
aliment reçoit une réponse propre, décrite par 7 noyaux lisses (raised-cosine)
centrés de 2 h à 60 h. Pas de « décalage à tester » : le profil est estimé.

**2 — Contrôles.** Rythme circadien (harmoniques 24 h et 12 h) et dérive
linéaire, retirés sans pénalité (Frisch-Waugh-Lovell). Sans cela, un aliment de
petit-déjeuner hérite mécaniquement du pic de douleur du soir sous l'étiquette
« décalage de 12 h ».

**3 — Blanchiment AR(1).** La douleur est autocorrélée d'un relevé au suivant.
Non traitée, cette inertie est « expliquée » par des aliments innocents.

**4 — Group-lasso positif.** Tous les aliments sont estimés *simultanément*,
chacun jugé à aliments concurrents constants. Un groupe = un aliment, c'est-à-dire
ses 7 coefficients de décalage : ils entrent ou sortent ensemble, sinon un vrai
effet étalé sur plusieurs noyaux paierait la pénalité sur chacun. Coefficients
contraints ≥ 0 : on cherche des déclencheurs.

**5 — Sélection par stabilité** (Meinshausen & Bühlmann). Pas de p-value : avec
30 aliments × 7 décalages, des tests multiples et un décalage choisi a
posteriori, toute p-value serait invalide. À la place, le modèle est réajusté
sur 200 sous-échantillons *par blocs de jours contigus* (l'autocorrélation
interdit de tirer les relevés indépendamment), et on retient la **fréquence de
sélection**. La grille de pénalités est bornée au régime parcimonieux, sans quoi
tout finit par être sélectionné et la fréquence ne discrimine plus rien.

**6 — Décalage et effet.** Réajustement non pénalisé (NNLS) sur les seuls
aliments retenus. Le décalage se lit sur le **pic de la fonction de réponse
reconstruite** — pas sur la masse des contributions, car les noyaux larges
agrègent plus de repas et biaiseraient le délai vers le haut.

## Ce que l'algorithme signale explicitement

- **Aliments indissociables** : ceux presque toujours consommés ensemble
  (Jaccard ≥ 0,85) sont fusionnés en un bloc. Aucune donnée d'observation ne
  peut les départager ; le dire vaut mieux qu'en désigner un au hasard.
- **Aliments indétectables** : ceux dont l'exposition est absorbée par les
  contrôles (consommés tous les jours à heure fixe). Il n'existe aucune
  variation à exploiter — seule une suppression temporaire peut les tester.
- **Aliments écartés** : moins de 3 occurrences.

## Validation

`python alim.py --valider` mesure les performances sur des journaux
synthétiques dont les coupables sont **connus** : 3 coupables parmi ~32
aliments, décalages 3 h / 6 h / 26 h, douleur bruitée et autocorrélée, aliments
liés au moment du repas, une paire indissociable.

| jours | relevés | rappel | précision | erreur de décalage |
|---:|---:|---:|---:|---:|
| **douleur relevée aux repas seuls** | | | | |
| 21 | 63 | 17 % | 67 % | 2,5 h |
| 28 | 84 | 36 % | 82 % | 3,1 h |
| 42 | 126 | 33 % | 78 % | 1,9 h |
| 56 | 168 | 64 % | 69 % | 2,0 h |
| 84 | 252 | 75 % | 72 % | 2,0 h |
| **+ 6 relevés/jour hors repas** | | | | |
| 21 | 189 | 44 % | 72 % | 2,0 h |
| 28 | 252 | 64 % | 81 % | 1,6 h |
| 42 | 378 | 86 % | 88 % | 0,9 h |
| 56 | 504 | **100 %** | 83 % | 1,8 h |
| 84 | 756 | **100 %** | 78 % | 0,9 h |

*rappel* = part des vrais coupables retenus · *précision* = part des aliments
retenus qui sont vraiment coupables · 12 journaux par ligne.

Deux enseignements :

- **Relever la douleur hors repas vaut autant que doubler la durée du journal.**
  Si la douleur n'est notée qu'aux repas, « 12 h après le petit-déjeuner » et
  « le soir » sont littéralement la même colonne : rien ne peut les distinguer.
- **En dessous de ~4 semaines, l'algorithme est prudent** : peu de faux positifs,
  mais il rate la majorité des coupables. Ce n'est pas un défaut à corriger, c'est
  la quantité d'information disponible.

## Limites

L'observation ne prouve pas la causalité. Un aliment retenu est un **candidat à
tester**, pas un verdict : la confirmation passe par une suppression de deux
semaines puis une réintroduction, un aliment à la fois. Le programme propose le
candidat à tester en priorité.

## Usage

```
python alim.py journal.csv         analyse un journal CSV
python alim.py journal.ods [nom]   analyse un classeur Calc ([nom] = feuille)
python alim.py --exemple f.ods     écrit un journal synthétique de test
python alim.py --valider           mesure les performances
python alim.py --aide-format       format de fichier attendu
python test_alim.py                tests de non-régression
```

Deux formats sont acceptés, au choix : **CSV** ou **classeur LibreOffice Calc
(`.ods`)** — ce dernier lu directement, sans dépendance supplémentaire. Les
en-têtes tolèrent la mise en forme (`Douleur (0-10)` → `douleur`), les dates les
formats usuels (`31/08/2026` comme `2026-08-31`), et les cellules date/heure
typées par Calc sont relues à leur valeur canonique et non à leur affichage
local.

Colonnes : `date, heure, repas, aliments, douleur`

```
2026-08-31,08:00,petit_dejeuner,oeufs; pain; cafe,2
2026-08-31,11:00,,,4              ← relevé de douleur seul
2026-08-31,12:30,dejeuner,riz; poulet; legumes,4
2026-08-31,19:00,diner,soupe; fromage,
```

Deux conseils de saisie qui pèsent plus lourd que l'algorithme :

1. **Relever la douleur toutes les 3-4 h**, y compris hors repas (voir plus haut).
2. **Varier.** Un aliment consommé tous les jours sans exception est
   indétectable quel que soit son effet : il n'y a aucun jour de comparaison.

## Fichiers

| | |
|---|---|
| `alim.py` | interface, lecture CSV, rapport |
| `modele.py` | exposition, contrôles, group-lasso, sélection par stabilité |
| `tableur.py` | lecture et écriture de classeurs `.ods` (bibliothèque standard) |
| `simu.py` | générateur de journaux à coupables connus, évaluation |
| `test_alim.py` | tests de non-régression (`python test_alim.py`) |

Dépendances : `numpy`, `scipy`.
