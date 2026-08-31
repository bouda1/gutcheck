# Installation

To install the dependencies, run:

```bash
uv --version                                      # Already installed?
curl -LsSf https://astral.sh/uv/install.sh | sh   # installs uv and uvx into ~/.local/bin
. ~/.local/bin/env                                # add them to PATH for this shell
uv --version                                      # 0.9 or later is fine.
```

Then install the Python dependencies:

```bash
uv venv gutcheck
. gutcheck/bin/activate
uv pip install scipy
```

# gutcheck — detecting foods that trigger pain

Three constraints define the problem and drive every design decision in the
algorithm:

1. **The lag is unknown.** Pain may follow ingestion by anything from a few
   hours to more than two days, and the delay differs from one food to the next.
2. **There is probably more than one culprit.** Testing foods one at a time
   credits the first with what belongs to the second as soon as the two are
   correlated — and they always are (meals resemble one another).
3. **Data are scarce and noisy.** A four-week diary means roughly 30 foods to
   separate on the basis of some 85 pain entries.

## The algorithm

**1 — Distributed-lag exposure.** Pain recorded at time *t* is modelled as the
sum of the contributions of *every* past meal. Each food gets its own response,
described by 7 smooth (raised-cosine) kernels centred from 2 h to 60 h. There is
no "lag to test": the profile is estimated.

**2 — Controls.** Circadian rhythm (24 h and 12 h harmonics) and linear drift,
partialled out without penalty (Frisch–Waugh–Lovell). Without this, a breakfast
food mechanically inherits the evening pain peak under the label "12 h lag".

**3 — AR(1) whitening.** Pain is autocorrelated from one entry to the next. Left
untreated, that inertia gets "explained" by innocent foods.

**4 — Non-negative group lasso.** All foods are estimated *simultaneously*, each
judged with competing foods held constant. One group = one food, meaning its 7
lag coefficients: they enter or leave together, since otherwise a genuine effect
spread across several kernels would pay the penalty on each one. Coefficients
are constrained to be ≥ 0: we are looking for triggers.

**5 — Stability selection** (Meinshausen & Bühlmann). No p-values: with 30 foods
× 7 lags, multiple testing and a lag chosen after the fact, any p-value would be
invalid. Instead the model is refitted on 200 subsamples drawn *as blocks of
contiguous days* (autocorrelation rules out sampling entries independently), and
what we keep is the **selection frequency**. The penalty grid is bounded to the
sparse regime, without which everything eventually gets selected and the
frequency no longer discriminates anything.

**6 — Lag and effect size.** An unpenalised refit (NNLS) on the retained foods
only. The lag is read off the **peak of the reconstructed response function** —
not off the mass of the contributions, because wide kernels aggregate more meals
and would bias the delay upwards.

## What the algorithm reports explicitly

- **Inseparable foods**: those almost always eaten together (Jaccard ≥ 0.85) are
  merged into a single block. No observational data can tell them apart; saying
  so is better than naming one of them at random.
- **Undetectable foods**: those whose exposure is absorbed by the controls (eaten
  every day at a fixed time). There is simply no variation to exploit — only a
  temporary elimination can test them.
- **Discarded foods**: fewer than 3 occurrences.

## Validation

`python gutcheck.py --valider` measures performance on synthetic diaries whose
culprits are **known**: 3 culprits among some 32 foods, lags of 3 h / 6 h / 26 h,
noisy and autocorrelated pain, foods tied to the time of the meal, and one
inseparable pair.

| days | entries | recall | precision | lag error |
|---:|---:|---:|---:|---:|
| **pain recorded at meals only** | | | | |
| 21 | 63 | 17 % | 67 % | 2.5 h |
| 28 | 84 | 36 % | 82 % | 3.1 h |
| 42 | 126 | 33 % | 78 % | 1.9 h |
| 56 | 168 | 64 % | 69 % | 2.0 h |
| 84 | 252 | 75 % | 72 % | 2.0 h |
| **+ 6 entries/day away from meals** | | | | |
| 21 | 189 | 44 % | 72 % | 2.0 h |
| 28 | 252 | 64 % | 81 % | 1.6 h |
| 42 | 378 | 86 % | 88 % | 0.9 h |
| 56 | 504 | **100 %** | 83 % | 1.8 h |
| 84 | 756 | **100 %** | 78 % | 0.9 h |

*recall* = share of the true culprits retained · *precision* = share of the
retained foods that really are culprits · 12 diaries per row.

Two lessons:

- **Recording pain away from meals is worth as much as doubling the length of the
  diary.** If pain is only noted at mealtimes, "12 h after breakfast" and "in the
  evening" are literally the same column: nothing can distinguish them.
- **Below about 4 weeks, the algorithm is cautious**: few false positives, but it
  misses most of the culprits. This is not a flaw to be fixed, it is the amount
  of information available.

## Limitations

Observation does not prove causation. A retained food is a **candidate to test**,
not a verdict: confirmation goes through a two-week elimination followed by a
reintroduction, one food at a time. The program suggests which candidate to test
first.

## Usage

```
python gutcheck.py journal.csv         analyse a CSV diary
python gutcheck.py journal.ods [name]  analyse a Calc workbook ([name] = sheet)
python gutcheck.py --exemple f.ods     write a synthetic test diary
python gutcheck.py --valider           measure performance
python gutcheck.py --aide-format       expected file format
python test_gutcheck.py                regression tests
```

Two formats are accepted, whichever you prefer: **CSV** or a **LibreOffice Calc
workbook (`.ods`)** — the latter read directly, with no extra dependency.
Headers tolerate formatting (`Douleur (0-10)` → `douleur`), dates accept the
usual formats (`31/08/2026` as well as `2026-08-31`), and date/time cells typed
by Calc are read back at their canonical value rather than their local display.

Columns: `date, heure, repas, aliments, douleur`
(date, time, meal, foods, pain)

```
2026-08-31,08:00,petit_dejeuner,oeufs; pain; cafe,2
2026-08-31,11:00,,,4              ← pain entry on its own
2026-08-31,12:30,dejeuner,riz; poulet; legumes,4
2026-08-31,19:00,diner,soupe; fromage,
```

Two data-entry tips that matter more than the algorithm does:

1. **Record pain every 3–4 h**, including away from meals (see above).
2. **Vary your diet.** A food eaten every single day is undetectable whatever its
   effect: there is no comparison day.

## Files

| | |
|---|---|
| `gutcheck.py` | interface, CSV reading, report |
| `modele.py` | exposure, controls, group lasso, stability selection |
| `tableur.py` | reading and writing `.ods` workbooks (standard library) |
| `simu.py` | generator of diaries with known culprits, evaluation |
| `test_gutcheck.py` | regression tests (`python test_gutcheck.py`) |

Dependencies: `numpy`, `scipy`.
