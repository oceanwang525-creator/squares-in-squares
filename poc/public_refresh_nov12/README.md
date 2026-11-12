# Public refresh, Nov 12

## Rounds 2 + 3 (on top of the team merged.csv at 22.173824282607068, upstream 79ef22e)

**41 n improved, official score 22.173820983104367 (−3.2995e-6).** `data/submission.csv` is now the round-3 full CSV (round-1 reports moved to `data/round1/`).

| Source | n | Score gain |
|---|---:|---:|
| Uniform shrink (`poc/shrink_pass.py`): packings whose every neighbouring pair has a positive SAT gap (1e-12 … 1e-9) are scaled toward their centre by half the safe amount | 36 | about 3e-7 |
| Round 3: second shrink pass at 80% of the remaining gap (34 n, +6.4e-10) and a tiny deletion/wrap pass on the round-2 file (n = 591, 963, 966) | 3 new | 6.4e-10 |
| Kaggle notebooks re-run since the last team refresh: jazivxt/boxed-in-sub (n = 370), nilanjansaha123/squares-gpu-seam-dp-soft-rigid-highs (n = 598) | 2 | about 3e-6 |

Every n was certified individually (official per-puzzle evaluator, literal strings and after a pandas 3.1.0 round trip, both strictly below merged.csv). Per-n list with L values: [docs/round2_wins.md](docs/round2_wins.md).

**Whole-file verification (`squares score`, pandas 3.1.0):**

| Transport | Score | Accepted | Missing | Rejected |
|---|---|---:|---:|---:|
| Literal `data/submission.csv` | 22.173820983104367 | 1000 | 0 | 0 |
| After `pd.read_csv(...).to_csv(index=False)` | 22.173820983104367 | 1000 | 0 | 0 |

All 41 n are strictly lower than merged.csv in both runs (`data/wins_vs_merged.json`); the score drop equals their sum (3.29950e-6). Note: `squares score` applies Kaggle's numeric (float) transport before scoring, while `border-scan/merged-literal-official.json` holds raw-string L values. For 52 untouched n (byte-identical rows) the raw-string reference is 1e-16 … 1e-15 lower than the transported value, so the script lists them as "not strictly better"; against `merged-pandas-official.json` there are 0 differences.

945 of 999 non-trivial n have a touching pair (gap < 1e-13), so uniform shrink cannot help them.

## Round 1

Per-n refresh of `border-scan/merged.csv` from public sources posted after the Nov 10 import, plus two team CSVs, plus a certified tiny-win pass (deletion and grid-border wraps). Every win is strictly smaller than merged.csv and passes the unchanged official metric.

**Result: 231 n improved, score 22.199718022945788 → 22.176602111957 (−0.023115911).**

| Final source of the n | n | Score gain vs merged.csv |
|---|---:|---:|
| Kaggle notebook jonathan1506/50-top-2-6-22-1793-fault-dp-fq-stretch | 125 | 0.015404 |
| Kaggle notebook jazivxt/boxed-in-sub | 31 | 0.003212 |
| Team files submission189.csv (24) + final_atlas.csv (7, Kaggle 22.1908487816) | 31 | 0.002853 |
| Tiny pass: 27 grid-border wraps + 3 deletions (`poc/tiny_pass.py`) | 30 | 0.001471 |
| Kaggle notebook prvsiyan/a-square-deal-research-to-verified-packings | 14 | 0.000176 |

Most of the tiny pass's 30 n had already been improved by a public packing; the pass itself adds 2.49e-10 on top (largest: n = 733 and 487, 3.0e-9 in L each). Original authors keep credit; their CSVs were read as data only, never executed. The full per-n list with both L values is in [docs/wins_vs_merged.md](docs/wins_vs_merged.md).

## Verification

`squares score` (bundled official metric, pandas 3.1.0) on the full 1000-row `data/submission.csv`:

| Transport | Score | Accepted | Missing | Rejected |
|---|---|---:|---:|---:|
| Literal file | 22.176602111957 | 1000 | 0 | 0 |
| After `pd.read_csv(...).to_csv(index=False)` | 22.176602111957 | 1000 | 0 | 0 |

231 n are strictly lower than merged.csv in both transports; no n is worse. Reports: `data/official-literal.json`, `data/official-pandas.json`, `data/wins_vs_merged.json`.

Each candidate was also certified per n before acceptance: official `evaluate_single_puzzle` on the literal strings and on the strings after an actual pandas round trip, keeping it only when both certified L are below the incumbent.

## What was scanned

- 129 public Kaggle notebooks with outputs (anonymous kernels API), all CSV outputs, sorted by public score (`docs/kernels.json`).
- About 30 public Kaggle datasets about the competition (`docs/datasets.json`): no wins.
- 14 public GitHub repos from other teams: no wins.
- Two team CSVs supplied directly: final_atlas.csv, submission189.csv.
- Border-wrap and deletion chains on the merged result: only ~2e-10 total, not used.

## Negative results

- **Polish:** a numba shrink-and-relax optimizer (SAT penetration penalty, free angles, L-BFGS) on six of the largest leader-gap n (269, 328, 333, 365, 367, 379), 150 s each, about 300 relaxations: no feasible box even 1e-5 below the incumbent. Consistent with neighbour_deletion and freeze_relax_momentum. See [docs/notes.md](docs/notes.md).
- **Constructions:** 45° band/rectangle families (sweep, exact LP model, critical positions), tilted blocks at any angle with a two-stage seed-and-squeeze search, and insert-one relaxation. The search rediscovers our records from scratch (3×3 → 0.62133, 3×5 → 0.59863, 5×5 → 0.53554, Pell-17 → 0.52085) but finds nothing better: 0 n. Tools and measurements in [docs/construction_search.md](docs/construction_search.md).

## Run

```sh
python poc/public_refresh_nov12/poc/listk.py                      # writes kernels.json
python poc/public_refresh_nov12/poc/merge_public.py merged.csv lit.json pandas.json kernels.json
python poc/public_refresh_nov12/poc/build.py merged.csv state/best.json out.csv
python poc/public_refresh_nov12/poc/verify_full.py out.csv lit.json pandas.json verify/
```

Scripts import `squares` from the shared package; run them from a working directory, since they write `state/`.
