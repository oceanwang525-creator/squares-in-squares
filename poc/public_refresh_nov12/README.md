# Public refresh, Nov 12

Per-n refresh of `border-scan/merged.csv` from public sources posted after the Nov 10 import, plus two team CSVs. No new geometry: every win is an existing packing that is strictly smaller than ours and passes the unchanged official metric.

**Result: 206 n improved, score 22.199718022945788 → 22.176602112206332 (−0.023116).**

| Source | n | Score gain |
|---|---:|---:|
| Kaggle notebook jonathan1506/50-top-2-6-22-1793-fault-dp-fq-stretch | 129 | 0.016723 |
| Kaggle notebook jazivxt/boxed-in-sub | 32 | 0.003364 |
| Team file submission189.csv | 24 | (0.002853 with the row below) |
| Team file final_atlas.csv (Kaggle 22.1908487816) | 7 | |
| Kaggle notebook prvsiyan/a-square-deal-research-to-verified-packings | 14 | 0.000176 |

Original authors keep credit; their CSVs were read as data only, never executed. The full per-n list with both L values is in [docs/wins_vs_merged.md](docs/wins_vs_merged.md).

## Verification

`squares score` (bundled official metric, pandas 3.1.0) on the full 1000-row `data/submission.csv`:

| Transport | Score | Accepted | Missing | Rejected |
|---|---|---:|---:|---:|
| Literal file | 22.176602112206332 | 1000 | 0 | 0 |
| After `pd.read_csv(...).to_csv(index=False)` | 22.176602112206332 | 1000 | 0 | 0 |

206 n are strictly lower than merged.csv in both transports; no n is worse. Reports: `data/official-literal.json`, `data/official-pandas.json`, `data/wins_vs_merged.json`.

Each candidate was also certified per n before acceptance: official `evaluate_single_puzzle` on the literal strings and on the strings after an actual pandas round trip, keeping it only when both certified L are below the incumbent.

## What was scanned

- 129 public Kaggle notebooks with outputs (anonymous kernels API), all CSV outputs, sorted by public score (`docs/kernels.json`).
- About 30 public Kaggle datasets about the competition (`docs/datasets.json`): no wins.
- 14 public GitHub repos from other teams: no wins.
- Two team CSVs supplied directly: final_atlas.csv, submission189.csv.
- Border-wrap and deletion chains on the merged result: only ~2e-10 total, not used.

## Negative result

A numba shrink-and-relax optimizer (SAT penetration penalty, free angles, L-BFGS) on six of the largest leader-gap n (269, 328, 333, 365, 367, 379), 150 s each, about 300 relaxations: no feasible box even 1e-5 below the incumbent. Consistent with neighbour_deletion and freeze_relax_momentum. See [docs/notes.md](docs/notes.md).

## Run

```sh
python poc/public_refresh_nov12/poc/listk.py                      # writes kernels.json
python poc/public_refresh_nov12/poc/merge_public.py merged.csv lit.json pandas.json kernels.json
python poc/public_refresh_nov12/poc/build.py merged.csv state/best.json out.csv
python poc/public_refresh_nov12/poc/verify_full.py out.csv lit.json pandas.json verify/
```

Scripts import `squares` from the shared package; run them from a working directory, since they write `state/`.
