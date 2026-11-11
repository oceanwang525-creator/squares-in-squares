# Fresh public sources (Nov 10 2026)

Result: 158 strict per-n wins over `border-scan/merged.csv` from two public Kaggle notebook outputs (no login needed). Full CSV with only those rows swapped: `out/fresh_merged.csv` (+ `out/fresh_merged_pandas.csv` after pandas read_csv/to_csv). Official `squares score`: both 1000/1000 accepted, score 22.199718022945788 (literal and pandas identical). Every swapped n has L strictly below the current merged.csv L (see `docs/wins.json`, `data/vs_current_merged.json`). Sum of L reduction over the 158 n: 0.1060 (max 0.0077 at n=979; 130 of them gain >1e-9). The score of the current merged.csv was not re-run here; the stale merged-literal-official.json says 22.2039, so expect roughly 0.004 gain. merged.csv was not modified.

Winning sources: alexchilton/fault-line-dp-basin-hopping-lb-22-2620 (115 n; c0.csv, merged.csv), kmmmvs/squares-public-merge-daily (41 n), lucifer19/think-select-reclose (2 n). Original authors retain credit; data only, never executed.

## Reachability (anonymous)
- Gallery: reachable (data/index.json, data/{n}.json); meta "updated 09 Nov 2026"; no n with gallery L below ours.
- Kaggle datasets: `https://www.kaggle.com/api/v1/datasets/download/{owner}/{slug}` works anonymously. Fetched kmmmvs best/private packings, dariushafshar certified-packings, lightningv08 layout-atlas, busyaprime fq-quenched, krisdreemurr big-universe-sides (n>1000, not used), cdeotte, prvsiyan, lostdecimal27, alimaadl3lim3d, skarin, arvindnandigam, sigmaborov qcc-frozen-* (quantum-circuit assets, no packings). No dataset CSV beats merged.csv.
- Kaggle kernels: `POST /api/i/kernels.KernelsService/ListKernels` (competitionId 166661) and `GET /api/v1/kernels/output?userName=&kernelSlug=` work anonymously. sigmaborov/squares-in-squares-certified-contact-polish v18 (LB 22.2043) fetched: no n beats merged.csv. Top-level CSV outputs of 61 of 109 kernels fetched (`poc/fetch_kernels.py`); CSVs in nested output folders were not downloaded.

## Method
`poc/scan.py`: dedupe per-n packings, bbox-side prefilter (bbox < ours - 1e-11), `poc/verify.py`: repo exact evaluate (first valid, up to 8 per n), `poc/true_wins.py`: re-evaluate current merged.csv rows, `poc/build_and_score.py`: build and score. Gains smaller than 1e-11 were ignored; some wins gain only ~1e-11.

`data/` is 4.6 GB of downloaded untrusted data (zips, src, kernels); delete or gitignore before committing.
