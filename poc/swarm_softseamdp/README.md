# Hypothesis H100 (`softseamdp`): Slack-Tolerant Staircase DP + Seam Melt + `lpslack` + `mechwalk`

## Summary
Hypothesis H100 replaces the strict rigid cutoff (`dL_rigid < 0`) of previous staircase DPs with a slack-tolerant candidate search (`dL_rigid = (K + max(f_i)) - L_inc(K, D) <= +6.0e-3`, `m in {2, 3}`, `K = 16..31`), builds each multi-pocket seam via `build_cert.py` (with C++ masked seam-overlap melting / micro-expansion when minor overlaps occur), and frees all internal angles across the seam using `lpslack` (`dual_non90` + `non90`) + `mechwalk_compress`.

Why it works:
- In multi-pocket assemblies (e.g., `n=852` pairing `(9,17,+5)<739>` with `f=0.6841436` and `(20,11,+6)<3007T>` with `f=0.6862194`), freeing angles across the boundary allows the wider sub-pocket to compress right down to the narrower sub-pocket's bounding span (`29.686219 -> 29.6841436094`), yielding an extra `-2.075e-3` compression beyond rigid staircase composition.

## PR1 Certified Wins (vs `origin/main` `21f7df8`, `literal == pandas`)

| `n` | `K` | `D` | Seam Composition | `L_inc` (`origin/main` `21f7df8`) | `L_new` (`softseamdp`) | `dL` | `ds` (`L/sqrt(n)-1`) | `ds_L2n` (`L^2/n`) |
| ---: | ---: | ---: | :--- | :--- | :--- | ---: | ---: | ---: |
| **852** | 29 | 11 | `(9,17,+5)<739> + (20,11,+6)<3007T>` | `29.688527389492101859` (`merged.csv`) | `29.684143609397919235` | **`-4.383780e-03`** | `-1.501858e-04` | `-3.054891e-04` |
| **910** | 30 | 10 | `(16,9,+4)<684> + (11,21,+6)<447>` | `30.656420974091060289` (`carried_lpslack_wins`) | `30.656352497397347824` | **`-6.847669e-05`** | `-2.269980e-06` | `-4.613732e-06` |
| **686** | 26 | 10 | `(9,17,+5)<739> + (16,9,+5)<686T>` | `26.696716552703286389` (`merged.csv`) | `26.696711174745917834` | **`-5.377957e-06`** | `-2.053314e-07` | `-4.185825e-07` |
| **849** | 29 | 8 | `(13,16,+4)<535T> + (16,12,+4)<683T>` | `29.636881743404257166` (`carried_lpslack_wins`) | `29.636881726566983479` | **`-1.683727e-08`** | `-5.778537e-10` | `-1.175511e-09` |
| **853** | 29 | 12 | `(10,20,+6)<3002> + (18,9,+6)<796T>` | `29.705882376792042049` (`merged.csv`) | `29.705882363831784515` | **`-1.296026e-08`** | `-4.437508e-10` | `-9.026867e-10` |
| **585** | 24 | 9 | `(5,13,+3)<4492> + (18,9,+6)<796T>` | `24.705882369141985921` (`merged.csv`) | `24.705882357996394081` | **`-1.114559e-08`** | `-4.608135e-10` | `-9.414075e-10` |
| **Total (6)** | | | | | | | **`-1.526626e-04`** | **`-3.105245e-04`** |

*(Note: `n=974` (`31.705882365...`, `dL=-4.377e-6` vs `f542119`) was also solved at the gate, and omitted from PR1 `final.csv` because `cleanpeel2` PR #82 holds `31.700186957`.)*

## Pipeline & Reproducibility
All commands enforce `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` and `workers <= 2`:

1. Build `origin/main` union & donor catalog (`merged.csv` + `poc/*/final.csv` + `poc/swarm_gstep/data/brick_*.json`):
   ```bash
   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/highrpeel_venv/bin/python \
     poc/swarm_softseamdp/poc/build_union_and_donors.py
   ```
2. Extract clean-cut `gadlib` pockets & rescan `n in {492, 852, 911, 914, 974, 975}` (`0.0625` grid) via `poc/pocket_lift_scan3/poc/scan_pockets.py`, then sanitize & Pareto-merge (`merge_and_sanitize.py`):
   ```bash
   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/highrpeel_venv/bin/python \
     poc/swarm_softseamdp/poc/add_gadlib_pockets.py
   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/highrpeel_venv/bin/python \
     poc/swarm_softseamdp/poc/merge_and_sanitize.py
   ```
3. Run slack-tolerant staircase DP (`K=16..31`, `m in {2,3}`, `dL_rigid <= +6.0e-3`, top-3 distinct seams/cell):
   ```bash
   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/highrpeel_venv/bin/python \
     poc/swarm_softseamdp/poc/dp_soft.py \
     --pockets poc/swarm_softseamdp/data/pockets_all_sane.json \
     --report poc/swarm_softseamdp/data/union_report.json \
     --k-min 16 --k-max 31 --slack-max 0.006 --top-k 3 \
     --out poc/swarm_softseamdp/data/pred_all.json
   ```
4. Build, melt minor seam overlaps, polish with `lpslack` + `mechwalk_compress`, and export certified `poc/swarm_softseamdp/final.csv`:
   ```bash
   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/highrpeel_venv/bin/python \
     poc/swarm_softseamdp/poc/build_and_polish.py \
     --pred poc/swarm_softseamdp/data/pred_all.json \
     --report poc/swarm_softseamdp/data/union_report.json \
     --csv poc/swarm_softseamdp/data/donors_all.csv \
     --pockets poc/swarm_softseamdp/data/pockets_all_sane.json \
     --n-list 974,910,853,849,686,585,914,502 --max-seams 2 --workers 2 \
     --out poc/swarm_softseamdp/docs/gate_results.json
   ```

## Verification
- `/tmp/highrpeel_venv/bin/python -m pytest -v poc/swarm_softseamdp/tests/test_softseamdp.py`
- `BASE_SHA=$(git rev-parse origin/main) HEAD_SHA=$(git rev-parse HEAD) /tmp/highrpeel_venv/bin/python .github/scripts/check_final_csv.py`
