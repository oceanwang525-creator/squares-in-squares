# Hypothesis H100 (`softseamdp`): Slack-Tolerant Staircase DP + Seam Melt + `lpslack` + `mechwalk`

## Summary
Hypothesis H100 replaces the strict rigid cutoff (`dL_rigid < 0`) of previous staircase DPs with a slack-tolerant candidate search (`dL_rigid = (K + max(f_i)) - L_inc(K, D) <= +6.0e-3`, `m in {2, 3}`, `K = 16..31`), builds each multi-pocket seam via `build_cert.py` (with C++ masked seam-overlap melting / micro-expansion when minor overlaps occur), and frees all internal angles across the seam using `lpslack` (`dual_non90` + `non90`) + `mechwalk_compress`.

Why it works:
- In multi-pocket assemblies (e.g., `n=852` pairing `(9,17,+5)<739>` with `f=0.6841436` and `(20,11,+6)<3007T>` with `f=0.6862194`), freeing angles across the boundary allows the wider sub-pocket to compress right down to the narrower sub-pocket's bounding span (`29.686219 -> 29.6841436094`), yielding an extra `-2.075e-3` compression beyond rigid staircase composition.
- Furthermore, even positive-slack DP seams like `n=793` (`dL_rigid = +1.707e-3`, `(9,16,+4)<684T> + (19,10,+5)<793T>`) unlock a `-2.673e-3` big-move drop (`4.380e-3` total compression below rigid!) once `lpslack` is followed by `mechwalk_compress`.

## PR2 Certified Wins (vs `origin/main` `cfe8f77`, `literal == pandas`)

| `n` | `K` | `D` | Seam Composition | `L_inc` (`origin/main` `cfe8f77`) | `L_new` (`softseamdp`) | `dL` | `ds` (`L/sqrt(n)-1`) | `ds_L2n` (`L^2/n`) |
| ---: | ---: | ---: | :--- | :--- | :--- | ---: | ---: | ---: |
| **793** | 28 | 9 | `(9,16,+4)<684T> + (19,10,+5)<793T>` | `28.659230707436795103` (`merged.csv`) | `28.656557862442773771` | **`-2.672845e-03`** | `-9.491551e-05` | `-1.931856e-04` |
| **492** | 22 | 8 | `(16,9,+5)<686T> + (5,13,+3)<4492>` | `22.701529381421047827` (`merged.csv`) | `22.701297987853754651` | **`-2.313936e-04`** | `-1.043203e-05` | `-2.135350e-05` |
| **915** | 30 | 15 | `(8,18,+6)<539T> + (21,12,+9)<915>` | `30.740334243727155101` (`merged.csv`) | `30.740333991268567890` | **`-2.524586e-07`** | `-8.346023e-09` | `-1.696319e-08` |
| **972** | 31 | 11 | `(20,10,+5)<971T> + (10,21,+6)<972>` | `31.656497298735282382` (`merged.csv`) | `31.656497289245665659` | **`-9.489617e-09`** | `-3.043796e-10` | `-6.181235e-10` |
| **798** | 28 | 14 | `(15,7,+5)<742T> + (12,21,+9)<915T>` | `28.740330079151405417` (`lift_polish2`) | `28.740330075437663539` | **`-3.713742e-09`** | `-1.314650e-10` | `-2.675042e-10` |
| **538** | 23 | 9 | `(13,5,+3)<4492T> + (9,18,+6)<796>` | `23.705882358905035206` (`swarm_liftpolish3`) | `23.705882355423798917` | **`-3.481236e-09`** | `-1.500868e-10` | `-3.067873e-10` |
| **740** | 27 | 11 | `(9,17,+5)<739> + (18,9,+6)<796T>` | `27.705882358799238724` (`lift_polish2`) | `27.705882355525172264` | **`-3.274066e-09`** | `-1.203571e-10` | `-2.451646e-10` |
| **413** | 20 | 13 | `(12,6,+6)<373> + (7,14,+7)<413T>` | `20.816982988739422928` (`merged.csv`) | `20.816982986632632164` | **`-2.106791e-09`** | `-1.036684e-10` | `-2.123827e-10` |
| **Total (8)** | | | | | | | **`-1.053567e-04`** | **`-2.145578e-04`** |

## PR1 Merged Wins (PR #87, `1afa0c2`, already in `origin/main` `cfe8f77`)
- **`n=852` (`K=29, D=11`)**: `29.688527389492101859` -> **`29.684143609397919235`** (`dL = -4.383780e-03`, `ds_L2n = -3.054891e-04`)
- **`n=910` (`K=30, D=10`)**: `30.656420974091060289` -> **`30.656352497397347824`** (`dL = -6.847669e-05`, `ds_L2n = -4.613732e-06`)
- **`n=686` (`K=26, D=10`)**: `26.696716552703286389` -> **`26.696711174745917834`** (`dL = -5.377957e-06`, `ds_L2n = -4.185825e-07`)
- **`n=849` (`K=29, D=8`)**: `29.636881743404257166` -> **`29.636881726566983479`** (`dL = -1.683727e-08`, `ds_L2n = -1.175511e-09`)
- **`n=853` (`K=29, D=12`)**: `29.705882376792042049` -> **`29.705882363831784515`** (`dL = -1.296026e-08`, `ds_L2n = -9.026867e-10`)
- **`n=585` (`K=24, D=9`)**: `24.705882369141985921` -> **`24.705882357996394081`** (`dL = -1.114559e-08`, `ds_L2n = -9.414075e-10`)

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
     poc/swarm_softseamdp/poc/merge_and_sanitize.py \
     --csv poc/swarm_softseamdp/data/donors_all.csv \
     --out poc/swarm_softseamdp/data/pockets_all_sane.json \
     poc/pocket_lift_scan3/data/pockets_pareto_e3_sane.json \
     poc/swarm_softseamdp/data/bricks_pockets.json \
     poc/swarm_softseamdp/data/gadlib_pockets.json \
     poc/swarm_softseamdp/data/rescan_o1.json \
     poc/swarm_softseamdp/data/rescan_o2.json
   ```
3. Run slack-tolerant staircase DP (`K=16..31`, `m in {2,3}`, `dL_rigid <= +6.0e-3`, top-3 distinct seams/cell):
   ```bash
   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/highrpeel_venv/bin/python \
     poc/swarm_softseamdp/poc/dp_soft.py \
     --pockets poc/swarm_softseamdp/data/pockets_all_sane.json \
     --report poc/swarm_softseamdp/data/union_report.json \
     --kmin 16 --kmax 31 --slack-max 0.006 --top-k 3 \
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
     --n-list <targets> --max-seams 2 --workers 2 \
     --out poc/swarm_softseamdp/docs/scale_wave1.json
   ```

## Verification
- `/tmp/highrpeel_venv/bin/python -m pytest -v poc/swarm_softseamdp/tests/test_softseamdp.py`
- `BASE_SHA=$(git rev-parse origin/main) HEAD_SHA=$(git rev-parse HEAD) /tmp/highrpeel_venv/bin/python .github/scripts/check_final_csv.py`
