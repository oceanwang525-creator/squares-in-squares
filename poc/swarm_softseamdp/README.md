# Hypothesis H100 (`softseamdp`): Slack-Tolerant Staircase DP + Seam Melt + `lpslack` + `mechwalk`

## Summary
Hypothesis H100 replaces the strict rigid cutoff (`dL_rigid < 0`) of previous staircase DPs with a slack-tolerant candidate search (`dL_rigid = (K + max(f_i)) - L_inc(K, D) <= +6.0e-3`, `m in {2, 3}`, `K = 16..31`), builds each multi-pocket seam via `build_cert.py` (with C++ masked seam-overlap melting / micro-expansion when minor overlaps occur), and frees all internal angles across the seam using `lpslack` (`dual_non90` + `non90`) + `mechwalk_compress`.

Why it works:
- In multi-pocket assemblies (e.g., `n=852` pairing `(9,17,+5)<739>` with `f=0.6841436` and `(20,11,+6)<3007T>` with `f=0.6862194`), freeing angles across the boundary allows the wider sub-pocket to compress right down to the narrower sub-pocket's bounding span (`29.686219 -> 29.6841436094`), yielding an extra `-2.075e-3` compression beyond rigid staircase composition.
- Furthermore, even positive-slack DP seams like `n=793` (`dL_rigid = +1.707e-3`, `(9,16,+4)<684T> + (19,10,+5)<793T>`) unlock a `-2.673e-3` big-move drop (`4.380e-3` total compression below rigid!) once `lpslack` is followed by `mechwalk_compress`.

## PR5 Certified Wins (vs `origin/main` `10a15e8`, `literal == pandas`, Final PR before Retirement)

After refreshing the union incumbent against `310ff68` (`312` bricks across `gstep`, `pocketpeel`, `b7xplant`, `vstrip492`, and `watchxplant` + `75` rescanned donors -> `1,315` Pareto pocket records and `240` qualifying DP cells, including `55` cells with new non-self compositions), Wave 6 Batch A (`pslp_ratchet.py`) landed **4 strict wins** (`literal == pandas`, verified against full `origin/main` `10a15e8` portfolio):

| `n` | `K` | `D` | Seam Composition | `L_inc` (`origin/main` `10a15e8`) | `L_new` (`softseamdp`) | `dL` | `ds_L2n` (`L^2/n`) |
| ---: | ---: | ---: | :--- | :--- | :--- | ---: | ---: |
| **970** | 31 | 9 | `(10,19,+4)<908> + (20,11,+5)<970>` | `31.637686864998290171` (`merged.csv`) | `31.637086208288271655` | **`-6.006567e-04`** | `-3.918187e-05` |
| **908** | 30 | 8 | `(19,11,+4)<791> + (10,19,+4)<908>` | `30.627807750446854375` (`merged.csv`) | `30.627635758528806102` | **`-1.719919e-04`** | `-1.160291e-05` |
| **977** | 31 | 16 | `(20,9,+7)<3026> + (11,21,+9)<689>` | `31.744187861759593775` (`merged.csv`) | `31.744087108776062180` | **`-1.007530e-04`** | `-6.547219e-06` |
| **915** | 30 | 15 | `(18,8,+6)<3180> + (12,21,+9)<3205>` | `30.740322728302614957` (`merged.csv`) | `30.740322725768730590` | **`-2.533884e-09`** | `-1.702567e-10` |
| **Total (4)** | | | | | | **`-8.734041e-04`** | **`-5.733217e-05`** |

### Retirement Summary (`WORKED` / `FAILED`)
- **WORKED**:
  - Cross-donor 2-pocket/3-pocket staircase DP with positive rigid slack (`dL_rigid <= +8.0e-3`), masked C++ seam-overlap melting, `lpslack` (`dual_non90`), and kink-safe `pslp.squeeze` (including multi-kick `center=0.08..0.4` + angle perturbations to escape local `lpslack` stalls, e.g., `n=632`, `n=798`).
  - Delivered 28 strict dual-certified wins across PR1–PR5 (`n=852` `-4.384e-3`, `n=793` `-2.673e-3`, `n=970` `-6.01e-4`, `n=492` `-2.31e-4`, `n=908` `-1.72e-4`, `n=977` `-1.01e-4`, `n=910` `-6.85e-5`, `n=405` `-3.63e-5`, etc.).
- **FAILED**:
  - Remaining positive-slack DP cells at `10a15e8` either fail rigid/melt seam feasibility (`849, 538, 739, 852, 913, 853` in Wave 6 Batch A) or stall above `L_inc` (e.g., `n=850` at `+2.613e-3`); multi-pocket seam recomposition no longer yields `>= 1e-3` wins without new `band3`/`peel` sub-brick families (`RETIRED` per `#717` / `#725`).

## PR4 Merged Wins (PR #118, `79fd5e2`, already in `origin/main` `10a15e8`)

After rebuilding the union incumbent against `99c95bd` (`180` bricks including `77` from `swarm_pocketpeel` and `45` from `swarm_b7xplant`, plus `67` rescanned donors -> `1,332` Pareto pocket records and `240` qualifying DP cells) and pairing `build_and_polish` with multi-kick `pslp.squeeze` (`center=0.08..0.4` + angle perturbations) + `lpslack`, PR4 lands **5 strict wins** (`literal == pandas`, all strictly beating `origin/main` `310ff68` `merged.csv`):

| `n` | `K` | `D` | Seam Composition | `L_inc` (`origin/main` `310ff68`) | `L_new` (`softseamdp`) | `dL` | `ds_L2n` (`L^2/n`) |
| ---: | ---: | ---: | :--- | :--- | :--- | ---: | ---: |
| **405** | 20 | 5 | `(4,5,+1)<26> + (16,14,+4)<4582>` | `20.633982502101998000` (`merged.csv`) | `20.633946160768035211` | **`-3.634133e-05`** | `-3.703041e-06` |
| **582** | 24 | 6 | `(6,10,+2)<293> + (16,14,+4)<4582>` | `24.633947688045686680` (`merged.csv`) | `24.633946160805644070` | **`-1.527240e-06`** | `-1.292851e-07` |
| **632** | 25 | 7 | `(14,9,+3)<683> + (10,16,+4)<909>` | `25.637915636585048401` (`merged.csv`) | `25.637914616487726357` | **`-1.020097e-06`** | `-8.276319e-08` |
| **798** | 28 | 14 | `(15,7,+5)<450> + (12,21,+9)<3169>` | `28.740330075437663539` (`merged.csv`) | `28.740330041124812218` | **`-3.431285e-08`** | `-2.471586e-09` |
| **740** | 27 | 11 | `(9,16,+5)<3110> + (18,9,+6)<740>` | `27.705882355525172264` (`merged.csv`) | `27.705882352943844090` | **`-2.581328e-09`** | `-1.932918e-10` |
| **Total (5)** | | | | | | **`-3.892556e-05`** | **`-3.917754e-06`** |

## PR3 Merged Wins (PR #106, `7d14a37`, already in `origin/main` `310ff68`)

After rescanning 117 donors updated through `cc58c7e` (`67,183` gadgets -> `245` qualifying DP cells, including `32` new non-self negative-`dL_rigid` compositions) and integrating kink-safe `pslp.squeeze` (`poc/row_ratchet_980/poc/pslp.py`) into `pslp_ratchet.py`, Batch A1 landed 5 strict wins (`literal == pandas`):

| `n` | `K` | `D` | Seam Composition | `L_inc` (`origin/main` `0e74cfc`) | `L_new` (`softseamdp`) | `dL` | `ds_L2n` (`L^2/n`) |
| ---: | ---: | ---: | :--- | :--- | :--- | ---: | ---: |
| **683** | 26 | 7 | `(9,14,+3)<683> + (16,12,+4)<849>` | `26.636881731629920577` (`merged.csv`) | `26.636881539888010148` | **`-1.917419e-07`** | `-1.495580e-08` |
| **796** | 28 | 12 | `(10,18,+6)<585> + (18,9,+6)<740>` | `28.705882375792044354` (`poc/swarm_lpslack/final.csv`) | `28.705882352943943156` | **`-2.284810e-08`** | `-1.647927e-09` |
| **849** | 29 | 8 | `(16,13,+4)<535> + (12,16,+4)<849>` | `29.636881726566983479` (`merged.csv`) | `29.636881713372210169` | **`-1.319477e-08`** | `-9.212060e-10` |
| **736** | 27 | 7 | `(16,13,+4)<535> + (10,14,+3)<736>` | `27.634362446166717285` (`merged.csv`) | `27.634362441139806626` | **`-5.026911e-09`** | `-3.774877e-10` |
| **642** | 25 | 17 | `(7,16,+8)<592> + (18,8,+9)<642>` | `25.817265879748838000` (`merged.csv`) | `25.817265878688882111` | **`-1.059956e-09`** | `-8.524973e-11` |
| **Total (5)** | | | | | | **`-2.338717e-07`** | **`-1.798767e-08`** |

## PR2 Merged Wins (PR #92, `b4e9208`, already in `origin/main` `0e74cfc`)

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
