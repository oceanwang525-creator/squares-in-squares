# Hypothesis H125 (`poc/swarm_armslack2`): Whole-Packing `armslack` (0-deg Boundary Liberation) + 4-Corner Reduced `pslp` + `lpslack` + `mechwalk`

## Method

1. **Arm-Adjacent `0 deg` Layer Liberation (`arm_model.py`, `armslack2_lib.py`)**: Identifies tilted arms/bricks (`> 2 deg`) and 1-2 bordering layers (`d <= 1.65..2.75`) of axis-aligned (`0 deg`) grid squares, plus `act_reach = 3.5` (with `2.6` fallback) in the 4-corner auto-partitioner (`BL_static`, `TR_rigid`, `TL_rigid`, `BR_rigid`).
2. **4-Corner Reduced `pslp` + `lpslack` (`dual_all`) + Masked C++ Interface Melt + `mechwalk`**:
   - Multi-restart 4-corner reduced `pslp.squeeze` (`act_reach=3.5`) followed by full-row `pslp.squeeze` and `lpslack.optimize_group` (`slp_modes=("dual_all", "dual_non90", "non90")`, liberating high-dual `0 deg` boundary squares).
   - Masked C++ interface melt (`highd_ils.cpp:batch_masked_melt_solve`) with small `1..5 deg` boundary-layer tilt kicks and inward void nudges on full interface layers, per-arm pockets, and corner junctions.
   - SVD soft-mode mechanism walk (`mechwalk_lib`) + `pslp` + `lpslack` polish.
3. **Dual Certification (`portfolio.py`, `verify_final.py`)**: Canonical `s`-prefixed export via `poc/auto_union/poc/union.py` verified in both `literal` and `pandas` transport modes against the live portfolio (`merged.csv` + all `poc/*/final.csv`).

## Certified Gate Wins (6/13 vs `origin/main` `5b1abb1`, `literal == pandas`)

| `n` | `(m, r)` | Incumbent `L` (`merged.csv`) | New `L` (`literal == pandas`) | `dL` | `ds_L2n` |
|---:|:---:|---:|---:|---:|---:|
| `857` | `(29, +16)` | `29.760808000270924` | `29.760462481591437` | `-3.455187e-04` | `-2.399733e-05` |
| `973` | `(31, +12)` | `31.673484444203384` | `31.673423567905926` | `-6.087630e-05` | `-3.963335e-06` |
| `640` | `(25, +15)` | `25.785289212668162` | `25.785280838171167` | `-8.374497e-06` | `-6.748087e-07` |
| `974` | `(31, +13)` | `31.692581933829140` | `31.692580887395498` | `-1.046434e-06` | `-6.809894e-08` |
| `934` | `(30, +34)` | `30.962413358186310` | `30.962413301783187` | `-5.640312e-08` | `-3.739565e-09` |
| `687` | `(26, +11)` | `26.710374831984527` | `26.710374818696886` | `-1.328764e-08` | `-1.033240e-09` |

**Total across 6 PR1 strict wins (merged in PR #117 `20fe263`)**: `total_ds_L2n = -2.870834e-05`.

## PR2 Certified Scaling Wins (`4/55` vs `origin/main` `10a15e8`, `literal == pandas`)

| `n` | `(m, r)` | Incumbent `L` (`merged.csv`) | New `L` (`literal == pandas`) | `dL` | `ds_L2n` |
|---:|:---:|---:|---:|---:|---:|
| `971` | `(31, +10)` | `31.638664886500045` | `31.638469608749843` | `-1.952778e-04` | `-1.272566e-05` |
| `505` | `(22, +21)` | `22.930439905847382` | `22.930439902954835` | `-2.892547e-09` | `-2.626827e-10` |
| `409` | `(20, +9)` | `20.739455560912987` | `20.739455558947787` | `-1.965200e-09` | `-1.993016e-10` |
| `463` | `(21, +22)` | `21.968003783280250` | `21.968003781943390` | `-1.336860e-09` | `-1.268602e-10` |

**Total across 4 PR2 strict wins**: `total_ds_L2n = -1.272625e-05` (with `n=686` `-1.608e-6` already superseded in `10a15e8` and `n=540, 684, 908` below `1e-9`).

## What Worked & Failed (Big-Move Self-Judgment per `#717`)

- **WORKED**: Liberating 1–2 boundary `0 deg` layers via 4-corner reduced `pslp.squeeze` (`act_reach=3.5, 2.6`) + `lpslack(dual_all)` captured `-6e-5..-3.5e-4` polish on freshly assembled pinwheels (`n=857, 973, 971`) before full `pslp`/`gstep` sweeps reached them.
- **FAILED (`>= 1e-3` Big-Move gate)**:
  - Masked C++ interface melt (`a_start=0.94`, `L_end=-0.0004..-0.0022`) had `0/20` feasible melts across the 13 gate targets: local `1–5 deg` boundary-layer tilts cannot jump discrete grid/brick indices on already-dense pinwheel packings.
  - Tier-1 Batch 1 (`docs/scale_t1_b1_summary.json`) across the top 55 unliberated (`slight=0..10`) `m=20..31, r=+7..+22` targets gave 50/55 `MISS`, 3 `1e-9` crumbs, `n=686` `-1.61e-6` (superseded in `10a15e8`), and `n=971` `-1.95e-4` (`0` wins `>= 1e-3`). Retired per `#717`.

## Reproduction & Checks

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/ratchetall_venv/bin/python -u poc/swarm_armslack2/poc/run_gate.py --workers 2 --threads 1 --tag gate --resume
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/ratchetall_venv/bin/python -u poc/swarm_armslack2/poc/run_gate.py --workers 2 --threads 1 --tier1-only --tag scale_t1_b1 --resume
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/ratchetall_venv/bin/python poc/swarm_armslack2/poc/verify_final.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/ratchetall_venv/bin/python -m pytest -v poc/swarm_armslack2/tests/test_armslack2.py
```
