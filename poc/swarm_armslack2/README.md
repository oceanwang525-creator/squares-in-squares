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

**Total across 6 strict wins**: `total_ds_L2n = -2.870834e-05`.

## Reproduction & Checks

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/ratchetall_venv/bin/python -u poc/swarm_armslack2/poc/run_gate.py --workers 2 --threads 1 --tag gate --resume
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/ratchetall_venv/bin/python poc/swarm_armslack2/poc/verify_final.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/ratchetall_venv/bin/python -m pytest -v poc/swarm_armslack2/tests/test_armslack2.py
```
