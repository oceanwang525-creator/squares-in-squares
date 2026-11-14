# Hypothesis H113 (`pocketpeel`): 1-Row / 1-Col Sub-Pocket Peeling + 2-Pocket Seam Assembly + `lpslack` + `pslp`

## Mechanism
1. **Donor Extraction & 1-Row / 1-Col Sub-Pocket Peeling (`poc/run_peel.py`)**:
   Extract the exact relaxed tilted corner sub-pockets from today's new `origin/main` champions (`n=852` `(20,11,+6)` & `(9,17,+5)` at `f=0.684144`; `n=912` `(9,21,+6)` & `(21,9,+6)` at `f=0.692585`; `n=974` `(9,21,+6)` & `(22,10,+7)` at `f=0.700187`; `n=975` `(9,22,+7)` & `(22,9,+7)` at `f=0.716566`) plus narrow Pareto pockets (`pockets_partb_sane.json`, `pockets_pareto_e3_sane.json`), peel 1 outer unit row or column (`peel_outer_unit_layer`), and compress via `compress_brick_slp` + `brick_mechwalk`:
   - `n852_20x11_g6_peel_top` (`20x10+6`): `f = 0.686898406`
   - `n852_20x11_g6_peel_right` (`19x11+6`): `f = 0.702731841`
   - `n852_9x17_g5_peel_top` (`9x16+5`): `f = 0.696721643`
   - `n852_9x17_g5_peel_right` (`8x17+5`): `f = 0.716575978`
   - `p_n447_11x21_g6_peel_left` (`10x21+6`): `f = 0.657548736`
   - `p_n971_10x20_g5_peel_left` (`9x20+5`): `f = 0.670738764`
   - `p_n793_10x19_g5_peel_left` (`9x19+5`): `f = 0.693813281`
   - `n912_9x21_g6_peel_top` (`9x20+6`): `f = 0.706667989`
   - `n974_22x10_g7_peel_bottom` (`22x9+7`): `f = 0.715898173`
   - `n974_22x10_g7_peel_left` (`21x10+7`): `f = 0.716699819`
   - `n975_9x22_g7_peel_right` (`8x22+7`): `f = 0.724212590`
   - `n975_9x22_g7_peel_bottom` (`9x21+7`): `f = 0.724344069`

2. **2-Pocket Pinwheel Assembly & Polishing (`poc/run_gate.py`, `poc/run_scale.py`)**:
   Assemble zero-overlap (`kx <= 0, ky <= 0, kx+ky < 0`) and corner-touching (`kx=0, ky=0`) / 1-unit seam-overlap (`kx+ky=1`) 2-pocket pinwheels, resolve minor overlaps via masked C++ melt (`cpp_batch_melt`) + `joint_xytheta_slp_tight`, and polish with `lpslack` + `mechwalk_compress` + `pslp.squeeze` (`poc/row_ratchet_980/poc/pslp.py`).

## PR1 Certified Wins (vs `origin/main` `cc58c7e`, `literal == pandas`)

| `n` | `K, D` | Pair & Method | `L_inc` (`origin/main` `cc58c7e`) | `L_new` (`pocketpeel`) | `dL` | `ds` (`d(L/sqrt(n))`) | `ds_L2n` (`d(L^2/n)`) |
| ---: | :---: | :--- | :--- | :--- | ---: | ---: | ---: |
| **914** | `30, +14` | `B7_8x22_vs_right(8x22+7) + T(22x8+7)` (`kx=0, ky=0`, melt + `lpslack` + `mw` + `pslp`) | `30.73111626861411572025` | `30.72975429984269149810` | **`-1.361969e-03`** | `-4.504992e-05` | `-9.158401e-05` |
| **795** | `28, +11` | `n852_9x17_g5_peel_top(9x16+5) + n852_20x11_g6_peel_right(19x11+6)` (`kx=0, ky=-1`, direct + `lpslack` + `mw` + `pslp`) | `28.70340392291307166515` | `28.70273024996269759297` | **`-6.736730e-04`** | `-2.389272e-05` | `-4.864523e-05` |
| **Total (2)** | | | | | | **`-6.894264e-05`** | **`-1.402292e-04`** |

## Verification
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/highrpeel_venv/bin/python -m pytest -v poc/swarm_pocketpeel/tests/test_pocketpeel.py
BASE_SHA=$(git rev-parse origin/main) HEAD_SHA=$(git rev-parse HEAD) /tmp/highrpeel_venv/bin/python .github/scripts/check_final_csv.py
```
