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

## PR2 Certified Win (vs `origin/main` `ccdac8f`, `literal == pandas`)
PR1 (`n=914`, `30.72975429984269149810`) merged in PR #97 (`015db05`) and is preserved in `origin/main:merged.csv`. PR2 adds a newly synthesized zero-overlap win for `n=795` beating PR #104 (`vstrip492` `28.69912621005546637760`):

| `n` | `K, D` | Pair & Method | `L_inc` (`origin/main` `ccdac8f`) | `L_new` (`pocketpeel`) | `dL` | `ds_L2n` (`d(L^2/n)`) |
| ---: | :---: | :--- | :--- | :--- | ---: | ---: |
| **795** | `28, +11` | `n852_9x17_g5_peel_top(9x16+5) + n852_BR_11x20_g6_peel_top_slp_walk_T(19x11+6)` (`kx=0, ky=-1`, direct + `mw` + `lpslack` + `pslp`) | `28.69912621005546637760` | `28.69912487031809005953` | **`-1.339737e-06`** | `-9.672778e-08` |

## PR3 Certified Wins (vs `origin/main` `10a15e8`, `literal == pandas`)
PR2's `n=795` was superseded in `origin/main` by PR #114 (`28.69887184145230235083`). Re-checking `poc/swarm_pocketpeel/data/wins/` against `origin/main` `10a15e8` (`merged.csv` + all `poc/*/final.csv`) yields 2 remaining strict wins (`final.csv`):

| `n` | `K, D` | Pair & Method | `L_inc` (`origin/main` `10a15e8`) | `L_new` (`pocketpeel`) | `dL` | `ds_L2n` (`d(L^2/n)`) |
| ---: | :---: | :--- | :--- | :--- | ---: | ---: |
| **849** | `29, +8` | `brick_H60_B4_12x16_n849 + p_n535_16x13_g4` (`id, id`, direct + `lpslack` + `mw` + `pslp`) | `29.63688171337221016936` | `29.63688153991011019007` | **`-1.734621e-07`** | `-1.211043e-08` |
| **538** | `23, +9` | `B3_5x13_n492_BR_mw + p_n538_9x18_g6_T` (`rot180, rot180`, direct + `pslp`) | `23.70588235542379891664` | `23.70588235296076402321` | **`-2.463035e-09`** | `-2.170573e-10` |
| **Total (2)** | | | | | | **`-1.232749e-08`** |

## Results Summary (`WORKED` / `FAILED`)
- **`WORKED`**:
  - Peeling 1 outer upright row/col (`peel_outer_unit_layer`) + `compress_brick_slp` + `brick_mechwalk` cleanly compressed newly landed donor corners (`docs/peel2.log`: `m793_9x17_g4_peel_top` `0.656559 -> 0.642573`, `m973_20x10_g5_peel_top` `0.735068 -> 0.678377`, `m973_10x21_g6_peel_bottom` `0.755937 -> 0.682414`).
  - `kx=0, ky=0` corner-touching melt with `id_flipy` landed `n=914` (`30.72975429984269149810`, `dL=-1.361969e-03`, merged in PR #97 `015db05`).
  - `kx=0, ky=-1` zero-overlap pinwheel with peeled `n852` bricks landed `n=795` (`28.70273024996269759297` in PR1, `28.69912487031809005953` in PR2 before PR #114).
  - `run_fast_scale.py` (`FLIPS_8` + selective `joint_xytheta_slp(rounds=6)` + `lpslack` + `mechwalk` + `pslp`) evaluated 62 ranked targets across `389` oriented bricks and found micro-wins on `n=849` (`-1.734621e-07`) and `n=538` (`-2.463035e-09`) still strict vs `10a15e8`.
- **`FAILED`**:
  - `kx=0, ky=0` square-corner pinwheels (`n=635` `9x16+16x9`, `n=912` `10x20+20x10`, `n=584` `12x12+12x12`, `n=633`, `n=911`) collide by an `f x f` square at `(m-h1, w1)` when both bricks occupy their inner corner.
  - `run_gate.py`'s `seeds[:6]` truncated `(id, flipy)`/`(flipx, flipy)` when `f_lo != f_hi` (fixed in `run_fast_scale.py` via `FLIPS_8`).
  - 1-row/1-col peeling incurs a `+0.01..+0.03` single-brick `f` penalty (`docs/peel2.log`), so cross-brick assemblies rarely beat `vstrip` / `band3` / `peelstep` (`0` new `>= 1e-3` wins across the 62 targets in `docs/fast_scale_summary.json`).

## Verification
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/highrpeel_venv/bin/python -m pytest -v poc/swarm_pocketpeel/tests/test_pocketpeel.py
BASE_SHA=$(git rev-parse origin/main) HEAD_SHA=$(git rev-parse HEAD) /tmp/highrpeel_venv/bin/python .github/scripts/check_final_csv.py
```
