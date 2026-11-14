# `poc/swarm_b8xplant` — Sub-Brick Transplant, Localized Contact-Jacobian SVD & H33/H34/H49 Standalone Brick Compression

Hypotheses **H25 + H28 + H33 + H34 + H49** targeting modular rectangular sub-brick extraction, standalone sub-brick Contact-Jacobian SVD mechanism walks (single singular modes + H34 diagonal mode pairs `(v_i ± v_j) / sqrt(2)`), shake-free SAT L-BFGS re-jamming (`slp_utils.cpp_batch_melt`, `n_ruin_iters=0`), and pinwheel / direct-region transplant + joint `(x, y, theta)` SLP polish.

## Certified REAL WIN in `final.csv` (vs `origin/main` `62aed22`)

| `n` | `L_inc` (`62aed22`) | `L_new` (`literal == pandas`) | `dL` | `ds = d(L^2 / n)` | Method / Recipe |
|---:|---:|---:|---:|---:|:---|
| **639** | `25.774501349466644129` | **`25.774498196051952238`** | **`-3.153415e-06`** | **`-2.543903e-07`** | Direct BR `10x18 (+8)` swap (`x639_flipx`) with compressed `brick_B8_10x18_NEW.json` (`f = 0.774495438`) + seam melt + `joint_xytheta_slp` (`non0/90 = 177/639`) |

## Earlier Gate / Scale Wins Superseded by Newer `origin/main` Merges (Not Exported in `final.csv`)

- **`n=800`** (H25 15-min Gate `xplant800_flipx_r5` vs `31dab295`): reached `L = 28.774609387604921545` (`dL = -7.648570e-04` vs `28.77537424459916150`), later superseded by `PR #40` (`28.774579276819314902`). Direct brick swap against `PR #40` reached `28.774582606564784804` (`dL = +3.330e-06`, miss).
- **`n=973`** (Track E `pw_melt_21x10+10x21_m31`): reached `L = 31.693306084945283364` (`dL = -3.300231e-03` vs `31.696606316`), superseded by `PR #44` (`31.681831637628565313`).
- **`n=850`** (Track E `pw_10x20+18x9_m29`): reached `L = 29.642572115518180000` (`dL = -8.977614e-07` vs `3c6b593`), matching `PR #44` (`29.642572115518184033`, tie within `4.03e-15`).
- **`n=909`** (Direct `clean_632_17x11` swap vs `d8b3699`): reached `L = 30.640084102102789403` (`dL = -2.178372e-07` vs `30.640084319940018183`), superseded by `PR #45` (`30.638899005608936646`).

## Standalone Sub-Brick `mechanism_walk` + H34 Diagonal Pairs (`data/brick_*.json`)

| Brick | `(w, h, g)` | `f0` | `f_walked` | `df` |
|:---|:---|---:|---:|---:|
| `B5_20x10_n972` | `(20, 10, +5)` | `0.656669305` | **`0.642569265`** | **`-1.410004e-02`** |
| `clean_632_17x11_n909` | `(17, 11, +4)` | `0.640784552` | **`0.638861557`** | **`-1.922995e-03`** |
| `B5_22x8` | `(22, 8, +5)` | `0.679381250` | **`0.678159029`** | **`-1.222221e-03`** |
| `B6_24x8` | `(24, 8, +6)` | `0.678889347` | **`0.678192665`** | **`-6.966820e-04`** |
| `B10_22x11` | `(22, 11, +10)` | `0.756006961` | **`0.755888311`** | **`-1.186502e-04`** |
| `B6_10x21_n972` | `(10, 21, +6)` | `0.656669406` | **`0.656551588`** | **`-1.178183e-04`** |
| `B8_10x18_NEW` | `(10, 18, +8)` | `0.774500649` | **`0.774495438`** | **`-5.211308e-06`** |
| `B8_18x8` | `(18, 8, +8)` | `0.776110986` | `0.776110986` | `0.0` (locked) |
| `B7_17x8` | `(17, 8, +7)` | `0.761536402` | `0.761536402` | `0.0` (locked) |
| `B5_22x7` | `(22, 7, +5)` | `0.679381250` | `0.679381250` | `0.0` (locked) |
| `B5_10x20_n850` | `(10, 20, +5)` | `0.642572113` | `0.642572113` | `0.0` (locked) |
| `B4_18x9_n850` | `(18, 9, +4)` | `0.642571913` | `0.642571913` | `0.0` (locked) |
| `B5_5x15_n744` | `(5, 15, +5)` | `0.757483418` | `0.757483418` | `0.0` (locked) |

## Falsified / Missed Targets Logged

- **Gate Part A (`n=858, 745`)**: `n=858` `L=29.863768814` (`dL=+8.766e-02`, `xplant858_TL_r4`: `9+18=27<=29` leaves corridor mismatch vs `B9_n858` `0.768291`); `n=745` `L=27.776115442` (`dL=+4.456e-06`, cropped `9x18` BR brick does not absorb the `18`-square strip loss).
- **Scale Track A (`n=918, 691`)**: `n=918` `no_feas` (two upright squares at `x~10, y~12` prevent melt absorption); `n=691` `L=26.831219994` (`dL=+5.511e-02`).
- **Scale Track B Localized SVD (`free_mask` on brick + collar)**: `n=858` `+6.344e-05` (gate) / `+3.117e-06` (scale), `n=800` `+1.367e-04`, `n=745` `+7.831e-06` (gate) / `+3.492e-05` (scale), `n=918` `+5.214e-06`, `n=691` `+6.348e-06`, `n=916` `+4.604e-06` (hard boundary constraint prevents local mechanism release without changing brick topology).
- **Scale Track D (`1x1` L-border extensions)**: `n=690` (from `639`, `+1.296e-02`), `n=857` (from `800`, `+1.307e-02`), `n=909` (from `850`, `+2.488e-03`).

## Layout

- `poc/b8_lib.py`: `B8_10x18_NEW` extractor/transformers, TRUE frontier loader, localized contact-Jacobian SVD modes, canonical dual certifier.
- `poc/brick_rect_lib.py`: self-contained rectangular L-BFGS / SLP and pinwheel assembly helpers.
- `poc/run_gate.py`, `poc/run_scale.py`, `poc/run_track_e_h49.py`, `poc/run_polish_800_639.py`: gate, standalone brick walk (with H34 diagonal pairs), Track E broadcast, and direct swap polish.
- `data/`: `B8_10x18_NEW.json`, walked brick checkpoints (`brick_*.json`), and certified win `win_n639.json`.
- `tests/test_b8xplant.py`: unit & `final.csv` dual-certification checks.
