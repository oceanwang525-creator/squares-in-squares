# Hypothesis H101 — 2-D Overhang (`kx + ky <= 2, |I| = 0`) Pinwheel Composer (`swarm_interlock2`)

## Summary

Hypothesis **H101** composes two `(w_i, h_i, g_i)` sub-pockets (including 90-degree transposes) placed in diagonally opposite corners (`BL` and `TR`) with **1-axis or 2-axis overhang** (`kx = max(0, w_1 + w_2 - K)`, `ky = max(0, h_1 + h_2 - K)`, `kx + ky in {1, 2}`) and **zero bounding-box intersection** (`|I| = kx * ky == 0`), tiled with upright squares on the remaining `(K - w_1) x (K - h_2)` and `(K - w_2) x (K - h_1)` blocks:

- **Regime 1 (`zero_gap`)**: `D = g_1 + g_2 - (kx + ky)` with `1` unit of non-overhanging axis slack (`h_1 + h_2 <= K - 1` when `kx > 0`, or `w_1 + w_2 <= K - 1` when `ky > 0`). Exact 50-digit donor extraction + upright tiling (`assemble_zero_gap_50d`) + `lpslack` + soft-mode `mechwalk_compress` polish.
- **Regime 2 (`seam_touch`)**: `D = g_1 + g_2` with exact seam contact on the non-overhanging axis (`h_1 + h_2 == K` when `kx > 0`, or `w_1 + w_2 == K` when `ky > 0`). Assembled with C++ masked penalty melt (`run_cpp_masked_melt`) + `brick_slp` + `lpslack`.

## Sources & Data

- `poc/pocket_lift_scan3/data/pockets_pareto_e3_sane.json` (1,516 sanitized proper sub-pockets)
- Bricks from `poc/swarm_gstep/data/brick_*.json`, `poc/swarm_b7xplant/data/brick_*.json`, `poc/swarm_thinb4/data/brick_*.json`
- Scanned sub-pockets from `n in {492, 852, 911, 974, 975}` (`poc/swarm_interlock2/poc/scan_5rows.py`, donors cached in `data/donors_5rows.json`; `data/scanned_5rows.json` excluded from PR to keep all committed files under 1 MB).
- Combined index: `7,730` unique `(w, h, g)` keys (`12,797` items).

## Gate Results (`poc/swarm_interlock2/poc/run_gate.py`, 8433.6s)

### Certified Wins (`literal == pandas` via `poc/auto_union/poc/union.py`)

| `n` | `(K, D)` | `(kx, ky)` | Regime | Formula | `L_inc` | `L_lit` | `dL` |
|---:|:---:|:---:|:---|:---|---:|---:|---:|
| `748` | `(27, 19)` | `(1, 0)` | `zero_gap` | `e3_n804_7x18_g9 + e3_n748_21x8_g11` (`s00_mw`) | `27.8271519546314247` | `27.8268092691779018` | **`-3.426855e-04`** |
| `910` | `(30, 10)` | `(1, 0)` | `zero_gap` | `thin_B5_10x19_n793_10x19_g5 + brick_B6_10x21_n972_b8_21x10_g6` (`s00_mw`) | `30.6564209740910615` | `30.6564056705494223` | **`-1.530354e-05`** |

### Non-Winning Gate Evaluations

- **Regime 1 (`zero_gap`)**: `849` (`+1.9605e-4`), `791` (`+1.2340e-3`), `640` (`+1.4342e-3`), `454` (`+2.4192e-3`), `795` (`+2.4785e-3`), `912` (`+2.6225e-3`).
- **Regime 2 (`seam_touch`)**: All 6 tested gate targets stalled above `L_inc` (`n=738`: `+1.8242e-2`, `n=852`: `+3.3635e-2`, `n=414`: `+3.3844e-2`, `n=973`: `+3.4652e-2`, `n=336`: `+6.5565e-2`, `n=579`: `+1.5166e-1`), showing that exact-touch (`h_1 + h_2 = K`) overhang seams lack the 1-unit corridor slack required to relax `L` below `L_inc`.

## Scale-Up Results (`poc/swarm_interlock2/poc/run_scale.py`, 8156.7s, 25 `zero_gap` targets)

Expanding the pocket pool with `pockets_partb_sane.json`, `poc/swarm_b8xplant/`, `poc/swarm_cleanpeel2/`, and evaluating all 4 `D2` corner symmetries `(0,0), (3,0), (0,3), (3,3)` + soft-mode `mechwalk_compress` yielded **2 additional Big-Move wins** in high-excess thin-strip (`w` or `h <= 8`) `zero_gap` pinwheels:

| `n` | `(K, D)` | `(kx, ky)` | Formula | `L_inc` | `L_lit` | `dL` |
|---:|:---:|:---:|:---|---:|---:|---:|
| `863` | `(29, 22)` | `(0, 1)` | `e3_n750_20x8_g11 + e3_n985_8x22_g12` (`s03_mw`) | `29.8439416619937890` | `29.8428057136543115` | **`-1.135948e-03`** |
| `924` | `(30, 24)` | `(0, 1)` | `e3_n807_21x8_g12 + e3_n986_8x23_g13` (`s00_mw`) | `30.8553010538996188` | `30.8550590032779526` | **`-2.420506e-04`** |

Notable near-miss in scale-up: `n=918` (`(30, 18)`, `brick_B8_18x8_g8 + e3_n980_11x23_g11`, `s30_mw`) reached `+6.4803e-04`.

## Round 3 Results (`H116 kx=1,ky=1` + `H121 w x 8` Synthesis + `pslp` Squeeze)

- **Kink-Free Free-Angle `pslp.squeeze` Tightening (`n=748`)**:
  - Applying `poc/row_ratchet_980/poc/pslp.py` (`pslp.squeeze`) + `lpslack` + `mechwalk_compress` to our `n=748` `(K=27, D=19, kx=1, ky=0)` `zero_gap` pinwheel tightened `n=748` from `27.826809269177901828` to **`27.826777074593507336`**, beating `origin/main` (`0e74cfc`: `27.826786853461192225`) by **`dL = -9.778868e-06`** (`literal == pandas`).
  - Superseded PR1/PR2 groups `[863, 910, 924]` are preserved and improved in `origin/main:merged.csv` (`0e74cfc`) and dropped from `final.csv` for PR3.
- **H116 (`kx=1, ky=1` 2-axis diagonal overhang, `poc/swarm_interlock2/poc/run_round3.py`)**:
  - Swept `14` targets across `del_cnt in {1, 2, 3}` deleted corner squares (`n in {300, 301, 305, 306, 377, 417, 418, 499, 500, 505, 754, 911, 974, 991}`) with C++ masked corner melt + `pslp.squeeze` (`0/14` wins; best realizations stalled at `sqL - L_inc >= +0.0538` for `del_cnt=1`, `>= +0.0114` for `del_cnt=2`, and `>= +0.0065` for `del_cnt=3` on `n=991`), confirming the diagonal `1x1` overlap corner cannot compress below `L_inc` without vertical/horizontal corridor slack.
- **H121 (`synth_wx8.py` + `scan_watch89.py`)**:
  - Synthesized `132` `w x 8` / `w x 7` peeled/`vstrip` bricks (`data/bricks_wx8.json`, excluded `>1MB`) and scanned `17,623` clean pockets from `88` `public_watch_nov14` / `row_ratchet_980` / `corrcasc` rows (`data/scanned_watch89.json`, excluded `>1MB`).

## Checks

- Unit tests: `/tmp/highrpeel_venv/bin/pytest -v poc/swarm_interlock2/tests/test_interlock2.py`
- CSV validation: `.github/scripts/check_final_csv.py poc/swarm_interlock2/final.csv`
