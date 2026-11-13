# `poc/swarm_homology` — Hypothesis H10: 4-Block `(BL, TR, A, B)` Modular Homology Transplant + End-Cap Strip (`+8/-8`) + Seam Melt

## Mechanism
Across `n = 600..1000`, `non0 > 0` incumbents decompose cleanly around `(cut, cut)` (`cut = 9` for the winning family) into 4 decoupled modules:
1. `BL`: bottom-left corner block (`x < cut, y < cut`, typically `80` squares),
2. `TR`: pure upright `(m - k_off) x (m - k_off)` grid (`x > cut, y > cut`),
3. `Arm A` (`x < cut, y > cut`) and `Arm B` (`x > cut, y < cut`): decoupled tilted corridor modules.

Cross-`m` homology takes `BL/TL/BR/TR` 1-unit crops (`x, y <= L - 1.05`) from donor `n_src` packings with `prot <= 1..2` protruding squares, relocates protruding squares into true outer wall gaps discovered by `wall_gap_slots()` (single inward pocket center `1.2` units inside the box), and melts the pocket with `L_inc`-anchored `sched_a`/`sched_b` schedules (`a: 0.78 -> 1.00`, `highd_ils.cpp` keeping best overall stage `unf_E <= best_overall_E`) followed by `joint_xytheta_slp` + multi-margin `fixed_angle_lp_tighten`.

## Certified Real Wins (literal == pandas vs `origin/main` `252fe8b` `merged.csv` & all `poc/*/final.csv`)

| `n` | `origin/main` `merged.csv` `L` | Best Valid `L_lit == L_pd` | `dL` vs `merged.csv` | Status | Source |
|---:|---:|---:|---:|---|---|
| **850** | `29.656854249492600` | `29.649235294877550` | **`-7.618955e-03`** | **REAL WIN** | `crop_909_BL_w0_top_R10.4_s5` (beats `merged.csv` & `swarm_replace45`) |
| **686** | `26.705913986119564` | `26.701648540631480` | **`-4.265445e-03`** | **REAL WIN** | `crop_739_BL_w0_top_R7.9_s1` (carried from PR #22 `a70ca15`) |
| **910** | `30.679149454240586` | `30.677024325828058` | **`-2.125128e-03`** | **REAL WIN** | `910_971crop_tuck1_R5.9_s4` (carried from PR #16 `c4e3272`) |

## 15-Minute Gate Results (against `origin/main` `e7f910b` `merged.csv`)

| `n` | `origin/main` `L` | Best Valid `L_lit == L_pd` | `dL` vs `origin/main` | Status | Source |
|---:|---:|---:|---:|---|---|
| **910** | `30.679149454` | `30.677024325828058` | **`-2.125128e-03`** | **REAL WIN** | `910_971crop_tuck1_R5.9_s4` (`win_n971` crop + protruding-square tuck + masked pocket melt) |
| 922 | `30.826655323` | `30.827162586636323` | `+5.0726e-04` | miss | `922_w862arm_flipB_R4.8_s1` |
| 863 | `29.843941662` | `29.850256733513790` | `+6.3151e-03` | miss | `863c_923strip1_R5.1_s7` |
| 805 | `28.843441055` | — | — | miss | `805_749cap` (`min_pen = 7.56e-3`) |

## Files
- Solver: [h10_gate.py](poc/h10_gate.py), [h10_scale.py](poc/h10_scale.py), [utils.py](poc/utils.py), [joint_slp.py](poc/joint_slp.py), [cpp_wrap.py](poc/cpp_wrap.py), [highd_ils.cpp](poc/highd_ils.cpp)
- Unit tests: [test_homology.py](poc/tests/test_homology.py)
- Certified wins: [final.csv](final.csv), [win_n686.csv](data/win_n686.csv), [win_n850.csv](data/win_n850.csv), [win_n910.csv](data/win_n910.csv)
- Logs/JSON: [gate_910.json](docs/gate_910.json), [scale_results.json](docs/scale_results.json)
