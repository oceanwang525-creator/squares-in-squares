# `poc/swarm_homology` — Hypothesis H10: 4-Block `(BL, TR, A, B)` Modular Homology Transplant + End-Cap Strip (`+8/-8`) + Seam Melt

## Mechanism
Across `n = 600..1000`, `non0 > 0` incumbents decompose cleanly around `(cut, cut)` (`cut = 9` for the winning family) into 4 decoupled modules:
1. `BL`: bottom-left corner block (`x < cut, y < cut`, typically `80` squares),
2. `TR`: pure upright `(m - k_off) x (m - k_off)` grid (`x > cut, y > cut`),
3. `Arm A` (`x < cut, y > cut`) and `Arm B` (`x > cut, y < cut`): decoupled tilted corridor modules.

Cross-`m` homology adds/strips `1`-unit outer end-cap strips (or `BL/TL/BR/TR` 1-unit crops `x, y <= L - 1.05`), reducing the cross-basin jump to a 1–2 square local seam protrusion or corner void resolved via `a: 0.78 -> 1.00` masked-pocket `cavity_relocate_orbit` melt + `joint_xytheta_slp` + multi-margin `fixed_angle_lp_tighten`.

## 15-Minute Gate Results (against `origin/main` `e7f910b` `merged.csv`)

| `n` | `origin/main` `L` | Best Valid `L_lit == L_pd` | `dL` vs `origin/main` | Status | Source |
|---:|---:|---:|---:|---|---|
| **910** | `30.679149454` | `30.677024325828058` | **`-2.125128e-03`** | **REAL WIN** | `910_971crop_tuck1_R5.9_s4` (`win_n971` crop + protruding-square tuck + masked pocket melt) |
| 922 | `30.826655323` | `30.827162586636323` | `+5.0726e-04` | miss | `922_w862arm_flipB_R4.8_s1` |
| 863 | `29.843941662` | `29.850256733513790` | `+6.3151e-03` | miss | `863c_923strip1_R5.1_s7` |
| 805 | `28.843441055` | — | — | miss | `805_749cap` (`min_pen = 7.56e-3`) |

*(Note: during the gate, `origin/main` updated at `e7f910b`, lowering `n=922` to `30.826655323`, below `n=862` + `1`. Our `n=910` win `30.677024325828058` strictly beats `origin/main` `30.679149454241` by **`dL = -2.125128e-03`**, `ds = -6.927e-05`.)*

## Files
- Solver: [h10_gate.py](poc/h10_gate.py), [utils.py](poc/utils.py), [joint_slp.py](poc/joint_slp.py), [cpp_wrap.py](poc/cpp_wrap.py), [highd_ils.cpp](poc/highd_ils.cpp)
- Unit tests: [test_homology.py](poc/tests/test_homology.py)
- Certified wins: [final.csv](final.csv), [win_n910.csv](data/win_n910.csv)
- Gate logs/JSON: [gate_910.json](docs/gate_910.json)
