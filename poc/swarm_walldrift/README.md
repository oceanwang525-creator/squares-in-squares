# `poc/swarm_walldrift` — Hypothesis H9: Directed Wall-Contact Imbalance Drift + Masked-Pocket Soft-`a` Melt + `joint_xytheta_slp`

## Mechanism
Incumbents with extreme wall-contact imbalance (one sparse bounding-box wall with `min_w <= 6` contacts vs. heavily loaded opposite/adjacent walls with `max_w >= 23`) are trapped in an asymmetric jamming basin. Hypothesis H9 releases this bottleneck via:
1. **Stage 0 (`direct_0jit_slp`)**: `0`-jitter exact signed-Jacobian `joint_xytheta_slp` + multi-margin `[1e-7, 2e-8, 5e-9, 1e-6]` `fixed_angle_lp_tighten`.
2. **Engine 1 (`cpp_batch_masked_melt` in `poc/walldrift_cpp.cpp`)**: Frozen-exterior masked-pocket (`35–90` active squares, `R = 4.8..7.4`) soft-`a` melt (`a: 0.80 -> 1.00`) with a **directed drift body potential** `E_drift = -drift_scale * sum_i (fx_i x_i + fy_i y_i)` pushing sparse-wall contacts inward toward the nearest high-clearance internal voids (`wd_mpk_void`, `wd_mpk_wall`, `wd_mpk_opp`), coupled with masked `cavity_relocate_orbit` ILS and gentle full-box `a = 1.0` unfreezing on the final stages.
3. **Engine 2 (`cpp_batch_melt`)**: `idea20`-style wall-balanced soft-`a` melt (`a: 0.84 -> 1.00`) with directed asymmetric drift (`i20_drift`), followed by `0`-jitter `joint_xytheta_slp` + multi-margin LP tighten and `poc/auto_union/poc/union.py` (`load` + `check`, literal & `pandas`) certification.

## Falsification Gate Results (`n = 754, 810, 598`)
Gate passed in `273.8s` (`poc/swarm_walldrift/docs/gate_results.json`):

| `n` | Incumbent `L` | Best Valid `L_lit == L_pd` | `dL` | Status | Source |
|---:|---:|---:|---:|---|---|
| **598** | `24.903159935` | `24.902727204157513` | **`-4.327310e-04`** | **REAL WIN** | `wd_mpk_wall_1` |
| 754 | `27.905568516` | `27.905568515890888` | `-2.141676e-15` | TIE | `direct_0jit_slp` |
| 810 | `28.905568517` | `28.905568516517923` | `-8.593571e-16` | TIE | `direct_0jit_slp` |

On `n=598` (`walls_1e-8 = [L=23, R=21, B=24, T=6]`), all top 4 polished candidates (`wd_mpk_wall_1`, `wd_mpk_opp_3`, `wd_mpk_wall_6`, `i20_drift_5`) beat `L_inc` by `-3.79e-4` to `-4.33e-4`.

## Verification
- `poc/auto_union/poc/union.py` `load` + `check` (literal & pandas) passed against `swarm` baseline on 15 entries (`598` real win + 14 ties), and after `origin/main` updated (`e7f910b`), `poc/swarm_walldrift/final.csv` was filtered to the 5 entries that still strictly tie `origin/main:poc/pell_corrugation/data/border-scan/merged.csv` (`637, 806, 916, 925, 979`; `-1.800e-15..-1.300e-15`).
- `pytest -q poc/baseline/poc/tests/ poc/swarm_walldrift/poc/tests/` passes.

## Scale-Up Results (`scale_results.json`)
Scaled H9 across 15 wall-imbalance targets (`916, 997, 979, 852, 909, 692, 695, 637, 647, 799, 806, 811, 861, 915, 925`), stopping cleanly after 17 consecutive non-real-wins (`>= 12` threshold per `#122`):
- **12 strict both-mode ties vs initial incumbent** (`-2.509e-15..-2.000e-16` via `direct_0jit_slp`, `union.py` verified): `637, 692, 695, 799, 806, 810` (gate), `811, 852, 915, 916, 925, 979, 997` (plus `754` in gate = 14 ties total). After `origin/main` `e7f910b` beat/matched `598, 692, 695, 754, 799, 810, 811, 852, 915, 997`, `final.csv` retains `637, 806, 916, 925, 979`.
- **3 misses**: `647` (`+0.000e+00`), `861` (`+1.242e-16`), `909` (dropped from `final.csv` after `union.check` failed at `mpmath` precision; `slp_utils.py` updated to serialize canonical `"s"`-prefixed 75-digit `union.final_text` and validate `union.check` inside `verify_official`).
- **Diagnosis**: Large (`n=637..997`) wall-imbalanced incumbents only yield `O(1e-15)` coupled slack under local pocket melts (`R=4.8..7.4`); only `n=598` (`dL = -4.327310e-04`, merged into `swarm` in `9e1c500` / PR #14 and subsequently improved on `origin/main` `e7f910b` to `24.9025211305`) had a deformable sparse-wall (`T=6`) bottleneck pocket.

