# Hypothesis H78: Constant-`r` Homology L-Border Cascade (`swarm_homocasc2`)

## Summary (PR #1 — Gate Breakthrough)
- **Branch**: `swarm-homocasc2` / `swarm-homocasc2-pr1` (based on `origin/main` @ `8c623fc`)
- **Result**: **2 strict REAL WINs (`n=852, 911`)**, **total `ds = -1.590375e-4`** (`literal == pandas` dual-certified against the full `origin/main` `8c623fc` portfolio).

| `n` | `(m, r)` | `L_inc` (`origin/main` `8c623fc`) | `L_win` (`literal == pandas`) | `dL` | `ds = dL / sqrt(n)` | Cascade Path |
|---:|:---:|---:|---:|---:|---:|---|
| 852 | `(29, +11)` | `29.699128044056555137` | `29.695584615062476056` | `-3.543429e-3` | `-1.213959e-4` | `911 -> 852` (`peel1_911_BL_k1.010_clamp_keepAng` + melt + `joint_xytheta_slp` + `mechwalk_compress` + `lpslack`) |
| 911 | `(30, +11)` | `30.676983696661892525` | `30.675847568497090971` | `-1.136128e-3` | `-3.764161e-5` | `972 -> 911` (`xcross_972_x20.0_y21.0_rt1` + melt + `joint_xytheta_slp` + `mechwalk_compress` + `lpslack`) |

## Geometric Mechanism & Gate Findings
1. **Why the H78 cascade broke through on `r = +11`**:
   - Newly tightened donors (`n=972` at `31.656512`, `n=911` at `30.676984`) carry `-20.5` to `-22.1` milli-units of fractional excess slack relative to `(m-1, +11)` incumbents (`911`, `852`).
   - On `972 -> 911`, scanning interior upright row+column excisions (`xcross_972_x20.0_y21.0_rt1`, removing only `1` tilted square) landed at `30.677162` (`dL = +1.78e-4`) after masked melt + `joint_xytheta_slp`, and chaining `mechwalk_compress` + `lpslack` (selective soft-angle SLP + unbounded fixed-angle LP + 80-digit uniform shrink) compressed the basin by `-1.314e-3` to **`30.675847568497` (`dL = -1.136e-3`)**.
   - On `911 -> 852`, peeling the `BL` `1x1` L-border (`remT=0`, `prot=2`) + clamping + masked melt + `joint_xytheta_slp` reached `29.703437` (`dL = +4.31e-3`), and chaining `mechwalk_compress` + `lpslack` compressed the basin by `-7.852e-3` to **`29.695584615062` (`dL = -3.543e-3`)**.
2. **Negative Gate Pairs (`docs/gate_results.json`, `docs/gate.log`)**:
   - `909 <- 970`: `970` had `f` slack of only `-1.14e-3` and `xcross` seeds (`rt3`) filled the `top_polish=8` queue ahead of `BL`/`TR` peel seeds, yielding `0` polished feasible seeds (`L_inc` unchanged at `30.638829`).
   - `912 <- 973`: `clamp` seeds reached `30.743179` (`dL = +3.99e-2`, above the `L_inc + 0.025` `mechwalk` trigger), while `reloc` seeds were ranked behind `clamp` by penetration.
   - `850 <- 970 + 909`: Best polished seed (`peel1_909_TR_reloc_v2_u0`) reached `29.692670` (`dL = +5.01e-2`).

## Files & Verification
- `poc/homocasc2_lib.py`: fast float-presorted `origin/main` + `lpslack` portfolio loader, `literal` + `pandas` + `kaggle_roundtrip` dual certifier, atomic win saver, `joint_xytheta_slp` + `lpslack` + `mechwalk_compress` pipeline.
- `poc/cascade_engine2.py`: 4-corner L-border extension, multi-cut L-border peel + interior-void relocation + seam-cell surgery, 0.5-step interior cross excision scanner, C++ masked melt + polish.
- `poc/run_gate.py`, `poc/run_scale.py`: gate and scale-up runners.
- `tests/test_homocasc2.py`: unit and `final.csv` dual-certification tests.
