# Hypothesis H78: Constant-`r` Homology L-Border Cascade (`swarm_homocasc2`)

## Summary
- **Branch**: `swarm-homocasc2` / `swarm-homocasc2-pr2` (merged with `origin/main` @ `aea8dc0`)
- **PR #1 (`origin/swarm-homocasc2-pr1`, merged into `origin/main`)**: Landed `n=911` (`30.676984 -> 30.675847568497090971`, `dL = -1.136128e-3`) and initial `n=852` (`29.699128 -> 29.695584615062476056`, `dL = -3.543429e-3`).
- **PR #2 (`origin/swarm-homocasc2-pr2`, vs `origin/main` `aea8dc0`)**: **3 strict REAL WINs (`n=852, 853, 912`)**, **total `ds = -3.849438e-4` (`L/sqrt(n)`), `ds_L2n = -7.829781e-4` (`L^2/n`)**, `literal == pandas` dual-certified against the full `origin/main` (`aea8dc0`) portfolio (`n=911` omitted here as already merged into `origin/main`):

| `n` | `(m, r)` | `L_inc` (`origin/main` `aea8dc0`) | `L_win` (`literal == pandas`) | `dL` | `ds = dL / sqrt(n)` | Cascade Path |
|---:|:---:|---:|---:|---:|---:|---|
| 852 | `(29, +11)` | `29.695584606875173959` | `29.688527389492101859` | `-7.057217e-3` | `-2.417763e-4` | `911 (30.675848) -> 852` (`peel1_911_TR_k1.010_dropProt_clamp` + melt + `joint_xytheta_slp` + `mechwalk_compress` + `lpslack`) |
| 853 | `(29, +12)` | `29.705882380544231341` | `29.705882376792042049` | `-3.752189e-9` | `-1.284725e-10` | `796 (28.705882376) -> 853` (`extTR` 1-unit upright L-border with `1e-9` spacing + direct SLP + `lpslack`) |
| 912 | `(30, +12)` | `30.703259882007011852` | `30.698936322874302400` | `-4.323559e-3` | `-1.431673e-4` | `973 (31.679167) -> 912` (`peel1_973_BL_k1.010_clamp_rot45` + melt + `joint_xytheta_slp` + `mechwalk_compress` + `lpslack`) |

## Geometric Mechanism & Scale-Up Findings
1. **Why the H78 cascade broke through on `r = +11` and `r = +12`**:
   - **Gate (`972 -> 911 -> 852`)**: `xcross_972_x20.0_y21.0_rt1` + `mechwalk_compress` + `lpslack` cut `n=911` to `30.675847568497` (`-1.136e-3`), unlocking an extra `-20.5` milli-units of `(m-1, +11)` slack for `n=852`.
   - **Scale-Up Wave 1 (`n=852` further drop & `973 -> 912`)**:
     - Using the new `30.675848` `n=911` donor, peeling the `TR` L-border (`peel1_911_TR_k1.010_dropProt_clamp`) reached `29.698142`, and 4-round SVD `mechwalk_compress` + `lpslack` compressed the basin by `-9.615e-3` to **`29.688527389492`** (`-7.057e-3` below PR #1 / `aea8dc0`, `-1.060e-2` below pre-gate `8c623fc`).
     - On `973 -> 912`, `peel1_973_BL_k1.010_clamp_rot45` + melt + `joint_xytheta_slp` reached `30.710556` (`dL = +7.30e-3`), and 4-round `mechwalk_compress` + `lpslack` drove it `-1.162e-2` down to **`30.698936322874`** (`dL = -4.324e-3`).
   - **Upward Extension (`796 -> 853`)**: Adding a 1-unit upright L-border with `1e-9` clearance spacing (`extend_one_layer_all_corners`) around `n=796` (`28.705882375792`) eliminates exact-touching rejection in `auto_union.check`, yielding a strict win on `n=853` (`29.705882376792042049`, `dL = -3.752e-9`).
2. **Non-Winning Scale-Up Targets (`docs/scale_results.json`, `docs/scale2_results.json`)**:
   - Closest near-misses after `mechwalk_compress + lpslack`: `n=738<-793` (`+7.330e-3`), `n=795<-852` (`+7.980e-3`), `n=910<-971` (`+8.203e-3`), `n=688<-741` (`+1.131e-2`), `n=855<-914` (`+1.352e-2`), `n=854<-913` (`+1.412e-2`), `n=794<-851` (`+1.727e-2`).

## Files & Verification
- `poc/homocasc2_lib.py`: fast float-presorted `origin/main` portfolio loader, `literal` + `pandas` + `kaggle_roundtrip` dual certifier, atomic win saver, `joint_xytheta_slp` (with zero-pen LP/lpslack fallback) + `lpslack` + `mechwalk_compress` pipeline.
- `poc/cascade_engine2.py`: 4-corner L-border extension (`1e-9` gap spacing), multi-cut L-border peel + interior-void relocation + seam-cell surgery, 0.5-step interior cross excision scanner, C++ masked melt + polish.
- `poc/run_gate.py`, `poc/run_scale.py`: gate and `max_workers=2` scale-up runners.
- `tests/test_homocasc2.py`: unit and `final.csv` dual-certification tests.
