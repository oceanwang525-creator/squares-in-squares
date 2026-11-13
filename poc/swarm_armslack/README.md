# Hypothesis H12 (`poc/swarm_armslack`): End-Cap Strip Gap Release & Asymmetric-Slack Corridor Equalization

## Physical & Geometric Mechanism
1. **End-Cap Strip Gap Release**: In `BL = k^2 - 1` / `L`-corridor packings with an un-slid upright end-cap strip (`y0 = 0.5 + frac > 1.0` on the right wall or trapped gap in the top strip), sliding the first `m - 1` upright end-cap squares against `0` (`0.5, 1.5, ..., m - 1.5`) moves the `frac` gap directly to the `TR` junction, allowing the `m`-th end-cap square to step inward and the `TR` / corridor seam to compress.
2. **Asymmetric-Slack Corridor Equalization**: When `Arm A` (`dA`) or `Arm B` (`dB`) has slack relative to `L`, combining the tight-arm end-cap strip slide with Idea 6 masked-pocket cavity-relocation melts (`L0 = L_base + 0.032`, `a: 0.78 -> 1.00`, `cavity_relocate_orbit`) around the outer end cap and `BL` corner (`R = 5.0..7.0`) followed by `relax_lbfgs`, `joint_xytheta_slp`, and `fixed_angle_lp_tighten` equalizes corridor load and lowers `L`.

## Files
- `poc/strip_slide.py`: TR-boundary-aware end-cap strip detector (`detect_endcap_strips`), explicit gap-release slide (`slide_endcap_strip`), and H12 seed builder (`build_armslack_seeds`).
- `poc/run_gate.py`: Gate and checkpointed scaling runner using `highd` C++ masked-pocket melt + `joint_xytheta_slp` + canonical `poc/auto_union/poc/union.py` `s`-prefixed literal + pandas certification.
- `tests/test_armslack.py`: Unit tests for strip detection, gap slide, seed shapes, and `final.csv` schema.
- `docs/gate_results.json`: Gate run metrics and per-seed diagnostics.
- `data/win_n915.csv`, `final.csv`: Certified winning packing(s) in canonical `s`-prefixed format.

## 15-Minute Falsification Gate Results (vs `origin/main` `e7f910b`)
Command: `python -u poc/swarm_armslack/poc/run_gate.py --targets 971 915 975 914 --threads 4` (`459.1s`)

- **`n=915` (REAL WIN, `asym_ec_bl_1`)**:
  - `inc_L` (`e7f910b`): `30.74081842683129861077203630312931151188852936835497` (`r_y0=1.2408`, `dA=-0.001612`)
  - `L_lit = L_pd`: `30.74041634322884935263322475786982481920538123803722`
  - **`dL = -4.020836e-04`**, `ds = -4.394356e-07`, `non0/90 = 361/915`
- `n=971`: `MISS` (`rt_slide_1`, `dL = +2.080095e-06` vs `e7f910b` `31.658053882`)
- `n=975`: `MISS` (`r_slide_4`, `dL = +2.072246e-05` vs `e7f910b` `31.721540002`)
- `n=914`: `MISS` (`asym_ec_bl_0`, `dL = +5.668670e-05` vs `e7f910b` `30.731743789`)

*(Note: an initial gate run prior to fetching `e7f910b` also compressed `50badc8`'s `win_n971` from `31.662036554` to `31.661253448`, `dL = -7.831062e-04`, via `r_slide_2`, before `e7f910b` superseded `win_n971`.)*
