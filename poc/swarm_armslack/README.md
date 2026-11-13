# Hypothesis H12 (`poc/swarm_armslack`): End-Cap Strip Gap Release & Asymmetric-Slack Corridor Equalization

## Physical & Geometric Mechanism
1. **End-Cap Strip Gap Release**: In `BL = k^2 - 1` / `L`-corridor packings with an un-slid upright end-cap strip (`y0 = 0.5 + frac > 1.0` on the right wall or trapped gap in the top strip), sliding the first `m - 1` upright end-cap squares against `0` (`0.5, 1.5, ..., m - 1.5`) moves the `frac` gap directly to the `TR` junction, allowing the `m`-th end-cap square to step inward and the `TR` / corridor seam to compress.
2. **Asymmetric-Slack Corridor Equalization**: When `Arm A` (`dA`) or `Arm B` (`dB`) has slack relative to `L`, combining the tight-arm end-cap strip slide with Idea 6 masked-pocket cavity-relocation melts (`L0 = L_base + 0.032`, `a: 0.78 -> 1.00`, `cavity_relocate_orbit`) around the outer end cap and `BL` corner (`R = 5.0..7.0`) followed by `relax_lbfgs`, `joint_xytheta_slp`, and `fixed_angle_lp_tighten` equalizes corridor load and lowers `L`.

## Files
- `poc/strip_slide.py`: TR-boundary-aware end-cap strip detector (`detect_endcap_strips`), explicit gap-release slide (`slide_endcap_strip`), and H12 seed builder (`build_armslack_seeds`).
- `poc/run_gate.py`: Gate runner using `highd` C++ masked-pocket melt + `joint_xytheta_slp` + canonical `poc/auto_union/poc/union.py` `s`-prefixed literal + pandas certification.
- `poc/scale_sweep.py`: Checkpointed detached scaling runner (Batch 1/2) combining strip-junction and outer-endcap+BL masked pockets, keeping only strict real wins (`dL <= -1e-9`) in `final.csv` and stopping after 12 consecutive non-real-wins.
- `tests/test_armslack.py`: Unit tests for strip detection, gap slide, seed shapes, and `final.csv` schema.
- `docs/gate_results.json`, `docs/scale_summary.json`, `docs/scale_ckpt/`: Gate and scaling run metrics/checkpoints.
- `data/win_n*.csv`, `final.csv`: Certified real wins (`dL <= -1e-9`) in canonical `s`-prefixed format.

## 15-Minute Falsification Gate Results (vs `origin/main` `e7f910b` / `be03ed7`)
Command: `python -u poc/swarm_armslack/poc/run_gate.py --targets 971 915 975 914 --threads 4` (`459.1s`)

- **`n=915` (REAL WIN, `asym_ec_bl_1`)**:
  - `inc_L` (`e7f910b` / `be03ed7`): `30.74081842683129861077203630312931151188852936835497` (`r_y0=1.2408`, `dA=-0.001612`)
  - `L_lit = L_pd`: `30.74041634322884935263322475786982481920538123803722`
  - **`dL = -4.020836e-04`**, `ds = -4.394356e-07`, `non0/90 = 361/915`
- `n=971`: `MISS` (`rt_slide_1`, `dL = +2.080095e-06` vs `e7f910b` `31.658053882`)
- `n=975`: `MISS` (`r_slide_4`, `dL = +2.072246e-05` vs `e7f910b` `31.721540002`)
- `n=914`: `MISS` (`asym_ec_bl_0`, `dL = +5.668670e-05` vs `e7f910b` `30.731743789`)

## Scaling Batch 1 Results (`scale_sweep.py`, `1726.8s`, stopped on `12/12` consecutive non-real-wins)
24 targets evaluated (`686, 791, 911, 985, 751, 859, 639, 800, 690, 696, 498, 796, 592, 739, 802, 909, 975, 683, 740, 857, 854, 918, 919, 848`):
- **`n=911` (REAL WIN, `r_slide_1`)**: `L_lit = L_pd = 30.686956998826932959400...` vs `inc = 30.689430606622261708041...` (**`dL = -2.473608e-03`**, `non0/90 = 292/911`)
- **`n=800` (REAL WIN, `r_slide_2`)**: `L_lit = L_pd = 28.775374244599161505279...` vs `inc = 28.775399323092784727828...` (**`dL = -2.507849e-05`**, `non0/90 = 269/800`)
- **`n=796` (REAL WIN, `r_slide_5`)**: `L_lit = L_pd = 28.705910015470158537059...` vs `inc = 28.705918138939055037498...` (**`dL = -8.123469e-06`**, `non0/90 = 312/796`)
- Remaining 21 targets: `MISS` (0 ties).

## Scaling Batch 2 Results (`scale_sweep.py`, `1316.9s` solver / `1335.9s` wall, 31 targets, rotated/unslid `canonicalize_tr`)
31 targets evaluated (`908, 793, 971, 904, 694, 849, 792, 371, 542, 410, 684, 496, 268, 733, 905, 577, 530, 485, 442, 635, 789, 966, 580, 332, 336, 414, 327, 533, 745, 845, 205`):
- **`n=542` (REAL WIN, `t_slide_0`)**: `L_lit = L_pd = 23.779285924414851...` vs `inc = 23.780902668492588...` (**`dL = -1.616744e-03`**, `ds = -2.982923e-06`)
- **`n=530` (REAL WIN, `r_slide_1`)**: `L_lit = L_pd = 23.533043261085421...` vs `inc = 23.533704455613332...` (**`dL = -6.611945e-04`**, `ds = -1.247537e-06`)
- **`n=577` (REAL WIN, `r_slide_3`)**: `L_lit = L_pd = 24.533050505846838...` vs `inc = 24.533704455633332...` (**`dL = -6.539498e-04`**, `ds = -1.133362e-06`)
- **`n=442` (REAL WIN, `t_slide_2`)**: `L_lit = L_pd = 21.533079968339098...` vs `inc = 21.533704455573330...` (**`dL = -6.244872e-04`**, `ds = -1.412867e-06`)
- **`n=485` (REAL WIN, `r_slide_1`)**: `L_lit = L_pd = 22.533083571885759...` vs `inc = 22.533704455593330...` (**`dL = -6.208837e-04`**, `ds = -1.280173e-06`)
- **`n=789` (REAL WIN, `r_slide_4`)**: `L_lit = L_pd = 28.591332243216104...` vs `inc = 28.591344634320595...` (**`dL = -1.239110e-05`**, `ds = -1.570482e-08`)
- Remaining 25 targets (including the final 10 consecutive misses `966..205`): `MISS` (0 ties). Total certified REAL WINs in `poc/swarm_armslack/final.csv`: 10 (`442, 485, 530, 542, 577, 789, 796, 800, 911, 915`).

## Scaling Batch 3 Results (`scale_sweep.py`, `855.2s` solver / `868.1s` wall, stopped on `12/12` consecutive non-real-wins, `canonicalize_void_tr` `q_void`)
13 targets evaluated (`401, 851, 963, 747, 790, 964, 582, 730, 366, 735, 936, 677, 591`):
- **`n=401` (REAL WIN, `t_slide_1`, `q=3`)**: `L_lit = L_pd = 20.533530106866359...` vs `inc = 20.533704455357993...` (**`dL = -1.743485e-04`**; also beats `252fe8b:poc/guestdata_atlas/final.csv` `20.533704455357993...`)
- Remaining 12 targets (`851..591`): `MISS` (0 ties). Total certified REAL WINs in `poc/swarm_armslack/final.csv`: 11 (`401, 442, 485, 530, 542, 577, 789, 796, 800, 911, 915`).

*(Note: an initial gate run prior to fetching `e7f910b` also compressed `50badc8`'s `win_n971` from `31.662036554` to `31.661253448`, `dL = -7.831062e-04`, via `r_slide_2`, before `e7f910b` superseded `win_n971`.)*
