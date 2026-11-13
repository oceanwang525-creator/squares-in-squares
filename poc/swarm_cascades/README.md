# swarm_cascades Log

Branch: `swarm-cascades` (from `origin/main` at `a6fbef7`, official score `22.17371545003316`).
Target pool (disjoint):
- Initial (1-3 `n`, <10 min): `902, 912, 738`
- Extended pool: `843, 786, 904, 582, 263, 646, 593, 499, 695, 749, 863, 640`

Resource caps: CPU only, max 4 workers/threads (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, HiGHS `threads=1`).

## Baseline Audit (at `a6fbef7`)
- `n=912 (30,12)`: `inc_L = 30.703260237`, `L_frontier = 30.687497`, gap `+0.015763`. Uses two `24.3°/65.7°` pockets (`149 + 150` tilted) along top and right. Donor `n=973 (31,12)` has `L = 31.696609` (`f = 0.696609`, `-0.006651` below `912`); donor `n=972 (31,11)` has `L = 31.683990` (`f = 0.683990`); donor `n=852 (29,11)` has `L = 29.699596` (`f = 0.699596`).
- `n=738 (27,9)`: `inc_L = 27.678647743`, `L_frontier = 27.665492`, gap `+0.013155`. Uses one connected `24.8°` L-pocket (`270` tilted). Donor `n=971 (31,10)` has `L = 31.664318` (`f = 0.664318`, two separated `25.2°` pockets `136+136`); donor `n=793 (28,9)` has `L = 28.677243` (`f = 0.677243`).
- `n=902 (30,2)`: `inc_L = 30.535533906`, `L_frontier = 30.514079`, gap `+0.021455`. Legacy `45°` (`74` at `45°`, `1` at `43°`, `827` upright). Neighbor `n=901 (30,1)` is `30.520815` (`17x17 @ 45°` + `612` upright); `n=904 (30,4)` is `30.563136` (`33.56°` `w=3` band).

---

## Idea 1: Cross-`n` Interior Seam Surgery + C++ L-BFGS `a`-Melt + Joint `(x, y, dtheta)` SLP on `{912, 738, 902}`

**Hypothesis:**
1. **Interior slab/seam cuts with local melt (`912`, `738`)**: Earlier rigid `-5m` slab cuts (`#66`) failed because `24.8°` curved pockets have slight non-uniformity at the cut line, creating local penetration when shifted by `1.0`. However, cutting an interior `(x_cut, y_cut)` strip of width `~1.0` from `n=973 -> 912` (`L_0 ~ 30.6966 < 30.7033`), `n=972 -> 912` (removing `60` squares), `n=793 -> 738` (`L_0 ~ 27.6772 < 27.6786`), and `n=971 -> 738` (`4`-unit interior cut removing `233` squares, `L_0 ~ 27.6643`), followed by a **local seam-focused C++ L-BFGS `a`-melt** (`a: 0.90 -> 1.00` on squares near the cut seam, or full gentle `a: 0.95 -> 1.00`) + **`joint_xytheta_slp`** (exact sparse HiGHS SLP optimizing `dx, dy, dtheta, L` simultaneously) can heal the cut seam and lock in the lower `f` of the larger-`k` donor.
2. **Direct pocket-seam ruin-and-recreate (`912`, `738`)**: Perturb/melt the `20–45` squares around the pocket-corner junction and high-gap defects in `merged[912]` and `merged[738]`, then apply `C++ L-BFGS a-melt` + `joint_xytheta_slp`.
3. **`D=2` Twin-Pocket & `901+1`/`904-2` Surgery (`902`)**:
   - `901 -> 902`: Insert `+1` square into `merged[901]` (`30.520815`, `-0.0147` below `902`) by opening a local `30–50` square cavity near the `25.997, 12.564` tilted defect or upright corner, melting + `joint_xytheta_slp`.
   - `904 -> 902`: Delete `2` squares from `merged[904]` (`w=3` `33.56°` band) or peel 1 layer + melt + `joint_xytheta_slp`.
   - `C2` twin-pocket / skewed `w=2` band cold starts in C++ (`sym_mode=2`, `a_start=0.88 -> 1.00`) + `joint_xytheta_slp`.

**Plan:**
- Build `poc/swarm_cascades/poc/cascades_cpp.cpp` (compiled to `/tmp/libcascades.so`, CPU only, `threads <= 4`) supporting C++ spatial-hash SAT L-BFGS with `a`-melt schedules (`sym_mode=1` or `2`, optional `free_mask` seam-only or full-box) + `poc/swarm_cascades/poc/slp_utils.py` (reusing exact `joint_xytheta_slp` + official `squares.core` certification + `final.csv` writer).
- Write unit tests in `poc/swarm_cascades/poc/tests/test_cascades.py`.
- Run `idea1_run.py` on `{912, 738, 902}` (< 8 minutes, max 4 workers), record exact results in `poc/swarm_cascades/docs/idea1_results.json` and here.

**Results (2026-11-13, `362.1s`, 4 threads, 0 wins -> no `final.csv`):**
- `n=912` (`inc_L = 30.703260237`, `L_frontier = 30.687497`):
  - `912_direct`: `ok=True`, `L_slp = 30.70326023727158` (exact incumbent).
  - `912_junc_1` / `912_junc_2` / `912_junc_4`: healed to `30.703286`, `30.703286`, `30.709301` (`+0.000026..+0.006`).
  - `973_to_912_cut1`: `pen_cpp = 2.10e-3` at `L_cpp = 30.697156`, SLP resolved overlap at `L_slp = 30.720790` (`+0.0175`).
  - `972_to_912_cut0` / `973_to_912_cut0,4,5`: failed SLP after `pen_cpp = 7.65e-3..1.65e-2`.
  - Root causes identified: (1) interior cuts at `cx, cy in [11..16]` sliced directly through `973`'s `BL` `45°` bend (`x,y < 16`) instead of the pure upright `TR` quadrant (`x,y > 16`); (2) `912_junc_0,3,5` had `pen_cpp <= 7.4e-12` in C++ SAT yet failed `verify_official` because `joint_xytheta_slp` accepted `pen <= 1e-9` without strict `1e-7` final LP clearance.
- `n=738` (`inc_L = 27.678647743`, `L_frontier = 27.665492`):
  - `738_direct`: `ok=True`, `L_slp = 27.67864774285407` (incumbent).
  - `738_defect_3,4,2,1`: `ok=True`, `L_slp = 27.678677..27.685803`.
  - `971_to_738_cut3`: healed overlap at `L_slp = 27.921312` (`+0.243`); `971_to_738_cut0,1` and `737_plus1_*` had `pen_cpp = 1.28e-2..2.04e-2` (`ok=False`). Same cause: `971` cuts sliced the `BL` bend instead of upright `TR` quadrant (`cx, cy ~ 26.9`).
- `n=902` (`inc_L = 30.535533906`, `L_frontier = 30.514079`):
  - `904_minus2_*`: `ok=True`, `L_slp = 30.564828..30.565928` (`33.6°` band spans full `30.56` width).
  - `901_plus1_*`: `pen_cpp = 1.34e-2..2.21e-2`, failed SLP.
  - `902_c2_*` (`sym=2`): best feasible `902_c2_w6_m10_33.0` (`L_slp = 30.738180`, `+0.203`).
