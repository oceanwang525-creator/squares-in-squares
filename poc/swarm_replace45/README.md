# Swarm `replace45`: Replacing Legacy 45° Diagonal-Band Incumbents

**Goal:** Drive the certified portfolio score toward `<= 22.0999` by replacing legacy `45°` diagonal-band incumbents (`f = L - k in {0.606602, 0.656854, 0.692388, 0.707107}`) with non-45° skewed bands (`40.97°`, `32.6°–35.1°`, `24.8°`) and sheared-row L-pockets on our 12 disjoint targets:

- Initial 3 targets: `850 (29, 9)`, `581 (24, 5)`, `974 (31, 13)`
- Extended pool: `847 (29, 6)`, `534 (23, 5)`, `366 (19, 5)`, `585 (24, 9)`, `853 (29, 12)`, `448 (21, 7)`, `330 (18, 6)`, `262 (16, 6)`, `147 (12, 3)`

Rules:
- CPU-only, max 4 workers/threads (`OMP/OPENBLAS/MKL/NUMEXPR=1`, HiGHS `threads=1`).
- Never modify `merged.csv` or any other `poc/` folder.
- Every idea is logged here before execution; every run (win or failure) is recorded and committed after `pytest -q poc/baseline/poc/tests/`.

---

## Idea 1: Donor-Adapted Non-45° Contraction/Expansion + C++ L-BFGS `a`-Melt + Joint `(x, y, dθ)` HiGHS SLP

**Hypothesis:**
All 12 targets are trapped in legacy `45°` diagonal-band basins (`f >= 0.606602`), while nearby `(k, D)` entries in `merged.csv` already achieve strictly lower `f` via non-45° bands and curved pockets:
1. **Family A (`581 (24,5)`, `847 (29,6)`, `534 (23,5)`, `f_front ~ 0.593–0.596`)**:
   - Adapt `merged[630] (25,5)` / `merged[681] (26,5)` / `merged[846] (29,5)` / `merged[906] (30,6)` (`40.97°` / `39.65°` multi-gadget skewed bands, `f = 0.5838–0.5914`).
   - Notice `n=630` decomposes into two disjoint tilted islands (`33` squares `+2` in top-left `[0..10.78] x [17.04..25.55]`, `56` squares `+3` in bottom-right `[12.04..24.55] x [0..16.78]`) separated in `x` by `1.26`. Removing one upright row (`y`-strip) + one upright column (`x`-strip) or deleting 1 tilted tooth along the band, followed by `a`-melt L-BFGS (`a: 0.95 -> 1.00`) and joint `(x, y, dθ)` HiGHS SLP, can shrink `630 -> 581` at `f ~ 0.591–0.594` (vs incumbent `0.606602`).
   - Also test expanding `533 (23,4) -> 534 (23,5)` / `580 (24,4) -> 581 (24,5)` / `846 (29,5) -> 847 (29,6)` by inserting a `+1` tilted pocket (`14`-square `36.87°` or `16`-square `36.87°` gadget) or 1 tooth on the tilted band, plus direct `40.97°` tilted-gadget tiling (`+2 + +2 + +1` for `D=5`; `+2 + +2 + +2` for `D=6`).
2. **Family B (`974 (31,13)`, `853 (29,12)`, `585 (24,9)`, `448 (21,7)`, `330 (18,6)`, `262 (16,6)`, `f_front ~ 0.681–0.692`)**:
   - Adapt `merged[973] (31,12)` / `merged[972] (31,11)` -> `974 (31,13)` by `+1`/`+2` pocket-tip/corner insertion or upright-seam replacement + `a`-melt L-BFGS + joint `(x, y, dθ)` SLP.
3. **Family C (`850 (29,9)`, `366 (19,5)`, `147 (12,3)`, `f_front ~ 0.636–0.651`)**:
   - Adapt `merged[849] (29,8)` / `merged[851] (29,10)` / `merged[909] (30,9)` / `merged[970] (31,9)` -> `850 (29,9)` via `+1`/`-1` pocket-tip surgery or `k`-row/col contraction + `a`-melt + joint `(x, y, dθ)` SLP.

**Initial Scope (< 10 min):** Start with the 3 priority targets `{581, 850, 974}` using a compiled C++ L-BFGS `a`-melt engine (`<= 4` threads) + Python `joint_xytheta_slp` (HiGHS `threads=1`), certifying every candidate strictly via `squares.core` literal + `pandas` roundtrip.

**Idea 1 Results (`docs/idea1_results.json`, Negative — 0 wins):**
- `n=581` (`inc=24.606601718`): `4/32` feasible (`d582_tminy` -> `24.6339825`, `c630_m3` -> `24.6436748`; near-feasible `c630_br` `L=24.604870` at `pen=9.85e-3`, `e580_v5` `L=24.605634` at `pen=1.64e-2`) in `36.53s`.
- `n=850` (`inc=29.656854249`): `6/28` feasible (`d851_tminy` -> `29.6841436`, `c909_tr` -> `29.688174, pen=3.00e-10`) in `238.79s`.
- `n=974` (`inc=31.707106781`): `6/25` feasible (`d975_tminy` -> `31.7215400`, `e973_v1` -> `31.7612157`) in `237.30s`.
- **Diagnosis:** (1) Euclidean nearest-neighbor strip contraction (`_contract_donor_k`) broke upright/tilted phase alignment instead of clean wall/cross-slice `surplus_peel` (`nr in [n, n+6]`); (2) `a_sched` starting at `0.92` was too stiff to melt jammed `45°` or `24.5°` pockets (`0.78 -> 1.0` needed); (3) `n+1 -> n` single deletions cannot escape `L_donor > L_inc`, whereas `inc_melt` (wall-slack drift + local pocket melts) and `surplus_peel` (`max_oh_rem` drop) start directly below `L_inc`.

---

## Idea 2: `surplus_peel` + `inc_melt` (5-Mode Wall-Slack & Pocket Unlock) + Upright-Phase MILP + Soft `a`-Melt (`0.78 -> 1.00`) + Joint `(x, y, dθ)` SLP (`< 5 min` Fast-Fail Batches)

**Hypothesis:**
Adapting `angles`'s winning recipe (`7b385ca` / `#80`) combined with phase-exact tilted-component + upright-lattice packing should break `45°` incumbents across our disjoint targets:
1. **`surplus_peel` (`nr in [n, n+6]`)**: Wall-split peels (`c=1,2,3` along `L, R, B, T`) and `5x5` interior `c=1` cross-slices on strictly higher-`(n2)` non-`45°` donors (`630/681/846 -> 581`, `906/967 -> 847`, `580/630 -> 534`, `909/970 -> 850`, `491/490 -> 448`, `405/446 -> 366`, `685/738/793 -> 585`, `912/973 -> 853`, `490/535 -> 330`, `295 -> 262`, `228/259/172 -> 147`), keeping peels ranked by `max_oh_rem` (worst overhang after dropping `nr - n` squares).
2. **Phase-Exact Tilted-Component + Upright-Lattice MILP Seeds**: Extract connected tilted (`non-0°`) gadgets from clean non-`45°` donors (e.g. `630`, `906`, `491`, `405`, `738`, `909`, `228`), shift their bounding box onto `L < L_inc`, and solve a HiGHS upright-lattice / mixed-border MILP + local void-fill to preserve exact non-`45°` pocket geometry without tearing tilted teeth.
3. **`inc_melt` (5-mode bottleneck/wall-slack unlock + soft `a: 0.78 -> 1.00`)**: On our `45°` incumbents themselves, run wall-pin void teleport, local `R=4.2..6.5` boundary-pin pocket melts, tilted-domain shear unlock, void-collapse inward pull, and asymmetric wall-slack drift (shifting loaded upright half toward its weakest wall) under soft `a`-melt (`0.78 -> 1.00`) targeting `L` slightly below `L_inc`.
4. **Joint `(x, y, dθ)` HiGHS SLP + Multi-Margin LP + Exact Radial Dilation Repair**: Polish top diverse candidates across all families, suppressing HiGHS `"threads": 1` warning (`OMP_NUM_THREADS=1` enforced) and checking exact literal + `pandas` roundtrip (`deg -> rad`) before saving any win.

**Scope (< 5 min per batch):** Fast-fail batches across `{581, 850, 974, 448, 366, 847, 534, 585, 853, 330, 262, 147}` (max 4 workers).
