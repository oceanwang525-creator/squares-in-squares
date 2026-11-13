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

**Idea 2 Wave 1 Results (`docs/idea2_results_wave1q.json`, quick, Negative — 0 wins):**
- `n=366`: 35 seeds; C++ feas `5/35`. Best certified non-45 result `s405_01_01_s1` -> `19.6907461` (`+0.0339` vs inc). `inc_melt` slid back into the incumbent basin (`19.6568628`, `+8.5e-6`).
- `n=448`: C++ feas `2/35`. Best non-45 result `s491_10_10_s1` -> `21.7174210` (`+0.0250`). `inc_melt_7` -> `21.6923911` (`+3.0e-6`, still the 45° basin).
- `n=581`: C++ feas `4/35`. Best non-45 result `s630_01_10_s1` -> `24.6383598` (`+0.0318`). `inc_melt_1` -> `24.6066020` (`+2.3e-7`, the 45° basin).
- **Diagnosis:** wall peels of the `40.97°`/`24.8°` donors still leave `oh_rem` between `0.10` and `0.30`, and the overhanging squares cannot be absorbed: the melt only reaches feasibility at about `L_inc + 0.03`. Tilted-MILP seeds are worse (`+0.25`). Incumbent melts always fall back into the `45°` basin.

**Idea 2 Wave 2 Results (`wave2q` on `850, 853, 585, 847`, quick, Negative — 0 wins; interrupted mid-`847` by server restart):**
- `n=850`: 33 seeds; C++ feas `2/33`. Best non-45 `s909_01_10_s0` -> `29.6862056` (`+0.0294` vs inc). `inc_melt_2` -> `29.6568976` (`+4.3e-5`).
- `n=853`: 33 seeds; C++ feas `0/33` (`min_pen=5.348e-4`). Best non-45 `s912_10_01_s1` -> `29.7354907` (`+0.0284`). `inc_melt_1` -> `29.7071178` (`+1.1e-5`).
- `n=585`: 35 seeds; C++ feas `3/35`. Best non-45 `s684_11_20_s1` -> `24.7553109` (`+0.0482`). `inc_melt_3` -> `24.7071149` (`+8.1e-6`).
- **Overhang audit insight:** Inspecting `405->366`, `630->581`, `491->448`, `909->850` directly showed why `surplus_peel` regressed: `630->581` at `L_box=24.591378` has **580/581** squares feasible (`0` overlap, `0` overhang) and only **1** overhanging square (`idx=139, 40.97°, oh=0.1874`); `405->366` at `19.633983` has only **3** overhanging squares (`two 45° @ 0.0972, one 0.94° @ 0.0096`). Unmasked global `a=0.78` melt shattered the feasible interior crystal.

---

## Idea 3: Multi-Margin Direct `joint_xytheta_slp` Harvest (`#90`) + Frozen-Interior Masked Wall/Seam Surgery (`< 4 min` Batches)

**Hypothesis:**
1. **Part A (`< 3 min` direct SLP harvest per `#90`):** Run multi-margin `[1e-7, 2e-8, 5e-9]` `joint_xytheta_slp` (with HiGHS `time_limit=10.0`) + 4 micro-jitter starts (`1e-4` pos, `0.02°` angle) directly on `merged.csv[n]` across all 12 targets (`850, 581, 974, 847, 534, 366, 585, 853, 448, 330, 262, 147`) to harvest any tight incumbent slack immediately.
**Idea 3A Harvest Results (`docs/idea3a_harvest.json`, `691.5s`, 2 strict wins):**
- `n=147`: **WIN** `L_lit=L_pd=12.65685424949241500000000000000000000000000000000000` vs `inc=12.65685424949241540000000000000000000000000000000000` (`dL=-4.000e-16`, saved to `data/win_n147.csv` + `final.csv`).
- `n=850`: **WIN** `L_lit=L_pd=29.65685424949260031415926535897938666094826185440144` vs `inc=29.65685424949260031923132169163970048778021020408988` (`dL=-5.072e-18`, saved to `data/win_n850.csv` + `final.csv`).
- Other 10 targets (`262, 330, 366, 448, 534, 581, 585, 847, 853, 974`): matched `inc_L` (tight 45° basins cannot break via local `joint_xytheta_slp` alone; next testing Idea 3B frozen-interior masked wall/void/seam surgery for real non-45° band replacements).

2. **Part B (Frozen-Interior Masked Wall/Void/Seam Surgery):** Preserve the feasible non-`45°` crystal in `surplus_peel` (`s630_01_10 -> 581`, `s405_01_01 -> 366`, `s491_10_10 -> 448`, `s909_01_10 -> 850`, `s912_01_01 -> 853`, `s685_11_02 -> 585`, `s906_01_01 -> 847`, `s630_02_11 -> 534`, `s199_11_02 -> 147`) and `cascades`-style interior `dk` cross-cuts by:
   - Setting C++ `free_mask` to ONLY squares within radius `R = 2.5..6.0` of overhanging squares / peeled seams / interior voids (holding the rest of the `L_box < L_inc` crystal rigid during initial soft `a`-melt!), OR dropping the `1..3` overhanging squares and re-inserting them into max-clearance interior voids before local `free_mask` melt + unmasked final `joint_xytheta_slp`!
