# H22 / H31 / H39: Iterated Sparse Contact-Jacobian SVD Soft-Mode Mechanism Walk + C++ Masked Melt

## Method

- **Exact Sparse Contact Jacobian (`mechwalk_lib.py`)**: Vectorized near-active square-square separating axes and 4-wall contacts across multiple slack thresholds (`contact_tol in {2e-5, 5e-5, 2e-4, 4e-4, 1e-3, 2e-3}`), with exact analytical rotation/owner derivatives verified against finite differences (`tests/test_mechwalk.py`).
- **Diverse Soft-Mode Extraction & `L_inf` Stepping**: Extracts `k=14..16` smallest right singular modes (`scipy.sparse.linalg.svds`), projects out whole-layout rigid translation and rotation, deduplicates by cosine similarity (`< 0.90`), keeps both softest (`min sigma`) and collective (`max n_eff`) modes, and walks `(+1, -1)` across `L_inf`-normalized steps `(0.008, 0.025, 0.075, 0.18)`.
- **Dual Repair + Iterated Beam (4 rounds, beam=2, top-k=8)**:
  - **`rejam`**: Full-box soft-core SAT L-BFGS re-jam (`run_cpp_masked_melt` with all-1s mask, no ILS) + 2-stage `joint_xytheta_slp` + `fixed_angle_lp_tighten`.
  - **`mmelt` (Hybrid)**: Mode-activity pocket mask (top moving squares + spatial neighbors within `R=2.4`, capped at `<= 50%` of `n`) -> `run_cpp_masked_melt` (`ils=[0, 5, 10]`, `a_start=0.82`) -> 2-stage `joint_xytheta_slp` + `fixed_angle_lp_tighten`.
- **Certification**: Every exported group in `final.csv` is certified in BOTH `literal` and `pandas` roundtrip modes via `poc/auto_union/poc/union.py` against the full `origin/main` portfolio (`merged.csv` + all `poc/*/final.csv`), keeping only strict REAL WINs (`dL <= -1e-9`).

## Batch 3 Summary (H39: 6 Certified REAL WINs in `final.csv`, total `ds = -6.588023e-04`)

| `n` | `dL` | `ds = dL / sqrt(n)` | Certified `L` (`literal == pandas`) | Incumbent `L` (`origin/main`) | Method Label |
|---:|---:|---:|---|---|---|
| 800 | `-7.899371e-04` | `-2.792849e-05` | `28.7745792768193149028` | `28.7753692138915145108` (merged.csv) | `mmelt_r2_b0_m2_t2e-04_p_0.008` |
| 851 | `-1.810607e-02` | `-6.206682e-04` | `29.6660375325127864711` | `29.6841436063593355262` (merged.csv) | `mmelt_r4_b0_m1_t5e-05_p_0.008` |
| 909 | `-3.649977e-06` | `-1.210621e-07` | `30.6425760212162027437` | `30.6425796711927575211` (merged.csv) | `rejam_r1_b0_m8_t2e-04_m_0.008` |
| 910 | `-1.954595e-06` | `-6.479418e-08` | `30.6570963197696587104` | `30.6570982743642207525` (merged.csv) | `rejam_r2_b0_m0_t1e-03_m_0.008` |
| 911 | `-3.726164e-06` | `-1.234533e-07` | `30.6841505804990773282` | `30.6841543066630585767` (merged.csv) | `mmelt_r3_b1_m4_t5e-05_m_0.008` |
| 971 | `-3.083753e-04` | `-9.896233e-06` | `31.6565261466309459837` | `31.6568345218948805623` (merged.csv) | `rejam_r4_b1_m3_t4e-04_m_0.008` |

## Batch 3 Run Log (11 Evaluated `n`s: 6 REAL WINs, 5 Ties/Misses)

| `n` | Run | Status | Best `dL` (vs seed) | Time (s) | Best Label |
|---:|---|---|---:|---:|---|
| 538 | `b3_s1_results.json` | **MISS** | `+9.107856e-07` | 149.16 | `rejam_r4_b1_m5_t4e-04_p_0.008` |
| 740 | `b3_s1_results.json` | **MISS** | `+5.473967e-06` | 271.69 | `rejam_r3_b1_m1_t4e-04_p_0.008` |
| 796 | `b3_s2_results.json` | **MISS** | `+3.444213e-07` | 345.23 | `rejam_r1_b0_m8_t2e-05_p_0.008` |
| 800 | `b3_s2_results.json` | **REAL_WIN** | `-3.011079e-05` | 326.42 | `mmelt_r2_b0_m2_t2e-04_p_0.008` |
| 850 | `b3_s2_results.json` | **MISS** | `+1.914437e-06` | 399.79 | `mmelt_r1_b0_m6_t1e-03_p_0.008` |
| 851 | `b3_s2_results.json` | **REAL_WIN** | `-9.378385e-03` | 412.66 | `mmelt_r4_b0_m1_t5e-05_p_0.008` |
| 909 | `b3_s2_results.json` | **REAL_WIN** | `-3.649977e-06` | 466.69 | `rejam_r1_b0_m8_t2e-04_m_0.008` |
| 910 | `b3_s1_results.json` | **REAL_WIN** | `-1.954595e-06` | 622.59 | `rejam_r2_b0_m0_t1e-03_m_0.008` |
| 911 | `b3_s1_results.json` | **REAL_WIN** | `-3.726164e-06` | 815.89 | `mmelt_r3_b1_m4_t5e-05_m_0.008` |
| 914 | `b3_s1_results.json` | **MISS** | `+3.206700e-07` | 523.65 | `rejam_r2_b0_m2_t2e-04_p_0.008` |
| 971 | `b3_s1_results.json` | **REAL_WIN** | `-3.083753e-04` | 766.98 | `rejam_r4_b1_m3_t4e-04_m_0.008` |

## Batch 2 Archive (Merged in PR #35 / `34fd019`: 25 REAL WINs, 6 Ties/Misses in post-restart shard logs + pre-restart `372, 798`)

| `n` | Run | Status | Best `dL` | Time (s) | Best Label |
|---:|---|---|---:|---:|---|
| 379 | `scale_b2_s1_results.json` | **REAL_WIN** | `-1.180276e-06` | 82.17 | `rejam_r4_b1_m3_t4e-04_p_0.008` |
| 413 | `scale_b2_s1_results.json` | **REAL_WIN** | `-1.401424e-06` | 98.02 | `rejam_r2_b0_m4_t2e-05_p_0.008` |
| 416 | `scale_b2_s1_results.json` | **REAL_WIN** | `-1.298144e-05` | 81.93 | `rejam_r4_b0_m3_t4e-04_p_0.008` |
| 419 | `scale_b2_s1_results.json` | **REAL_WIN** | `-7.819445e-06` | 83.63 | `mmelt_r4_b0_m4_t5e-05_p_0.008` |
| 421 | `scale_b2_s1_results.json` | **REAL_WIN** | `-1.331479e-05` | 77.11 | `mmelt_r4_b0_m5_t2e-03_p_0.008` |
| 455 | `scale_b2_s1_results.json` | **MISS** | `+1.472518e-06` | 130.37 | `mmelt_r3_b1_m1_t4e-04_p_0.008` |
| 464 | `scale_b2_s1_results.json` | **REAL_WIN** | `-6.657875e-06` | 88.89 | `mmelt_r4_b1_m0_t4e-04_p_0.008` |
| 507 | `scale_b2_s1_results.json` | **REAL_WIN** | `-3.897566e-06` | 168.28 | `mmelt_r2_b1_m2_t2e-05_m_0.008` |
| 540 | `scale_b2_s1_results.json` | **MISS** | `+4.312613e-07` | 143.4 | `rejam_r2_b1_m0_t2e-05_p_0.008` |
| 546 | `scale_b2_s1_results.json` | **REAL_WIN** | `-3.090315e-05` | 116.58 | `mmelt_r2_b0_m1_t2e-05_p_0.008` |
| 585 | `scale_b2_s2_results.json` | **REAL_WIN** | `-1.704926e-05` | 140.58 | `mmelt_r4_b1_m0_t4e-04_p_0.008` |
| 587 | `scale_b2_s2_results.json` | **REAL_WIN** | `-4.499676e-06` | 149.48 | `rejam_r4_b1_m3_t4e-04_p_0.008` |
| 640 | `scale_b2_s2_results.json` | **MISS** | `+6.629110e-06` | 255.19 | `rejam_r1_b0_m7_t2e-05_p_0.008` |
| 642 | `scale_b2_s2_results.json` | **REAL_WIN** | `-1.710419e-05` | 226.35 | `mmelt_r3_b1_m4_t2e-03_m_0.008` |
| 646 | `scale_b2_s1_results.json` | **REAL_WIN** | `-2.281228e-05` | 213.01 | `rejam_r4_b1_m3_t5e-05_m_0.008` |
| 687 | `scale_b2_s2_results.json` | **REAL_WIN** | `-5.031107e-06` | 303.66 | `mmelt_r2_b0_m2_t2e-05_m_0.008` |
| 694 | `scale_b2_s2_results.json` | **MISS** | `+4.083411e-07` | 188.36 | `rejam_r1_b0_m0_t2e-04_p_0.008` |
| 698 | `scale_b2_s2_results.json` | **REAL_WIN** | `-1.317480e-06` | 182.51 | `rejam_r2_b1_m2_t1e-03_p_0.008` |
| 700 | `scale_b2_s2_results.json` | **REAL_WIN** | `-5.099505e-06` | 197.25 | `mmelt_r4_b1_m3_t5e-05_p_0.008` |
| 703 | `scale_b2_s2_results.json` | **REAL_WIN** | `-1.033235e-07` | 249.57 | `rejam_r2_b0_m1_t2e-04_p_0.008` |
| 736 | `scale_b2_s1_results.json` | **REAL_WIN** | `-2.708571e-06` | 297.38 | `rejam_r2_b1_m1_t2e-04_p_0.008` |
| 789 | `scale_b2_s2_results.json` | **REAL_WIN** | `-1.422289e-06` | 382.5 | `rejam_r1_b0_m2_t2e-04_p_0.008` |
| 797 | `scale_b2_s2_results.json` | **REAL_WIN** | `-6.311970e-04` | 387.64 | `mmelt_r2_b0_m1_t2e-04_p_0.008` |
| 811 | `scale_b2_s2_results.json` | **REAL_WIN** | `-3.051682e-06` | 596.01 | `mmelt_r4_b0_m0_t4e-04_p_0.008` |
| 853 | `scale_b2_s2_results.json` | **REAL_WIN** | `-3.099264e-06` | 369.53 | `rejam_r4_b0_m5_t5e-05_p_0.008` |
| 866 | `scale_b2_s2_results.json` | **MISS** | `+1.081777e-07` | 240.43 | `mmelt_r2_b0_m5_t1e-03_p_0.008` |
| 875 | `scale_b2_s1_results.json` | **REAL_WIN** | `-5.736497e-06` | 338.08 | `rejam_r2_b0_m3_t2e-05_p_0.008` |
| 904 | `scale_b2_s2_results.json` | **MISS** | `+8.240138e-06` | 308.62 | `mmelt_r3_b0_m4_t2e-03_p_0.008` |
| 922 | `scale_b2_s2_results.json` | **REAL_WIN** | `-3.060452e-05` | 545.43 | `mmelt_r3_b0_m1_t4e-04_p_0.008` |
| 936 | `scale_b2_s2_results.json` | **REAL_WIN** | `-3.918276e-04` | 353.0 | `rejam_r4_b0_m3_t5e-05_m_0.008` |
| 988 | `scale_b2_s1_results.json` | **REAL_WIN** | `-5.159286e-06` | 643.48 | `rejam_r2_b0_m4_t2e-04_p_0.008` |

## Batch 1 Archive (Merged in PR #32 / `e5751c5`: 18 REAL WINs, 10 Ties/Misses across 28 `n`s)

| `n` | Run | Status | Best `dL` | Time (s) | Best Label |
|---:|---|---|---:|---:|---|
| 459 | `scale_s1_results.json` | **REAL_WIN** | `-4.203411e-05` | 98.66 | `rejam_r4_b1_m1_t2e-03_p_0.008` |
| 463 | `gate_results.json` | **REAL_WIN** | `-1.423650e-05` | 77.71 | `mmelt_r2_b1_m0_t2e-05_p_0.025` |
| 499 | `scale_s2_results.json` | **REAL_WIN** | `-3.042003e-05` | 103.92 | `rejam_r1_b0_m0_t1e-03_p_0.008` |
| 542 | `scale_s1_results.json` | **REAL_WIN** | `-1.245804e-04` | 125.35 | `mmelt_r4_b1_m5_t5e-05_p_0.008` |
| 598 | `scale_s1_results.json` | **REAL_WIN** | `-5.807060e-06` | 153.76 | `mmelt_r2_b0_m4_t2e-05_p_0.008` |
| 634 | `scale_s1_results.json` | **MISS** | `+1.338857e-06` | 168.74 | `mmelt_r1_b0_m2_t2e-04_p_0.008` |
| 639 | `scale_s1_results.json` | **REAL_WIN** | `-1.609637e-03` | 152.18 | `rejam_r3_b0_m4_t2e-03_p_0.008` |
| 686 | `scale_s2_results.json` | **REAL_WIN** | `-4.925768e-03` | 273.9 | `mmelt_r3_b0_m5_t2e-03_p_0.025` |
| 691 | `scale_s1_results.json` | **MISS** | `+3.706832e-06` | 164.71 | `rejam_r1_b0_m0_t2e-04_p_0.008` |
| 692 | `gate_results.json` | **MISS** | `+3.245739e-06` | 225.13 | `mmelt_r1_b0_m6_t2e-04_p_0.025` |
| 695 | `scale_s1_results.json` | **REAL_WIN** | `-9.573239e-06` | 185.51 | `rejam_r2_b1_m1_t1e-03_p_0.008` |
| 701 | `gate_results.json` | **MISS** | `+2.826322e-06` | 150.56 | `mmelt_r4_b1_m0_t5e-05_m_0.025` |
| 745 | `scale_s1_results.json` | **MISS** | `+8.026616e-07` | 185.65 | `mmelt_r4_b0_m1_t2e-03_p_0.008` |
| 793 | `scale_s2_results.json` | **REAL_WIN** | `-3.459514e-05` | 321.05 | `mmelt_r4_b0_m4_t5e-05_m_0.008` |
| 804 | `scale_s2_results.json` | **REAL_WIN** | `-4.233042e-05` | 394.9 | `mmelt_r4_b0_m5_t4e-04_m_0.008` |
| 850 | `scale_s2_results.json` | **REAL_WIN** | `-9.313785e-05` | 470.51 | `mmelt_r1_b0_m7_t2e-04_m_0.008` |
| 851 | `scale_s2_results.json` | **MISS** | `+1.492527e-05` | 667.26 | `rejam_r4_b1_m3_t2e-03_p_0.008` |
| 852 | `scale_s2_results.json` | **REAL_WIN** | `-6.187867e-07` | 499.45 | `mmelt_r2_b0_m2_t2e-04_m_0.008` |
| 858 | `scale_s1_results.json` | **MISS** | `+9.668365e-07` | 328.83 | `rejam_r1_b0_m2_t2e-04_p_0.008` |
| 861 | `scale_s2_results.json` | **REAL_WIN** | `-3.009402e-05` | 391.43 | `rejam_r1_b0_m0_t1e-03_p_0.008` |
| 909 | `scale_s2_results.json` | **REAL_WIN** | `-2.290068e-03` | 505.08 | `rejam_r4_b0_m3_t5e-05_p_0.008` |
| 912 | `scale_s2_results.json` | **MISS** | `+6.539067e-07` | 458.69 | `mmelt_r1_b0_m3_t1e-03_p_0.008` |
| 915 | `scale_s1_results.json` | **REAL_WIN** | `-7.741169e-05` | 545.96 | `rejam_r3_b0_m2_t4e-04_p_0.008` |
| 918 | `scale_s1_results.json` | **MISS** | `+3.331219e-06` | 603.72 | `mmelt_r3_b1_m2_t5e-05_p_0.008` |
| 920 | `scale_s2_results.json` | **REAL_WIN** | `-5.739813e-06` | 577.96 | `rejam_r2_b1_m2_t2e-05_p_0.008` |
| 926 | `scale_s2_results.json` | **REAL_WIN** | `-4.400093e-04` | 559.92 | `rejam_r3_b0_m2_t4e-04_p_0.008` |
| 973 | `scale_s1_results.json` | **MISS** | `+6.634325e-06` | 634.41 | `mmelt_r1_b0_m4_t1e-03_p_0.008` |
| 978 | `scale_s1_results.json` | **REAL_WIN** | `-5.091701e-04` | 686.18 | `rejam_r4_b0_m2_t2e-03_p_0.008` |
