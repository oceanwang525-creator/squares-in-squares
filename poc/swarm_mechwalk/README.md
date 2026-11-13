# H22: Iterated Sparse Contact-Jacobian SVD Soft-Mode Mechanism Walk + C++ Masked Melt

## Method

- **Exact Sparse Contact Jacobian (`mechwalk_lib.py`)**: Vectorized near-active square-square separating axes and 4-wall contacts across multiple slack thresholds (`contact_tol in {2e-5, 5e-5, 2e-4, 4e-4, 1e-3, 2e-3}`), with exact analytical rotation/owner derivatives verified against finite differences (`tests/test_mechwalk.py`).
- **Diverse Soft-Mode Extraction & `L_inf` Stepping**: Extracts `k=14..16` smallest right singular modes (`scipy.sparse.linalg.svds`), projects out whole-layout rigid translation and rotation, deduplicates by cosine similarity (`< 0.90`), keeps both softest (`min sigma`) and collective (`max n_eff`) modes, and walks `(+1, -1)` across `L_inf`-normalized steps `(0.008, 0.025, 0.075, 0.18)`.
- **Dual Repair + Iterated Beam (4 rounds, beam=2, top-k=8)**:
  - **`rejam`**: Full-box soft-core SAT L-BFGS re-jam (`run_cpp_masked_melt` with all-1s mask, no ILS) + 2-stage `joint_xytheta_slp` + `fixed_angle_lp_tighten`.
  - **`mmelt` (Hybrid)**: Mode-activity pocket mask (top moving squares + spatial neighbors within `R=2.4`, capped at `<= 50%` of `n`) -> `run_cpp_masked_melt` (`ils=[0, 5, 10]`, `a_start=0.82`) -> 2-stage `joint_xytheta_slp` + `fixed_angle_lp_tighten`.
- **Certification**: Every exported group in `final.csv` is certified in BOTH `literal` and `pandas` roundtrip modes via `poc/auto_union/poc/union.py` against the full `origin/main` portfolio (`merged.csv` + all `poc/*/final.csv`), keeping only strict REAL WINs (`dL <= -1e-9`).

## Summary (18 Certified REAL WINs, total `ds = -3.780891e-04`)

| `n` | `dL` | `ds = dL / sqrt(n)` | Certified `L` (`literal == pandas`) | Incumbent `L` (`origin/main`) | Method Label |
|---:|---:|---:|---|---|---|
| 459 | `-4.203411e-05` | `-1.961984e-06` | `21.8995449567783883188` | `21.8995869908842722543` (merged.csv) | `rejam_r4_b1_m1_t2e-03_p_0.008` |
| 463 | `-1.423650e-05` | `-6.616263e-07` | `21.9680092460936040267` | `21.9680234825935569828` (merged.csv) | `mmelt_r2_b1_m0_t2e-05_p_0.025` |
| 499 | `-3.042003e-05` | `-1.361787e-06` | `22.8271722974585617946` | `22.8272027174862750860` (merged.csv) | `rejam_r1_b0_m0_t1e-03_p_0.008` |
| 542 | `-1.245804e-04` | `-5.351185e-06` | `23.7791613440577479432` | `23.7792859244148509427` (merged.csv) | `mmelt_r4_b1_m5_t5e-05_p_0.008` |
| 598 | `-5.807060e-06` | `-2.374684e-07` | `24.9025153229695168505` | `24.9025211300297414358` (merged.csv) | `mmelt_r2_b0_m4_t2e-05_p_0.008` |
| 639 | `-1.609637e-03` | `-6.367625e-05` | `25.7745013494666441294` | `25.7761109863485231000` (merged.csv) | `rejam_r3_b0_m4_t2e-03_p_0.008` |
| 686 | `-4.925768e-03` | `-1.880667e-04` | `26.6967227726035104825` | `26.7016485406314798330` (merged.csv) | `mmelt_r3_b0_m5_t2e-03_p_0.025` |
| 695 | `-9.573239e-06` | `-3.631337e-07` | `26.8317813069034272607` | `26.8317908801426572228` (merged.csv) | `rejam_r2_b1_m1_t1e-03_p_0.008` |
| 793 | `-3.459514e-05` | `-1.228509e-06` | `28.6593974870022634749` | `28.6594320821419367516` (merged.csv) | `mmelt_r4_b0_m4_t5e-05_m_0.008` |
| 804 | `-4.233042e-05` | `-1.492879e-06` | `28.8271625222459864040` | `28.8272048526637452877` (merged.csv) | `mmelt_r4_b0_m5_t4e-04_m_0.008` |
| 850 | `-9.313785e-05` | `-3.194602e-06` | `29.6489842772737736656` | `29.6490774151191365016` (merged.csv) | `mmelt_r1_b0_m7_t2e-04_m_0.008` |
| 852 | `-6.187867e-07` | `-2.119928e-08` | `29.6991307539681200181` | `29.6991313727548219428` (merged.csv) | `mmelt_r2_b0_m2_t2e-04_m_0.008` |
| 861 | `-3.009402e-05` | `-1.025601e-06` | `29.8170386514651322990` | `29.8170687454816695999` (merged.csv) | `rejam_r1_b0_m0_t1e-03_p_0.008` |
| 909 | `-2.290068e-03` | `-7.595675e-05` | `30.6467848003246143212` | `30.6490748679929315954` (merged.csv) | `rejam_r4_b0_m3_t5e-05_p_0.008` |
| 915 | `-7.741169e-05` | `-2.559151e-06` | `30.7403389315437687038` | `30.7404163432288493526` (merged.csv) | `rejam_r3_b0_m2_t4e-04_p_0.008` |
| 920 | `-5.739813e-06` | `-1.892360e-07` | `30.8018481040310359964` | `30.8018538438435622226` (merged.csv) | `rejam_r2_b1_m2_t2e-05_p_0.008` |
| 926 | `-4.400093e-04` | `-1.445960e-05` | `30.8706392703575229198` | `30.8710792796453282393` (merged.csv) | `rejam_r3_b0_m2_t4e-04_p_0.008` |
| 978 | `-5.091701e-04` | `-1.628147e-05` | `31.7538743812749641666` | `31.7543835513996504016` (merged.csv) | `rejam_r4_b0_m2_t2e-03_p_0.008` |

## Gate & Scale-Up Log (All 28 Evaluated `n`s: 18 REAL WINs, 10 Ties/Misses)

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
