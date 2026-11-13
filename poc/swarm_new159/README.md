# Hypothesis H13 (`poc/swarm_new159`): Idea 6 Structural Masked Pocket Melt on Newly Updated `origin/main` Packings

## Physical & Geometric Mechanism
1. **Interior Void & Contact-Stress Pocket Melting (`mpk_void`, `mpk_hot`, `mpk_seam`)**: Packings recently imported into `origin/main` (`poc/pell_corrugation/data/border-scan/merged.csv` at `be03ed7` / `9612bcb`) frequently have flush outer walls (`r_y0 = 0.500, t_x0 = 0.500`) while trapping excess bounding-box length `L` in interior contact-graph voids and over-constrained seam pockets.
2. **Pipeline**: Reuses `poc/swarm_highd/poc/` C++ `batch_masked_melt_solve` (`L0 = L_base + 0.032`, `a_start = 0.78 -> 1.00`, 4 stages, `masked_iters = 180`, `full_iters = 180`, `--threads 4`) across 24 masked-pocket starts (`R in [4.5, 7.2]`) per `n`:
   - `10 x mpk_void`: top interior clearance voids (`clearance_points`, alternating cavity-relocate and pure non-teleport void melts).
   - `8 x mpk_hot`: tightest local SAT contact-stress clusters (`tightest_contact_clusters`, half with weak-wall slack drift).
   - `6 x mpk_seam`: around `(k-0.5, k-0.5)`, `(L0-(k-0.5), L0-(k-0.5))`, and upright/tilted seam interface squares.
   Followed by `relax_lbfgs`, 2-stage `joint_xytheta_slp`, `fixed_angle_lp_tighten`, and canonical `poc/auto_union/poc/union.py` certification (both `Decimal` literal and `pandas` float64 roundtrip).

## Files
- `poc/pocket_seeds.py`: Tightest contact-stress detector (`tightest_contact_clusters`) and 24-seed H13 generator (`build_h13_seeds`).
- `poc/run_gate.py`: Gate and checkpointed scaling runner (`--stop-consec-nonreal 12`).
- `tests/test_swarm_new159.py`: Unit tests for contact cluster detection, seed shapes, and `final.csv` schema validation.
- `docs/gate_results.json`, `docs/scale_results.json`, `docs/ckpt/n_*.json`: Per-`n` checkpoints and diagnostics.
- `data/win_n*.csv`, `final.csv`: Certified strict REAL WINs (`dL <= -1e-9` in both literal and pandas modes; 0 ties).

## 15-Minute Falsification Gate + Batch 1 Results (vs `origin/main` `be03ed7` / `9612bcb`)
Gate command: `python -u poc/swarm_new159/poc/run_gate.py --targets 972 803 861 --threads 4` (`331.1s`) -> **PASSED (`n=861` REAL WIN, `dL = -5.647383e-04`)**.
Batch 1 scaling (`1493.4s`, `21` targets total, **10 REAL WINS / 0 TIES**, **`sum(ds) = -2.030092e-04`**):

| `n` | Status | Seed | `L_base` | `L_lit = L_pd` | `dL` | `ds` | `non0/90` |
|---:|---|---|---:|---:|---:|---:|---:|
| 499 | **REAL WIN** | `mpk_hot_2` | `22.827629367368791` | `22.827202717486275` | `-4.266499e-04` | `-3.903533e-05` | `149/499` |
| 585 | **REAL WIN** | `mpk_seam_1` | `24.705913670017148` | `24.705905418060527` | `-8.251957e-06` | `-6.969986e-07` | `180/585` |
| 587 | **REAL WIN** | `mpk_void_5` | `24.741865346534651` | `24.741854073297438` | `-1.127324e-05` | `-9.503266e-07` | `174/587` |
| 687 | **REAL WIN** | `mpk_void_1` | `26.718028330463901` | `26.717925937591301` | `-1.023929e-04` | `-7.964281e-06` | `225/687` |
| 695 | **REAL WIN** | `mpk_void_5` | `26.832528010318957` | `26.831790880142657` | `-7.371302e-04` | `-5.691739e-05` | `201/695` |
| 804 | **REAL WIN** | `mpk_hot_7` | `28.827472923190198` | `28.827204852663745` | `-2.680705e-04` | `-1.922328e-05` | `252/804` |
| 853 | **REAL WIN** | `mpk_hot_2` | `29.705918139236115` | `29.705908315604437` | `-9.823632e-06` | `-6.842203e-07` | `247/853` |
| 861 | **REAL WIN** | `mpk_seam_3` | `29.817633483810261` | `29.817068745481670` | `-5.647383e-04` | `-3.911498e-05` | `217/861` |
| 920 | **REAL WIN** | `mpk_hot_5` | `30.802405511271515` | `30.801853843843562` | `-5.516674e-04` | `-3.694029e-05` | `289/920` |
| 936 | **REAL WIN** | `mpk_void_7` | `30.995936464968079` | `30.995914086591432` | `-2.237838e-05` | `-1.482134e-06` | `331/936` |

Misses (`11/21`): `972` (`+1.558964e-04`), `803` (`+9.400874e-05`), `746` (`+4.694010e-06`), `333` (`+4.795502e-06`), `986` (`+7.002148e-05`), `913` (`+5.366419e-06`), `914` (`+3.527132e-05`), `980` (`+1.883981e-05`), `500` (`+1.129054e-05`), `918` (`+4.892980e-06`), `506` (`+5.061001e-06`).
