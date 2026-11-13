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

## Batch 2 Scaling Results (`33` targets from `#183`, vs `origin/main` `252fe8b` `merged.csv` & `guestdata_atlas/final.csv`)
Command: `python -u poc/swarm_new159/poc/run_gate.py --targets 861 ... 464 --threads 4 --stop-consec-nonreal 12 --out-json poc/swarm_new159/docs/scale_b2_results.json` (`2494.4s`) -> **19 NEW REAL WINS / 0 TIES**, **`B2_ds_exact = -2.185313e-04`** (29 total certified REAL WINs in `final.csv`, combined `ds_exact = -3.182608e-04`):

| `n` | Status | Seed | `L_base` | `L_lit = L_pd` | `dL` | `ds` (exact) | `non0/90` |
|---:|---|---|---:|---:|---:|---:|---:|
| 372 | **REAL WIN** | `mpk_hot_4` | `19.793396182035972` | `19.793383173313466` | `-1.300872e-05` | `-6.744708e-07` | `102/372` |
| 401 | **REAL WIN** | `mpk_hot_4` | `20.533704455357992` | `20.533532374172990` | `-1.720812e-04` | `-8.593324e-06` | `209/401` |
| 459 | **REAL WIN** | `mpk_void_6` | `21.901136288038465` | `21.899586990884272` | `-1.549297e-03` | `-7.231500e-05` | `194/459` |
| 464 | **REAL WIN** | `mpk_hot_0` | `21.983366795247327` | `21.983258973386352` | `-1.078219e-04` | `-5.005504e-06` | `141/464` |
| 540 | **REAL WIN** | `mpk_void_9` | `23.755398075531024` | `23.755246283753560` | `-1.517918e-04` | `-6.532078e-06` | `184/540` |
| 634 | **REAL WIN** | `mpk_void_7` | `25.689163011628471` | `25.688386061294070` | `-7.769503e-04` | `-3.085664e-05` | `204/634` |
| 642 | **REAL WIN** | `mpk_void_5` | `25.817412507584841` | `25.817307926116020` | `-1.045815e-04` | `-4.127501e-06` | `167/642` |
| 700 | **REAL WIN** | `mpk_seam_4` | `26.903174165880785` | `26.903132428296752` | `-4.173758e-05` | `-1.577532e-06` | `256/700` |
| 703 | **REAL WIN** | `mpk_hot_3` | `26.940218535129839` | `26.940185998202880` | `-3.253693e-05` | `-1.227153e-06` | `299/703` |
| 736 | **REAL WIN** | `mpk_hot_6` | `27.634480510465057` | `27.634370196420712` | `-1.103140e-04` | `-4.066230e-06` | `188/736` |
| 797 | **REAL WIN** | `mpk_hot_0` | `28.732796468402374` | `28.732780533259013` | `-1.593514e-05` | `-5.644517e-07` | `245/797` |
| 798 | **REAL WIN** | `mpk_seam_2` | `28.740789019969816` | `28.740492163895733` | `-2.968561e-04` | `-1.050859e-05` | `258/798` |
| 811 | **REAL WIN** | `mpk_seam_1` | `28.914402401907053` | `28.914389101343317` | `-1.330056e-05` | `-4.670460e-07` | `290/811` |
| 852 | **REAL WIN** | `mpk_void_7` | `29.699145863828089` | `29.699131372754821` | `-1.449107e-05` | `-4.964560e-07` | `316/852` |
| 875 | **REAL WIN** | `mpk_void_9` | `29.988702612665090` | `29.988448134153464` | `-2.544785e-04` | `-8.602944e-06` | `281/875` |
| 922 | **REAL WIN** | `mpk_hot_3` | `30.826655323120065` | `30.826591075074133` | `-6.424805e-05` | `-2.115897e-06` | `229/922` |
| 926 | **REAL WIN** | `mpk_hot_1` | `30.871594204051698` | `30.871079279645328` | `-5.149244e-04` | `-1.692147e-05` | `284/926` |
| 978 | **REAL WIN** | `mpk_hot_2` | `31.755750873640405` | `31.754383551399650` | `-1.367322e-03` | `-4.372214e-05` | `330/978` |
| 988 | **REAL WIN** | `mpk_void_8` | `31.873091167770951` | `31.873086236830360` | `-4.930941e-06` | `-1.568741e-07` | `319/988` |

*(Note: on `n=401`, H13 beats `252fe8b` `merged.csv` and `guestdata_atlas/final.csv` by `-1.720812e-04`, while PR #24 `swarm-armslack` achieved `20.533530106866359`; `auto_union` selects the best per `n`.)*

Batch 2 misses (`14/33`): `971` (`+3.408943e-06`), `854` (`+1.906906e-05`), `919` (`+2.123948e-05`), `593` (`+2.585452e-06`), `692` (`+5.848021e-06`), `750` (`+3.185910e-06`), `987` (`+3.115340e-05`), `923` (`+1.377245e-05`), `740` (`+5.163857e-07`), `868` (`+4.572596e-06`), `599` (`+8.155674e-07`), `858` (`+3.473638e-06`), `644` (`+1.017994e-05`), `597` (`+4.138598e-05`).

