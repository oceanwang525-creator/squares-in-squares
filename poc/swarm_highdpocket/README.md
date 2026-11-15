# `swarm_highdpocket` (H115: High-`D` Fine Sub-Pocket Scan + Soft-Seam & Overhang DP)

Cycle 22 Rank 3 Big-Move Worker targeting `F3B_high_r` (`D = 15..28, K = 18..31`, where `n = K^2 + D <= 1000`).

## Method

1. **Union Baseline & Donor Collection (`poc/build_union.py`)**:
   - Evaluates the live union of `poc/pell_corrugation/data/border-scan/merged.csv` and all `poc/*/final.csv` submissions using `poc/auto_union/poc/union.py`.
   - Extracts brick sub-pockets from `swarm_gstep`, `swarm_b7xplant`, `swarm_thinb4`, and `swarm_peelstep`.
2. **Fine High-`D` Sub-Pocket Scan & Sanitization (`poc/run_scan_b1.sh`, `poc/run_scan_b2.sh`, `poc/run_scan_b3.sh`, `poc/prepare_pockets.py`, `poc/merge_all.py`)**:
   - Scans `D = 14..34` donor rows at `step = 0.0625` and `step = 0.05` using `poc/pocket_lift_scan3/poc/scan_pockets.py`.
   - Sanitizes all pockets with `poc/pocket_lift_scan3/poc/sanitize.py` and merges them with `pockets_pareto_e3_sane.json`, `pockets_partb_sane.json`, `gadlib` pockets, and bricks into:
     - `data/pockets_all_sane.json` (Pareto-pruned for seam DP)
     - `data/pockets_all_nopareto.json` (best `(w, h, g)` without Pareto pruning across exact dimensions, for exact-sum overhang DP)
3. **Composition & Polishing**:
   - **Slack-Tolerant 2-Pocket Staircase DP (`poc/seam_build_polish.py` + `poc/swarm_softseamdp/poc/dp_soft.py`)**: `sum w <= K`, `sum h <= K`, `sum g = D`, `dL_rigid <= 6.0e-3`, followed by C++ seam melt, `lpslack`, `pslp.squeeze`, and `mechwalk_polish_once`.
   - **1-Axis Overhang Pinwheel DP (`poc/overhang_dp.py`)**: `kx + ky = 1, |I| = 0, g_1 + g_2 - 1 = D`, `assemble_zero_gap_50d`, `lpslack`, `pslp.squeeze`, and `mechwalk_polish_once`.

## PR2 Dual-Certified Wins (`origin/main` `ccdac8f`, `literal == pandas`)

PR1 wins (`n=857`, `n=915`, `n=978`) are merged and preserved identically in `origin/main` `merged.csv`.

| `n` | `K` | `D` | Construction | Incumbent `L` | New `L` | `dL` |
|---:|---:|---:|---|---:|---:|---:|
| 748 | 27 | 19 | `oh_p0_s00_804+804+mw` | `27.8267634660094332092` | `27.8267531676012362018` | `-1.029840820e-05` |
| 805 | 28 | 21 | `oh_p0_s00_985+644` | `28.8434410518001599164` | `28.8394841288806526610` | `-3.956922920e-03` |

## PR1 Dual-Certified Wins (Merged into `origin/main`)

| `n` | `K` | `D` | Construction | Incumbent `L` | New `L` | `dL` |
|---:|---:|---:|---|---:|---:|---:|
| 857 | 29 | 16 | `oh_p0_s00_10917+541` | `29.7615364017008534382` | `29.7608080002709241284` | `-7.284014299e-04` |
| 915 | 30 | 15 | `(18,8,+6)<539> + (12,21,+9)<915T>` | `30.7403339912685678903` | `30.7403227283026149572` | `-1.126296595e-05` |
| 978 | 31 | 17 | `(9,18,+7)<10916T> + (22,12,+10)<10743T>` | `31.7538529591753972428` | `31.7538528313463733561` | `-1.278290239e-07` |

## Verification

```bash
BASE_SHA=$(git rev-parse origin/main) HEAD_SHA=$(git rev-parse HEAD) \
  /tmp/highrpeel_venv/bin/python .github/scripts/check_final_csv.py
/tmp/highrpeel_venv/bin/python -m pytest -v poc/swarm_highdpocket/tests/test_highdpocket.py
```
