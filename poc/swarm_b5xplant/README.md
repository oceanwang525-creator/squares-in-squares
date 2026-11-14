# Hypotheses H40, H50, H51, H53–H55 & H63: Sub-Brick Harvest, SLP Compression & Pinwheel Transplant (`poc/swarm_b5xplant/`)

Self-contained harvest, sub-brick compression, and broadcast of tilted `g in {+4..+8}` rectangular sub-bricks across `r in {+5, +8..+13}` (`m = 19..31`), including:
- **H40 (`run_gate.py`, `run_scale.py`, PR #41)**: `B5_10x20` (`g=+5`), `B4_18x9` (`g=+4`), and directional stripped variants from `data/donor_win_n850.json` (`n=850`), plus `B6_10x21` / `clean_632` / `clean_914`.
- **H50 & H51 (`run_h50_h51.py`, PR #44)**: Harvest & sub-brick SLP compression from `mechwalk` Batch 4 (`data/donor_b4_win_n909.json`, `data/donor_b4_win_n972.json`) and `origin/main`, combined with `regrid` corner-interlock, `L_inc`-targeted melt headroom schedules, multi-stage tight `joint_xytheta_slp_tight`, and `fixed_angle_lp_tighten`.
- **H53–H55 (`run_h53_h55.py`, PR #53)**: Ultra-thin sub-brick harvest & `joint_xytheta_slp_rect` compression (`data/h53_compressed_cache.json`) from `thinb4` donors (`data/donor_thin_win_n{632,849,909}.json`) and `origin/main`, paired via zero-overlap pre-shifted pinwheels (`sf in {0.0, 0.5, 0.8}`) and minimal-overlap `B5`/`B6` seams.
- **H63 (`run_h63_high_r.py`, PR4)**: Harvest & `joint_xytheta_slp_rect` compression of newly ratcheted sub-bricks from `data/donor_h63_*` (`mechwalk` `n=970`, `b8xplant` `n=909/737` & standalone `B5..B8` bricks, cached in `data/h63_compressed_cache.json`) + `origin/main`, paired via `assemble_preshifted_flex` (`sf=0.5` slack sharing between `B5_20x10_n909_slp` and `B6_10x21_n972_b8_slp`).

## Certified Real Wins vs `origin/main` (`409b443`)

All wins in `final.csv` are dual-certified (`literal == pandas`) via `poc/auto_union/poc/union.py` and pass `.github/scripts/check_final_csv.py`:

| `n` | `L_inc` (`409b443`) | `L_lit` (= `L_pd`) | `dL` | `ds` | `non0/90` | Recipe |
|---:|---:|---:|---:|---:|---:|---|
| 972 | `31.6565325460975...` | `31.6565320383354242193...` | `-5.077622e-07` | `-3.307405e-08` | `371/972` | `B5_20x10_n909_slp+B6_10x21_n972_b8_slp_z_sf5_s00_h50_b1_bot_edge_3` |

**Total `ds = -3.307405e-08`** across 1 certified win (`n = 972`). (Earlier merged wins `737, 740, 796, 850, 853, 909, 910, 911, 970, 971, 972, 973` are already in `origin/main` `409b443` and excluded from PR4 `final.csv`.)

## Verification

```bash
/tmp/highrpeel_venv/bin/python -u poc/swarm_b5xplant/poc/polish_and_export.py --no-polish
/tmp/highrpeel_venv/bin/python -m pytest -q poc/baseline/poc/tests/ poc/auto_union/poc/ .github/scripts/test_check_final_csv.py poc/swarm_b5xplant/tests/
BASE_SHA=$(git rev-parse origin/main) HEAD_SHA=$(git rev-parse HEAD) /tmp/highrpeel_venv/bin/python .github/scripts/check_final_csv.py
```
