# Hypotheses H40, H50, H51 & H53–H55: Sub-Brick Harvest, SLP Compression & Pinwheel Transplant (`poc/swarm_b5xplant/`)

Self-contained harvest, sub-brick compression, and broadcast of tilted `g in {+4, +5, +6}` rectangular sub-bricks across `r in {+5, +8..+12}` (`m = 19..31`), including:
- **H40 (`run_gate.py`, `run_scale.py`, PR #41)**: `B5_10x20` (`g=+5`), `B4_18x9` (`g=+4`), and directional stripped variants from `data/donor_win_n850.json` (`n=850`), plus `B6_10x21` / `clean_632` / `clean_914`.
- **H50 & H51 (`run_h50_h51.py`, PR #44)**: Harvest & sub-brick SLP compression from `mechwalk` Batch 4 (`data/donor_b4_win_n909.json`, `data/donor_b4_win_n972.json`) and `origin/main`, combined with `regrid` corner-interlock, `L_inc`-targeted melt headroom schedules, multi-stage tight `joint_xytheta_slp_tight`, and `fixed_angle_lp_tighten`.
- **H53–H55 (`run_h53_h55.py`, PR3)**: Ultra-thin sub-brick harvest & `joint_xytheta_slp_rect` compression (`B5_20x11_n909`, `B5_20x10_n909_slp`, `B4_10x16_n909`, `B4_11x16_n632`, `B4_10x16_n632_slp`, `B4_12x16_n849`, `B4_11x16_n849`, `B4_9x16_n684(_slp)`, `B6_9x18_n796(_slp)`, `B6_11x18_n585(_slp)`, `B5_9x17_n739(_slp)` — cached in `data/h53_compressed_cache.json`) from `thinb4` donors (`data/donor_thin_win_n{632,849,909}.json`) and `origin/main`, paired via zero-overlap (`w1+w2 <= m`, `h1+h2 <= m`, `w1+w2+h1+h2 <= 2m-1`) pre-shifted pinwheels (`sf in {0.0, 0.5, 0.8}`) and minimal-overlap (`kx+ky <= 2`) `B5`/`B6` seams + `L_inc`-targeted masked melt + SLP/mechwalk certification.

## Certified Real Wins vs `origin/main` (`d12df26`)

All 5 wins in `final.csv` are dual-certified (`literal == pandas`) via `poc/auto_union/poc/union.py` and pass `.github/scripts/check_final_csv.py`:

| `n` | `L_inc` (`d12df26`) | `L_lit` (= `L_pd`) | `dL` | `ds` | `non0/90` | Recipe |
|---:|---:|---:|---:|---:|---:|---|
| 737 | `27.6388918153424...` | `27.6388358149218295782...` | `-5.600042e-05` | `-4.200239e-06` | `326/737` | `B4_11x16_n632+B4_11x16_n849_T_ov_s30_h50_b1_bot_edge_19` |
| 796 | `28.7059057019294...` | `28.7059025286906601302...` | `-3.173239e-06` | `-2.288711e-07` | `336/796` | `B6_9x18_n796_slp+B6_9x18_n796_slp_T_z_sf0_s00_direct` |
| 853 | `29.7058876665656...` | `29.7058862180581697233...` | `-1.448507e-06` | `-1.008891e-07` | `334/853` | `B6_11x18_n585_slp+B6_11x18_n585_T_ov_s23_h50_bl_quad_5` |
| 909 | `30.6388990056089...` | `30.6388924193297404742...` | `-6.586279e-06` | `-4.439963e-07` | `372/909` | `B4_10x16_n909+B5_20x11_n909_z_sf0_s00_h50_b1_bot_edge_11` |
| 970 | `31.6389817979899...` | `31.6387122152024988918...` | `-2.695828e-04` | `-1.758616e-05` | `332/970` | `B4_11x16_n849+B5_20x11_n909_z_sf0_s00_h50_b1_bot_right_10` |

**Total `ds = -2.256016e-05`** across 5 certified wins (`n = 737, 796, 853, 909, 970`). (Earlier PR #41 wins `909, 910, 972` and PR #44 wins `740, 850, 911, 971, 973` are already merged on `origin/main` `d12df26` and excluded from PR3 `final.csv`.)

## Verification

```bash
/tmp/highrpeel_venv/bin/python -u poc/swarm_b5xplant/poc/polish_and_export.py --no-polish
/tmp/highrpeel_venv/bin/python -m pytest -q poc/baseline/poc/tests/ poc/auto_union/poc/ .github/scripts/test_check_final_csv.py poc/swarm_b5xplant/tests/
BASE_SHA=$(git rev-parse origin/main) HEAD_SHA=$(git rev-parse HEAD) /tmp/highrpeel_venv/bin/python .github/scripts/check_final_csv.py
```
