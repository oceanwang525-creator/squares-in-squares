# Hypotheses H40, H50 & H51: Sub-Brick Harvest & Compressed `B5`/`B6` Transplant (`poc/swarm_b5xplant/`)

Self-contained harvest, sub-brick compression, and broadcast of tilted `g in {+4, +5, +6}` rectangular sub-bricks across `r in {+5, +8..+12}` (`m = 19..31`), including:
- **H40 (`run_gate.py`, `run_scale.py`, PR #41)**: `B5_10x20` (`g=+5`), `B4_18x9` (`g=+4`), and directional stripped variants from `data/donor_win_n850.json` (`n=850`), plus `B6_10x21` / `clean_632` / `clean_914`.
- **H50 & H51 (`run_h50_h51.py`, PR2)**: Harvest & sub-brick SLP compression from `mechwalk` Batch 4 (`data/donor_b4_win_n909.json` `L=30.640084506`, `data/donor_b4_win_n972.json` `L=31.656533614` — noting `mechwalk` JSON `"angle"` values are stored in radians) and `origin/main` (`n=970`, `n=793`), combined with `regrid` corner-interlock (`interlock_lib.assemble_interlock`), `L_inc`-targeted melt headroom schedules (`build_h50_melt_seeds`), multi-stage tight `joint_xytheta_slp_tight`, and `fixed_angle_lp_tighten`.

## Certified Real Wins vs `origin/main` (`852beed`)

All 5 wins in `final.csv` are dual-certified (`literal == pandas`) via `poc/auto_union/poc/union.py` and pass `.github/scripts/check_final_csv.py`:

| `n` | `L_inc` (`852beed`) | `L_lit` (= `L_pd`) | `dL` | `ds` | `non0/90` | Recipe |
|---:|---:|---:|---:|---:|---:|---|
| 740 | `27.7059044664630...` | `27.7059035057599362139...` | `-9.607031e-07` | `-7.193824e-08` | `331/740` | `B5_17x9_w911+clean_914_tight_s01_b1_h50_b1_bot_right_10` |
| 850 | `29.6425730132796...` | `29.6425721155181840327...` | `-8.977614e-07` | `-6.261637e-08` | `348/850` | `B5_10x20_n850_T+B4_18x9_n850_T_tight_s00_b1_direct` |
| 911 | `30.6841505804990...` | `30.6783303213147234149...` | `-5.820259e-03` | `-3.920368e-04` | `382/911` | `B5_10x20_b4_slp+B6_10x21_b4_T_tight_s03_b1_h50_tr_corner_23` |
| 971 | `31.6425492618212...` | `31.6424851207162927578...` | `-6.414110e-05` | `-4.180404e-06` | `365/971` | `B5_10x20_b4_slp_T+B5_10x20_b4_slp_tight_s00_b1_h50_b1_bot_right_10` |
| 973 | `31.6966063160911...` | `31.6818316376285653133...` | `-1.477468e-02` | `-9.623803e-04` | `421/973` | `B6_10x21_w971+B6_10x21_w971_T_tight_s11_b1_h50_bl_corner_14` |

**Total `ds = -1.358732e-03`** across 5 certified wins (`n = 740, 850, 911, 971, 973`). (Earlier PR #41 wins `909`, `910`, `972` are already merged on `origin/main` `852beed` and excluded from PR2 `final.csv`.)

## Verification

```bash
/tmp/highrpeel_venv/bin/python -u poc/swarm_b5xplant/poc/polish_and_export.py --no-polish
/tmp/highrpeel_venv/bin/python -m pytest -q poc/baseline/poc/tests/ poc/auto_union/poc/ .github/scripts/test_check_final_csv.py poc/swarm_b5xplant/tests/
BASE_SHA=$(git rev-parse origin/main) HEAD_SHA=$(git rev-parse HEAD) /tmp/highrpeel_venv/bin/python .github/scripts/check_final_csv.py
```
