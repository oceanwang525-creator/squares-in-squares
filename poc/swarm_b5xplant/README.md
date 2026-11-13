# Hypothesis H40: `B5_10x20` (`g=+5`) & `B4_18x9` (`g=+4`) Sub-Brick Transplant (`poc/swarm_b5xplant/`)

Self-contained harvest and broadcast of the `n=850` (`L = 29.642573012673164`, `f = 0.642572`) tilted sub-bricks across `r in {+5, +8, +9, +10, +11}` (`m = 19..31`).

## Geometric Mechanism

From `data/donor_win_n850.json` (`n=850`), cutting cleanly along the interior `1x1` grid seams isolates two compact tilted sub-bricks and their unit-stripped directional variants:
- **`B5_10x20`** (`205` squares = `10*20 + 5`, `g = +5`): `x + h <= 10.65`, `y - h >= 8.999` (`fx = 0.642571`, `fy = 0.642572`, beats prior `g=+5` `f=0.679381` by `-0.03681`), plus directional bottom-row-stripped **`B5_10x19`** (`195` sq, `fx = 0.642571`, `fy = 0.893664`).
- **`B4_18x9`** (`166` squares = `18*9 + 4`, `g = +4`): `x - h >= 10.999`, `y + h <= 9.65` (`fx = 0.642572`, `fy = 0.642571`), plus column-stripped variants **`B4_17x9`** (`157` sq), **`B4_16x9`** (`148` sq), and **`B4_15x9`** (`139` sq).
- Paired with `B6_10x21` (`w971` / `w910`), `B4_19x9`, and `clean_632`/`clean_914` via `D2` brick orientation pairs, unit-grid infill, C++ batch masked seam melt (`run_cpp_masked_melt`), tight multi-stage `joint_xytheta_slp_tight`, and `fixed_angle_lp_tighten`.

## Certified Real Wins vs `origin/main` (`37be2a8`)

All 4 wins in `final.csv` are dual-certified (`literal == pandas`) via `poc/auto_union/poc/union.py` and pass `.github/scripts/check_final_csv.py`:

| `n` | `L_inc` (`37be2a8`) | `L_lit` (= `L_pd`) | `dL` | `ds` | `non0/90` | Recipe |
|---:|---:|---:|---:|---:|---:|---|
| 909 | `30.6425760212162...` | `30.6407858534571673104...` | `-1.790168e-03` | `-1.206903e-04` | `351/909` | `B5_10x20_n850+clean_632_tight_s20_b1_b5m_bl_quad_5` |
| 910 | `30.6570963197696...` | `30.6565356804301738637...` | `-5.606393e-04` | `-3.777454e-05` | `330/910` | `B4_18x9_n850+B6_10x21_w971_tight_s02_b1_b5m_tr_corner_15` |
| 971 | `31.6565261466309...` | `31.6425492618212468848...` | `-1.397688e-02` | `-9.111472e-04` | `347/971` | `B5_10x20_n850+B5_10x20_n850_T_tight_s00_b1_b5m_b1_bot_edge_19` |
| 972 | `31.6793812503515...` | `31.6566704049228508650...` | `-2.271085e-02` | `-1.479851e-03` | `378/972` | `B5_10x20_n850_T+B6_10x21_w971_tight_s02_b1_b5m_bl_quad_13` |

**Total `ds = -2.549463e-03`** across 4 wins (`n = 909, 910, 971, 972`).

## Scale-Up Notes (`docs/scale.log`)

- **Gate & scale wins**: `n=971` (`-1.398e-02`), `n=972` (`-2.271e-02`), `n=909` (`-1.790e-03`), `n=910` (`-5.606e-04`) all landed where brick extents fit disjointly or with `<= 1` unit-column stepped seam.
- **Misses so far**: `n=793` (`dL=+1.12e-02`), `n=911` (`dL=+1.39e-02`), `n=851` (`dL=+1.31e-02`), `n=852` (`dL=+1.03e-01`), `n=794` (`dL=+2.93e-02`), `n=738` (`dL=+2.50e-02`), `n=685` (`dL=+4.91e-02`), `n=795` (`dL=+2.51e-03`), `n=739` (`dL=+1.60e-01`), `n=740` (`dL=+1.78e-02`), where `20.64 + 10.64 > m` forces `pen0 >= 0.54..0.64` core corner collisions. `run_scale.py` continues evaluating remaining targets in background.

## Verification

```bash
/tmp/highrpeel_venv/bin/python -u poc/swarm_b5xplant/poc/polish_and_export.py --no-polish
/tmp/highrpeel_venv/bin/python -m pytest -q poc/baseline/poc/tests/ poc/auto_union/poc/ .github/scripts/test_check_final_csv.py poc/swarm_b5xplant/tests/
BASE_SHA=37be2a805f4e4425babbfd64e417b7bbf9f1e953 HEAD_SHA=HEAD /tmp/highrpeel_venv/bin/python .github/scripts/check_final_csv.py
```
