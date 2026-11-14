# Hypothesis H52 (`swarm_thinb4`): Compact `g=+4` & `g=+3` Corner Sub-Brick Extraction, Standalone Compression, & Zero-Overlap Pinwheel Broadcast

## Method

1. **Exact Corner Sub-Brick Extraction (`poc/thinb4_lib.py`)**:
   - Extracted self-contained corner sub-bricks (zero boundary-crossing squares, exact `w + f` by `h + f` bounding box) from `poc/pell_corrugation/data/border-scan/merged.csv` and `poc/swarm_mechwalk/data/` (`win_n909.json`, `win_n972.json`), normalized cleanly into `TL` (`[0, w+f] x [0, h+f]`) and `BR` orientations:
     - `g = +4`: `B4_9x16_n684` (`9x16`, `148` sq), `B4_17x9_n684` (`17x9`, `157` sq), `B4_10x16_n737` (`10x16`, `164` sq), `B4_11x16_n849` (`11x16`, `180` sq), `B4_18x12_n849` (`18x12`, `220` sq), `B4_13x16_n908` (`13x16`, `212` sq), `B4_17x13_n908` (`17x13`, `225` sq), `B4_16x14_n969` (`16x14`, `228` sq), plus stripped `12x16` (`B4_12x16_n908`).
     - `g = +3` / `+2`: `B3_9x15_n848` (`9x15`, `138` sq), `B2_10x6_n848` (`10x6`, `62` sq), and stripped `14x9` (`B3_14x9_n632`) and `10x14` (`B3_10x14_n736`).
     - `g = +5` / `+6`: `B5_10x19_n793` (`10x19`, `195` sq), `B5_10x20_b4` (`10x20`, `205` sq), `B6_10x21_b4` (`10x21`, `216` sq), plus stripped `10x19` (`n=850`), `10x20` (`n=850`), `11x20` (`b4`).
   - Standalone rectangular compression (`joint_xytheta_slp_rect` + rejam/mmelt) cached in `data/compressed_bricks_cache.json`:
     - `B4_12x16_n908`: `0.647598 -> 0.636884` (`df = -1.071e-02`)
     - `B5_10x19_n850`: `0.893664 -> 0.672754` (`df = -2.209e-01`)
     - `B3_14x9_n632`: `0.636374 -> 0.636136` (`df = -2.382e-04`)
     - `B5_11x20_b4`: `0.640084 -> 0.640033` (`df = -5.083e-05`)
     - `B5_10x20_b4`: `0.685502 -> 0.642550`
     - `B5_10x20_n850`: `0.642572 -> 0.642555` (`df = -1.746e-05`)
     - `B3_10x14_n736`: `0.634366 -> 0.634363` (`df = -3.030e-06`)

2. **Zero-Overlap Pinwheel Assembly + H32 Pre-Shifted Seam Slack + C++ Masked Melt**:
   - True zero-overlap pinwheel assembly requires `w1 + w2 <= m`, `h1 + h2 <= m`, and strict slack along at least one axis (`w1 + w2 < m` or `h1 + h2 < m`) so opposite `TL`/`BR` corners avoid colliding in the central `f x f` square.
   - Pre-shifts opposite `TL`/`BR` pairs across `shift_fracs` along slack axes, applies `run_cpp_masked_melt` seam-targeted perturbations (`build_thinb4_melt_seeds`) + `joint_xytheta_slp_tight`, and polishes certified wins with `mechwalk` nullspace kicks (`polish_and_export.py --polish --prune`).

## Certified Results (vs `origin/main` @ `ead0ace`)

Run over 28 targets across `r = +7, +8, +9, +10` (`docs/scale.log`, `docs/scale_ckpt.json`) initially yielded 7 wins (`632, 683, 793, 849, 850, 909, 910`), of which `n=793, 909, 910` were superseded by PR #43 / PR #45 merged into `origin/main` (`ead0ace`) and pruned from `data/` and `final.csv`.

The remaining **4 strict REAL WINs** (`dL <= -1e-9`, verified in both `literal` and `pandas` modes via `poc/auto_union/poc/union.py`) total **`ds = -1.001487e-04`**:

| `n` | `m` | `r` | Incumbent `L` (`origin/main`) | New `L` (`literal == pandas`) | `dL` | `ds` | Non-`0/90` | Seed / Method |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **632** | 25 | +7 | `25.6388918166912...` | `25.6379246123087816155` | `-9.672044e-04` | `-7.847336e-05` | 193/632 | `B3_14x9_n632_cmp+B4_10x16_n737_zw_sf0.8_s00_tb4m_bl_quad_13_mw` |
| **683** | 26 | +7 | `26.6370294017538...` | `26.6368843610871182471` | `-1.450407e-04` | `-1.131315e-05` | 190/683 | `B3_14x9_n632_cmp+B4_12x16_n908_cmp_zw_sf0.0_s00_direct` |
| **849** | 29 | +8 | `29.6370294017538...` | `29.6368833849497471318` | `-1.460168e-04` | `-1.019433e-05` | 214/849 | `B4_12x16_n908_cmp+B4_12x16_n908_cmp_T_zw_sf0.0_s00_b5m_bl_quad_5` |
| **850** | 29 | +9 | `29.6425721155181...` | `29.6425697086158730688` | `-2.406902e-06` | `-1.678748e-07` | 242/850 | `B4_9x16_n684+B5_11x20_b4_cmp_T_zw_sf0.0_s00_direct` |

## Layout & Reproduction

- [poc/thinb4_lib.py](poc/thinb4_lib.py): corner sub-brick extraction, standalone brick compression, zero-overlap/pre-shifted assembly, masked melt seeds, certification.
- [poc/run_gate.py](poc/run_gate.py): 15-minute Step 1 falsification gate (`{684, 849, 737}` -> [docs/gate_report.json](docs/gate_report.json)).
- [poc/run_scale.py](poc/run_scale.py): Step 2 scale-up across 28 targets in `r = +7, +8, +9, +10` (`docs/scale.log`, `docs/scale_ckpt.json`).
- [poc/polish_and_export.py](poc/polish_and_export.py): polish, prune superseded groups vs `origin/main`, verify, and export [final.csv](final.csv).
- [tests/test_thinb4.py](tests/test_thinb4.py): unit tests for corner extraction and zero-overlap pinwheel assembly.

```bash
/tmp/highrpeel_venv/bin/pytest -v poc/swarm_thinb4/tests/test_thinb4.py
/tmp/highrpeel_venv/bin/python -u poc/swarm_thinb4/poc/run_gate.py
/tmp/highrpeel_venv/bin/python -u poc/swarm_thinb4/poc/run_scale.py
/tmp/highrpeel_venv/bin/python -u poc/swarm_thinb4/poc/polish_and_export.py --polish --prune
```
