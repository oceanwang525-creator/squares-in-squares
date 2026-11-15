# H98: Upward Constant-`r` `(m-1, r) -> (m, r)` L-Border & Cross-Corridor Expansion (`upwardcasc`)

## Mechanism
Given a donor packing for `n_donor = (m - 1)^2 + r` with side length `L(m - 1, r)`, expanding upward to `n_target = m^2 + r` adds `2m - 1` unit squares across a `+1.0` span (`L_0 = L(m - 1, r) + 1.0`) via two complementary operators:

1. **Zero-Overlap Outer L-Border (`extTR`, `extTL`, `extBR`, `extBL`, pack & spread)**: Shifts/places `2m - 1` axis-aligned (`0 deg`) unit squares along two adjacent outer walls, guaranteeing zero overlap (`pen0 = 0.0`) at `L_0 = L(m - 1, r) + 1.0`.
2. **Interior Cross-Corridor Expansion (`xcross_expand` & `tsplit`)**: Finds the cleanest interior vertical/horizontal split lines `(x0, y0)` cutting the fewest tilted squares, opens a `+1.0` cross corridor, and inserts `2m - 1` `0 deg` unit squares into the opened corridor before running `joint_xytheta_slp`, `mechwalk_compress`, and `lpslack`.

When `n_target` is plateau-tied or within `~1e-4` of `L(m - 1, r) + 1.0`, opening a `+1.0` L-border or interior cross corridor breaks boundary lock-in so `mechwalk_compress` + `lpslack` compresses `L` below `L_inc` — and any resulting strict win cascades upward to `(m + 1, r)`.

## Dual-Certified Wins (`literal == pandas`, vs `origin/main` `cc58c7e`)
All 3 certified wins in `poc/swarm_upwardcasc/final.csv`:

| `n` | `(m, r)` | Donor | Method | `L_old` (`origin/main`) | `L_new` | `dL` | `ds` (`L/sqrt(n)-1`) | `ds_L2n` (`L^2/n`) |
| ---: | :---: | ---: | :--- | :--- | :--- | ---: | ---: | ---: |
| `530` | `(23, +1)` | `485` | `mech_xcross_clean_10_x3.0_y1.5_485` | `23.525734298491243836...` | `23.525695710243513879...` | `-3.858825e-05` | `-1.676166e-06` | `-3.425721e-06` |
| `577` | `(24, +1)` | `530` (win) | `lpslack_extBR_spread_530` (2-hop cascade) | `24.525734299491243836...` | `24.525695695797764590...` | `-3.860369e-05` | `-1.607093e-06` | `-3.281744e-06` |
| `582` | `(24, +6)` | `535` | `upward_mechwalk` + `lpslack` | `24.633982502102097457...` | `24.633947688045686680...` | `-3.481406e-05` | `-1.443089e-06` | `-2.947107e-06` |

- **Total (`3/3` strict wins)**: `total_ds = -4.726348e-06`, `total_ds_L2n = -9.654573e-06`.
- Note: Earlier intermediate wins against `9abec6b`/`254bb0f` (`n=332<-297` `18.7394555859` `-9.617e-4`, `n=585<-538` `24.7058823631`, `n=853<-796` `29.7058823780`) were superseded by `poc/external/` / `poc/external_union_nov14/` in `origin/main` `cc58c7e` and pruned from `data/` and `final.csv`.

## Layout & Verification
- `poc/scan_pairs.py`, `poc/scan_xcross_expand.py`: Pair & clean-cut diagnostics over `1..1000`.
- `poc/upward_lib.py`, `poc/upward_engine.py`: Zero-overlap L-border & `xcross_expand` generators, SLP/`mechwalk_compress`/`lpslack` pipeline, dual `literal`/`pandas` certification.
- `poc/run_gate.py`, `poc/run_scale.py`, `poc/run_finish.py`: Gate, full 46-pair scale sweep, and `254bb0f`/`cc58c7e` follow-up scripts.
- `tests/test_upward.py`: Unit tests for L-border zero-overlap invariant and `xcross_expand` geometry.

Run tests and structure check:
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 poc/baseline/poc/.venv/bin/pytest -v poc/swarm_upwardcasc/tests/
BASE_SHA=$(git rev-parse origin/main) HEAD_SHA=$(git rev-parse HEAD) poc/baseline/poc/.venv/bin/python3 .github/scripts/check_final_csv.py
```
