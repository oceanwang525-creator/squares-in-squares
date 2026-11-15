# H114 (`vstrip852`): Periodic-Strip Splicing + Corner/Zero-Overlap Pinwheel Composition + `pslp` Polish

Synthesizes new `g = +6, +7, +8` sub-pockets by extracting constituent sub-pockets from the `n=852` (`(11, 20, +6)` and `(17, 9, +5)`), `n=912` (`(9, 21, +6)`), and `n=974` (`(9, 21, +6)`, `(22, 10, +7)`) packings along with `gstep`/`b8xplant` bricks, applying `gstep_lib` periodic vertical/horizontal strip splicing (`vstrip`/`hstrip`) and 1-row/1-col outer peels (`peel_outer_unit_layer`), compressing with `pslp` (`squeeze` + `shake_squeeze`) and `brick_mechwalk`, and composing 4-corner pinwheels (`zero` non-overlapping and `1x1` corner-melted) followed by `lpslack` + `pslp` compression.

## Key Technical Notes

- `gstep_lib.splice_vertical_strip` clusters strip columns around `~23 deg` tilt and assumes `w <= h` with median tilt `<= 45 deg`. Transposed (`"T"`) bricks have `theta -> 90 - theta` (`~67 deg`), so `vstrip_lib.orient_for_vstrip` normalizes both `w <= h` and `median(deg) <= 45` (via `flipx`) prior to strip detection/splicing.
- Combining synthesized `vstrip` bricks (`n852_9x18_vs_1` `f=12/17=0.705882353`, `n852_11x21_peel_top` `f=0.717965456`, `n852_10x21_g7_vs_0` `f=0.700568`, `gs_11x22_vs_3` `f=0.688383`) into `zero` and `1x1` corner-melted pinwheels + `lpslack` + `pslp` yields strict improvements across `K = 28, 29, 30`.

## Pipeline

1. Brick synthesis and compression:
   ```bash
   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
     /tmp/highrpeel_venv/bin/python poc/swarm_vstrip852/poc/run_synth.py
   ```
2. Pinwheel + `lpslack` + `pslp` scale sweep:
   ```bash
   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
     /tmp/highrpeel_venv/bin/python poc/swarm_vstrip852/poc/run_scale.py --workers 2
   ```
3. Dual-mode certification (`literal` + `pandas`) and CSV export:
   ```bash
   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
     /tmp/highrpeel_venv/bin/python poc/swarm_vstrip852/poc/export_final.py
   ```

## Dual-Certified Wins (`literal == pandas` vs `origin/main` `0e74cfc`)

| `n` | `(K, D)` | Incumbent `L` (`origin/main`) | New `L` (`literal == pandas`) | `dL` | Method |
|---:|:---:|---:|---:|---:|:---|
| `796` | `(28, 12)` | `28.705882375792043` | `28.705882352968883` | `-2.282316e-08` | `zero` pinwheel `9x18(g=+6) + 18x9(g=+6)` |
| `854` | `(29, 13)` | `29.719475060115124` | `29.717965456364590` | `-1.509604e-03` | `1x1` corner melt `9x18(g=+6) + 20x11(g=+7)` + `pslp` |
| `912` | `(30, 12)` | `30.692576840848356` | `30.692567874989000` | `-8.965859e-06` | `1x1` corner melt `9x21(g=+6) + 21x9(g=+6)` + `pslp` |
| `913` | `(30, 13)` | `30.706062291589736` | `30.705882352970884` | `-1.799386e-04` | `zero` pinwheel `9x18(g=+6) + 21x11(g=+7)` |

## Verification

- `BASE_SHA=$(git rev-parse origin/main) HEAD_SHA=$(git rev-parse HEAD) /tmp/highrpeel_venv/bin/python .github/scripts/check_final_csv.py`
- `/tmp/highrpeel_venv/bin/python -m pytest -v poc/swarm_vstrip852/tests/test_vstrip852.py poc/auto_union/poc/test_union.py .github/scripts/test_check_final_csv.py`
