# H106 (`vstrip492`): Periodic-Strip Splicing (`vstrip`), Outer Peeling & Pinwheel Composition from `n=492/852/911/975`

## Method

1. **Sub-brick extraction (`extract_donor_subbricks`)**: Extracts tilted `(w, h, g)` corner bricks from `n=492` (`22.701529`, e.g. `5x13 g=+3`, `9x16 g=+5`), `n=852` (`9x17 g=+4`, `10x18 g=+5`, `11x20 g=+6`), `n=911` (`10x18 g=+4`, `10x19 g=+5`, `10x21 g=+6`), and `n=975` (`9x21 g=+6`, `9x22 g=+7`).
2. **Periodic-strip splicing & peeling (`run_synth_bricks.py`)**: Synthesizes 93 relaxed bricks in `poc/swarm_vstrip492/data/brick_*.json` via `vstrip` insertion (`g -> g+1`), `vstrip` removal (`remove_vertical_strip`, `g -> g-1`), `hcol` splicing, and 1-row/1-column outer peeling (`peel_outer_unit_layer`), compressing each with `compress_brick_slp` and `brick_mechwalk`.
3. **Pinwheel Assembly & Polishing (`run_gate_probe.py`, `run_scale.py`)**: Composes zero-overlap (`ov=0, ct=0`) and corner/seam-melted (`max_ov=1`) pinwheels across `K = 20..31, D = 7..16`, polishing with `joint_xytheta_slp_tight`, `lpslack`, and `mechwalk_compress`.

## Certified PR1 Wins (vs `origin/main` `0e74cfc`, `literal == pandas`)

| `n` | `(m, r)` | Incumbent `L` (source) | New `L` | `dL` | `d(L^2/n)` | Brick Pair & Method |
|---:|:---:|---:|---:|---:|---:|:---|
| **585** | `(24, +9)` | `24.70447095241959160778` (`merged.csv`) | **`24.70129798695283559115`** | **`-3.172965e-03`** | `-2.679706e-04` | `5x13(g=+3, n=492)` + `19x10(g=+6, n=852 peel_bottom)` + `lpslack` + `mechwalk` |
| **686** | `(26, +10)` | `26.69671117474591783374` (`merged.csv`) | **`26.69261044029394645228`** | **`-4.100734e-03`** | `-3.191479e-04` | `16x9(g=+5, n=492)` + `9x17(g=+5, n=852 peel_top)` + `lpslack` + `mechwalk` |
| **795** | `(28, +11)` | `28.69977978646164907569` (`swarm_peelstep`) | **`28.69912621005546539351`** | **`-6.535764e-04`** | `-4.718814e-05` | `11x20(g=+6, n=852 peel_top)` + `9x16(g=+5, n=492)` + `lpslack` + `mechwalk` |
| **796** | `(28, +12)` | `28.70588237579204346739` (`swarm_lpslack`) | **`28.70588237115237810416`** | **`-4.639665e-09`** | `-3.346374e-10` | `9x18(g=+6)` + `18x9(g=+6)` + `lpslack` + `mechwalk` |

**Total across 4 strict wins**: `d(L^2/n) = -6.343069e-04`.

## Reproduction & Verification

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest poc/swarm_vstrip492/tests/test_vstrip492.py -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python poc/swarm_vstrip492/poc/run_synth_bricks.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python poc/swarm_vstrip492/poc/run_scale.py
BASE_SHA=$(git merge-base origin/main HEAD) HEAD_SHA=$(git rev-parse HEAD) python .github/scripts/check_final_csv.py
```
