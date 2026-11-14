# H97 (`stepdown`) — Brick `g -> g-1` Step-Down & Pinwheel Re-Assembly

**Branch**: `swarm-stepdown` / `swarm-stepdown-pr1`  
**Status**: **1st Big-Move Certified Win (`n=912`, `-5.948e-3`)**; scaling up step-down bricks across `K=27..30`.

## Certified PR1 Result (vs `origin/main` `cf838aa`)

| `n` | `(m, r)` | Incumbent (`origin/main`) | `stepdown` (`literal == pandas`) | `dL` | Method |
|---:|:---:|---:|---:|---:|---|
| **912** | `(30, +12)` | `30.698936322874302400...` (`merged.csv`) | **`30.692987901774137644...`** | **`-5.948421e-03`** | `9x21 (g=+6)` + `21x9 (g=+6)` `id_rot180` pinwheel + C++ seam melt (`a_start=0.90`) + `joint_xytheta_slp` + `mechwalk` + `lpslack` |

## Artifacts

- `poc/stepdown_lib.py` — Portfolio loader, D4 symmetry + pinwheel assembly, `stepdown_brick` (`g -> g-1` rectangular L-BFGS + `joint_xytheta_slp_rect`), global `slp_and_polish` (`joint_xytheta_slp` + `fixed_angle_lp_tighten` + `mechwalk` + `lpslack`), and `literal == pandas` + `kaggle_roundtrip` dual certification.
- `poc/run_gate.py` — 30-minute gate script (direct `(m, r+1) -> (m, r)` deletion + walldrift/melt + brick `g -> g-1` step-down).
- `data/brick_B5_5x15_n744_g4.json`, `brick_B6_24x8_g5.json`, `brick_B7_17x8_g6.json`, `brick_B8_10x18_NEW_g7.json`, `brick_B8_18x8_g7.json` — Step-down sub-bricks (`g -> g-1` from `b8xplant` donors).
- `data/win_n912.json`, `final.csv` — Certified `n=912` packing.
- `tests/test_stepdown.py` — Unit and certification tests for `stepdown_lib` and `final.csv`.

## Reproducing & Testing

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --project poc/baseline/poc --with scipy --with numba --with pytest \
  pytest poc/swarm_stepdown/tests/test_stepdown.py -v

python3 .github/scripts/check_final_csv.py poc/swarm_stepdown/final.csv
```
