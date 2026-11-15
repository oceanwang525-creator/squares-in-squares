# H122 (`watchxplant`): Cross-Donor `public_watch_nov14` + New Champions Sub-Brick Extraction, Peeling, `vstrip` Splicing & Pinwheel + Masked Melt + `pslp`

Extracts tilted corner sub-bricks (`g = +3..+12`) from `poc/public_watch_nov14/final.csv` (`n >= 232`) and newly landed champions (`n = 973, 795, 793, 974, 743, 914, 912, 975, 854, 863, 924, 585, 634`), synthesizes 1-row/1-col peeled and `vstrip`/`hcol`-spliced variants compressed with `compress_brick_slp` + rectangular `pslp.squeeze`, pools them with the repo's brick catalog (`1,527` oriented bricks total), and runs `kx, ky in {-1, 0, +1}` pinwheels with masked C++ corner/seam melt, `joint_xytheta_slp_tight`, `lpslack`, and `pslp` kink-safe squeeze.

## Gate Result (PR1)

Dual-certified (`literal == pandas` via `poc/auto_union/poc/union.py`) against `origin/main` (`b3357b7`):

| `n` | `m, r` | Incumbent (`origin/main` `merged.csv`) | `watchxplant` `L` | `dL` | Mechanism |
|---:|:---:|---:|---:|---:|---|
| **908** | `30, +8` | `30.63031083364909632402707168752530735225970891185782` | **`30.62780775044685437491462450980854358972279806958093`** | **`-2.503083e-03`** | `pw_791_TL_11x19_g4_pr_slp` (`10x19+4`, `f=0.629860`) + `pw_791_TL_11x20_g4_T` (`20x11+4`, `f=0.627636`), `kx=0, ky=0`, `flipy_id`, masked C++ melt + `pslp` + `lpslack` + `pslp` |

## Reproduction

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /mnt/e/squares-in-squares/.venv/bin/python -m pytest poc/swarm_watchxplant/tests/test_watchxplant.py -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /mnt/e/squares-in-squares/.venv/bin/python poc/swarm_watchxplant/poc/harvest_and_synth.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /mnt/e/squares-in-squares/.venv/bin/python poc/swarm_watchxplant/poc/run_gate.py --top-k 4 --time-limit 1500
```
