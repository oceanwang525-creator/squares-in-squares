# H122 (`watchxplant`): Cross-Donor `public_watch_nov14` + New Champions Sub-Brick Extraction, Peeling, `vstrip` Splicing & Pinwheel + Masked Melt + `pslp`

Extracts tilted corner sub-bricks (`g = +3..+12`) from `poc/public_watch_nov14/final.csv` (`n >= 232`) and newly landed champions, synthesizes 1-row/1-col peeled and `vstrip`/`hcol`-spliced variants compressed with `compress_brick_slp` + rectangular `pslp.squeeze`, pools them with the repo's brick catalog (`1,751` oriented bricks in v2), and runs `kx, ky in {-1, 0, +1}` pinwheels with masked C++ corner/seam melt, joint SLP, `lpslack`, and `pslp` kink-safe squeeze.

## Results Summary

### PR1 (Merged in PR #111)
Dual-certified (`literal == pandas` via `poc/auto_union/poc/union.py`) against `origin/main` (`b3357b7`), preserved in `origin/main:merged.csv`:

| `n` | `m, r` | Incumbent (`origin/main` `merged.csv`) | `watchxplant` `L` | `dL` | Mechanism |
|---:|:---:|---:|---:|---:|---|
| **908** | `30, +8` | `30.63031083364909632402707168752530735225970891185782` | **`30.62780775044685437491462450980854358972279806958093`** | **`-2.503083e-03`** | `pw_791_TL_11x19_g4_pr_slp` (`10x19+4`, `f=0.629860`) + `pw_791_TL_11x20_g4_T` (`20x11+4`, `f=0.627636`), `kx=0, ky=0`, `flipy_id`, masked C++ melt + `pslp` + `lpslack` + `pslp` |

### PR2 (vs `origin/main` `fb533ce`)
After harvesting newly merged PR #111–#116 champions (`107` donor packings, `7,738` raw bricks) and peeling `n=908` (`ch_908_TL_10x20_g4_pb_slp`, `10x19+4`, `f=0.625880`):

| `n` | `m, r` | Incumbent (`origin/main` `merged.csv`) | `watchxplant` `L` | `dL` | Mechanism |
|---:|:---:|---:|---:|---:|---|
| **849** | `29, +8` | `29.63688171337221016935754856335454179598630809094671` | **`29.63590815332729009935501256550294580618467311401071`** | **`-9.735600e-04`** | `ch_908_TL_10x20_g4_pb_slp` (`10x19+4`) + `_T` (`19x10+4`), `kx=0, ky=0`, `id_rot180`, masked C++ melt + `lpslack` + `pslp` |
| **973** | `31, +12` | `31.67330922358059414916429773007088008702474697984690` | **`31.67328783156610012202743536135885409596771605150165`** | **`-2.139201e-05`** | `p_n972_10x21_g6` (`10x21+6`) + `_T` (`21x10+6`), `kx=0, ky=0`, `rot180_flipy`, masked C++ melt + `pslp` |

## What Worked vs. Exhausted (H122 Verdict)
- **WORKED**: Cross-donor sub-brick extraction + 1-row/1-col peeling + `pslp` brick compression (`ch_908_TL_10x20_g4_pb_slp` won `n=849` by `-9.736e-04`; `pw_791_TL_11x19_g4_pr_slp` won `n=908` by `-2.503e-03`).
- **EXHAUSTED**: Across the 21 highest-headroom non-`(1,1)` targets in `m = 21..31, r = +6..+18` (`docs/scale_summary_v2.json`), the remaining 19 targets (`911, 793, 584, 858, 691, 583, 638, 859, 692, 975, 795, 633, 540, 800, 639, 457, 854, 743, 747`) yielded `0` additional `>= 1e-3` wins (corner/seam geometry did not clear `pen=0` below existing `origin/main` champions). Retiring H122 after PR2.

## Reproduction

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /mnt/e/squares-in-squares/.venv/bin/python -m pytest poc/swarm_watchxplant/tests/test_watchxplant.py -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /mnt/e/squares-in-squares/.venv/bin/python poc/swarm_watchxplant/poc/harvest_and_synth.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /mnt/e/squares-in-squares/.venv/bin/python poc/swarm_watchxplant/poc/run_gate.py --top-k 3 --fast --time-limit 2400
```
