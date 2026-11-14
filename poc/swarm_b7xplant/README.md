# Hypothesis H77 (`swarm_b7xplant`): Diagonal-Mode `brick_mechwalk` (`d(i±j)`) on `g = +6, +7, +8` Sub-Bricks + Zero-Overlap Pinwheels (`r = +13..+17`)

## Summary

Applying standalone SVD + diagonal-mode `brick_mechwalk` (`d(i±j)`) + `compress_brick_slp` to `g = +6, +7, +8` sub-bricks extracted from the portfolio (`origin/main`), followed by zero-overlap corner pinwheel assembly + `joint_xytheta_slp_tight` + `lpslack`, produces a **certified win on `n=914`** (`dL = -9.477487e-05` vs `origin/main` `f542119`):

- **`n=914` (`m=30, r=+14`)**: `L = 30.73111626861411601998466698695337722793776985081244` vs incumbent `30.7312110434865026991` (`dL = -9.477487239e-05`), certified in BOTH `literal` and `pandas` modes via `poc/auto_union/poc/union.py`.

### Why `n=914` Won
- `B6_9x18_n854_cut` (`9×18, +6`, extracted from `n=854` and walked with `d(i±j)` SVD modes + `compress_brick_slp`) compressed from `0.705985624` to `0.705904362` (`-2.53e-2` thinner than `n=914`'s original `18×9` sub-brick at `0.731213720`).
- Re-assembling `B8_12x20_n914_cut` (`12×20, +8`, top-left) with transposed `B6_9x18_n854_cut_T` (`18×9, +6`, bottom-right) in a zero-overlap pinwheel (`12 + 18 = 30`, `20 + 9 = 29`) + `joint_xytheta_slp_tight` + `lpslack.optimize_group` relaxed the bottleneck across the `B8_12x20` sub-brick from `30.731211043` down to `30.731116269`.

## Gate Results (`{854, 914, 975}`)

Report saved in `poc/swarm_b7xplant/docs/gate_report.json`:

| `n` | `(m, r)` | Incumbent (`origin/main` `f542119`) | Best Candidate `L` | `dL` | Status |
|---:|:---|---:|---:|---:|:---|
| `854` | `(29, +13)` | `29.72154000327154` | `29.72154000322275` | `-4.88e-11` | Below `-1e-9` win threshold |
| `914` | `(30, +14)` | `30.73121104348650` | `30.73111626861412` | **`-9.477e-05`** | **CERTIFIED WIN** |
| `975` | `(31, +14)` | `31.71877369634442` | `31.73111801805375` | `+2.34e-03` | No win (donor `12×20 + 18×9` at `0.7311 > 0.7188`) |

## Structure & Reproduction

- `poc/swarm_b7xplant/poc/b7_lib.py` — Portfolio loader, corner-cut extractor, `standalone_brick_walk_diag` (SVD + `d(i±j)` diagonal modes + `compress_brick_slp`), `polish_and_certify_b7`, dual certification (`literal` + `pandas`).
- `poc/swarm_b7xplant/poc/run_gate_fast.py` — 30-minute gate runner on `{854, 914, 975}`.
- `poc/swarm_b7xplant/final.csv` — Certified candidate CSV (`id,x,y,deg`).
- `poc/swarm_b7xplant/tests/test_b7xplant.py` — Unit and `final.csv` dual-certification tests.

Run verification:
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /tmp/highrpeel_venv/bin/pytest poc/swarm_b7xplant/tests/test_b7xplant.py
```
