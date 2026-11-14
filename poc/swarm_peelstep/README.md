# H109 (`peelstep`): Unified Cross-Donor Pinwheel & Masked C++ Melt

Pools sub-bricks from `cleanpeel2`, `stepdown`, `gstep`, `band3`, plus newly extracted and 1-row/1-column peeled+compressed (`compress_brick_slp` + `brick_mechwalk`) sub-bricks from `win_n974` (`31.700187`), `win_n912` (`30.692988` / `30.692586`), and `n=852` (`29.688527`), and sweeps 2-brick pinwheels across `kx, ky in {-1, 0, +1}` with masked C++ corner/seam melt, `joint_xytheta_slp_tight`, and `lpslack`.

## Gate Result (PR1)

| `n` | `m, r` | Incumbent (`origin/main`) | `peelstep` `L` | `dL` | Mechanism |
|---:|:---:|---:|---:|---:|---|
| **913** | `30, +13` | `30.707106781206181` (`merged.csv`) | **`30.706062294652124`** | **`-1.044487e-03`** | `B6_9x20_n853_BRT` (`band3`, `9x20+6`, `f=0.705882`) + `B7_10x22_vstrip_T` (`gstep`, `22x10+7`, `f=0.688507`), `kx=+1, ky=0`, `flipy_rot180`, masked C++ corner melt + SLP + `lpslack` |

## Reproduction

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/highrpeel_venv/bin/python -m pytest poc/swarm_peelstep/tests/test_peelstep.py -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/highrpeel_venv/bin/python poc/swarm_peelstep/poc/run_gate.py
```
