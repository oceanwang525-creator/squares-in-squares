# Hypothesis H77 (`swarm_b7xplant`): Diagonal-Mode `brick_mechwalk` (`d(i±j)`) on `g = +6..+9` Sub-Bricks + Zero-Overlap Pinwheels (`r = +13..+17`)

## Summary

Applying standalone SVD + diagonal-mode `brick_mechwalk` (`d(i±j)`) + `compress_brick_slp` to `g = +6..+9` sub-bricks extracted from the portfolio (`origin/main`), followed by zero-overlap corner pinwheel assembly + `pslp.squeeze` / `joint_xytheta_slp_tight` + `lpslack`, produced certified wins across PR1 and PR2:

### PR1 (Merged in `origin/main`)
- **`n=914` (`m=30, r=+14`)**: `30.7312110434865026991` -> `30.7311162686141160200` (`dL = -9.477487e-05`, `B8_12x20_n914_cut + B6_9x18_n854_cut_T` zero-overlap pinwheel; subsequently superseded in `origin/main` `0e74cfc` at `30.7297542998426915`, so dropped cleanly from PR2 `final.csv`).

### PR2 (Against `origin/main` `0e74cfc`)
- **`n=915` (`m=30, r=+15`)**: `30.7403339912685678904` (`merged.csv`) -> **`30.7403302241520865409`** (**`dL = -3.767116e-06`**) via `B6_7x18_n915` (`7×18, +6`, diagonal-walked from `0.740334` to `0.738152`, `df = -2.182e-03`) + `B9_22x12_n915` (`22×12, +9`) zero-overlap pinwheel (`id_flipy_direct`) + `pslp.squeeze` + `lpslack`.
- **`n=800` (`m=28, r=+16`)**: `28.7745781046842140953` (`merged.csv`) -> **`28.7745781003578347045`** (**`dL = -4.326379e-09`**) via `B8_10x18_NEW_T + B8_18x9_n800_T` zero-overlap pinwheel (`rot180_rot180_direct`) + `lpslack`.

## Key Diagonal-Mode Sub-Brick Compressions (`poc/swarm_b7xplant/data/brick_*.json`)

| Brick | `(w×h, +g)` | `f0` | `f` after `d(i±j)` walk | `df` |
|:---|---:|---:|---:|---:|
| `B8_12x20_win914` | `12×20, +8` | `0.744848` | `0.731217` | `-1.363e-02` |
| `B8_19x20_n798_pRR` | `17×20, +8` | `0.740330` | `0.731455` | `-8.875e-03` |
| `B7_20x9_n977` | `20×9, +7` | `0.745619` | `0.740594` | `-5.025e-03` |
| `ext_brick_B8_10x22_from23_top` | `10×22, +8` | `0.768782` | `0.766309` | `-2.473e-03` |
| `B6_7x18_n915` | `7×18, +6` | `0.740334` | `0.738152` | `-2.182e-03` |
| `B7_9x19_n916` | `9×19, +7` | `0.752190` | `0.750374` | `-1.816e-03` |
| `B9_21x11_n916` | `21×11, +9` | `0.752161` | `0.750957` | `-1.204e-03` |

## Scale-Up Coverage & Remaining Bottlenecks (`{743, 798, 855, 800, 856, 915, 976, 916, 977}`)

- **`n=855` (`29, +14`)**: `B6_8x18_n855_cut (0.732065) + B8_20x11_n855 (0.733980)` tied at `+3.35e-10`; `B8_20x11_n855` stayed locked (`df=0`) under `d(i±j)`.
- **`n=976` (`31, +15`)**: `B7_11x19_n975_cut (0.721540) + B8_20x11_n855 (0.733980)` tied at `+3.59e-10`.
- **`n=916, 977, 856, 743, 798`**: Best walked zero-overlap pairs stalled `+4.75e-03 .. +2.23e-02` above `origin/main`.

## Structure & Reproduction

- `poc/swarm_b7xplant/poc/b7_lib.py` — Portfolio loader, corner-cut extractor, `standalone_brick_walk_diag` (`d(i±j)`), `polish_and_certify_b7`, dual certification.
- `poc/swarm_b7xplant/poc/run_gate_fast.py`, `run_scale.py`, `run_scale2.py`, `run_fast_pslp.py` — Gate and scale-up runners.
- `poc/swarm_b7xplant/final.csv` — Dual-certified PR2 winners (`800, 915`).
- `poc/swarm_b7xplant/tests/test_b7xplant.py` — Unit and `final.csv` dual-certification tests.

Run verification:
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /tmp/highrpeel_venv/bin/pytest poc/swarm_b7xplant/tests/test_b7xplant.py
```
