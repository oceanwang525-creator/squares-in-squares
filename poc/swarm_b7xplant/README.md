# Hypothesis H77 (`swarm_b7xplant`): Diagonal-Mode `brick_mechwalk` (`d(i±j)`) on `g = +6..+9` Sub-Bricks + Zero-Overlap Pinwheels (`r = +13..+17`)

## Summary

Applying standalone SVD + diagonal-mode `brick_mechwalk` (`d(i±j)`) + `compress_brick_slp` to `g = +6..+9` sub-bricks extracted from the portfolio (`origin/main`), followed by zero-overlap corner pinwheel assembly + `pslp.squeeze` / `joint_xytheta_slp_tight` + `lpslack`, produced certified wins across PR1, PR2, and PR3:

### PR1 (Merged in `origin/main`)
- **`n=914` (`m=30, r=+14`)**: `30.7312110434865026991` -> `30.7311162686141160200` (`dL = -9.477487e-05`, `B8_12x20_n914_cut + B6_9x18_n854_cut_T` zero-overlap pinwheel; subsequently superseded in `origin/main` `0e74cfc` at `30.7297542998426915`).

### PR2 (Merged in `origin/main` `99c95bd`)
- **`n=915` (`m=30, r=+15`)**: `30.7403339912685678904` -> `30.7403302241520865409` (`dL = -3.767116e-06`) via `B6_7x18_n915` + `B9_22x12_n915` (`id_flipy_direct`) + `pslp.squeeze` + `lpslack` (preserved/improved in `origin/main` `merged.csv` at `30.7403227283026149572`).
- **`n=800` (`m=28, r=+16`)**: `28.7745781046842140953` -> **`28.7745781003578347045`** (`dL = -4.326379e-09`) via `B8_10x18_NEW_T + B8_18x9_n800_T` (`rot180_rot180_direct`) + `lpslack` (preserved identically in `origin/main` `merged.csv`).

### PR3 (Against `origin/main` `b3ac0d7`)
- **`n=975` (`m=31, r=+14`)**: `31.7158414438057201438` (`merged.csv`) -> **`31.7158234301261719324`** (**`dL = -1.801368e-05`**) via `vs492_9x22_n975_g7` (`9×22, +7, f=0.715802275`) + `vs492_9x22_n975_g7_T` (`22×9, +7`) zero-overlap pinwheel (`id_rot180_sf0.0_fhi_si200`) + `pslp.squeeze` + `lpslack`.
- Dropped PR2 groups `[800, 915]` are preserved/improved in `origin/main:merged.csv`.

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

## Big-Move Gate Outcome (Per Board `#717` Course Correction)

- **Outcome**: **WORKED** for `-1e-6 .. -1e-4` diagonal-walked sub-brick polish (`n=914` `-9.48e-5`, `n=915` `-3.77e-6`, `n=800` `-4.33e-9`, `n=975` `-1.80e-5`), **FAILED** the `>= 1e-3` Big-Move single-`n` / family rebuild gate in Round 3.
- **Geometric Root Cause**:
  1. Dominant sub-bricks on the `r = +14..+17` bottleneck (`B8_20x11_n855` at `0.733980`, `B7_19x11_n854` at `0.721540`, `B9_21x12_n915` at `0.740334`, `B7_8x19_n916` at `0.752190`) already sit at zero-slack locked local minima (`df = 0` under `compress_brick_slp` + `d(i±j)` SVD walks).
  2. Consequently, `n=855` and `n=976` stall at `dL ~ +3.4e-10` against `0.733980`, while `n=916, 977, 856, 743, 798` stall `+4.75e-03 .. +2.23e-02` above `origin/main`, requiring structural pocket/corridor rebuilds rather than single-brick diagonal walks.
- **Status**: Packaging PR3 (`n=975`) and retiring `b7xplant` per `#717`.

## Structure & Reproduction

- `poc/swarm_b7xplant/poc/b7_lib.py` — Portfolio loader, corner-cut extractor, `standalone_brick_walk_diag` (`d(i±j)`), `polish_and_certify_b7`, dual certification.
- `poc/swarm_b7xplant/poc/run_gate_fast.py`, `run_scale.py`, `run_scale2.py`, `run_fast_pslp.py`, `run_scale3.py`, `run_scale3b.py` — Gate and scale-up runners.
- `poc/swarm_b7xplant/final.csv` — Dual-certified PR3 winner (`975`).
- `poc/swarm_b7xplant/tests/test_b7xplant.py` — Unit and `final.csv` dual-certification tests.

Run verification:
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /tmp/highrpeel_venv/bin/pytest poc/swarm_b7xplant/tests/test_b7xplant.py
```
