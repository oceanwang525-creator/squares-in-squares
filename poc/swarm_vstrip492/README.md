# H106 (`vstrip492`): Periodic-Strip Splicing (`vstrip`), Outer Peeling & Pinwheel Composition from `n=492/852/911/975`

## Method

1. **Sub-brick extraction (`extract_donor_subbricks`)**: Extracts tilted `(w, h, g)` corner bricks from `n=492` (`22.701529`, e.g. `5x13 g=+3`, `9x16 g=+5`), `n=852` (`9x17 g=+4`, `10x18 g=+5`, `11x20 g=+6`), `n=911` (`10x18 g=+4`, `10x19 g=+5`, `10x21 g=+6`), and `n=975` (`9x21 g=+6`, `9x22 g=+7`).
2. **Periodic-strip splicing & peeling (`run_synth_bricks.py`)**: Synthesizes 93 relaxed bricks in `poc/swarm_vstrip492/data/brick_*.json` via `vstrip` insertion (`g -> g+1`), `vstrip` removal (`remove_vertical_strip`, `g -> g-1`), `hcol` splicing, and 1-row/1-column outer peeling (`peel_outer_unit_layer`), compressing each with `compress_brick_slp` and `brick_mechwalk`.
3. **Pinwheel Assembly & Polishing (`run_gate_probe.py`, `run_scale.py`, `run_phase2.py`)**: Composes zero-overlap (`ov=0, ct=0`) and corner/seam-melted (`max_ov=1`) pinwheels across `K = 20..31, D = 7..16`, polishing with `joint_xytheta_slp_tight`, `lpslack`, `pslp` (`squeeze` + `shake_squeeze`), and `mechwalk_compress`.

## Certified PR4 Big-Move Win (vs `origin/main` `fb533ce`, `literal == pandas`)

| `n` | `(m, r)` | Incumbent `L` (`merged.csv`) | New `L` | `dL` | `d(L^2/n)` | Brick Pair & Method |
|---:|:---:|---:|---:|---:|---:|:---|
| **973** | `(31, +12)` | `31.67330922358059414916` | **`31.67202707235603316343`** | **`-1.282151e-03`** | `-8.347205e-05` | `10x21(g=+6, n=972)` + `21x10(g=+6, n=972)` (`ov=0, ct=1` corner melt) + `lpslack` + `pslp` |

**Total across 1 strict PR4 win**: `d(L^2/n) = -8.347205e-05`, `d(L/sqrt(n)) = -4.110388e-05` (PR3 wins `{538, 800, 849, 970}` are staged separately in PR #128 on `origin/swarm-vstrip492-pr3`; PR1/PR2 wins `{585, 686, 739, 795, 796, 852}` are preserved/improved in `origin/main` `merged.csv`).

### Corner-Melt Findings (`run_phase2.py`, `docs/phase2_results.json`)
- **WORKED**: `ov=0, ct=1` single-corner touch (`w1 + k2 = m + 1`) with high-density `10x21 g=+6` (`n=972`) sub-brick + C++ penalty melt + `lpslack` + `pslp` broke `n=973 (31,+12)` by **`-1.282151e-03`**.
- **FAILED / High-cost**: `ov=1` edge-overlap melts (`n=584, 638, 692, 738`) stalled at `dL = +0.0206..+0.0385` due to full-row/col boundary collisions; `df >= -1e-4` `zero_ct0` micro-polish was frozen per `#717`/`#725`.

## Certified PR3 Wins (vs `origin/main` `10a15e8`, `literal == pandas`)

| `n` | `(m, r)` | Incumbent `L` (`merged.csv`) | New `L` | `dL` | `d(L^2/n)` | Brick Pair & Method |
|---:|:---:|---:|---:|---:|---:|:---|
| **800** | `(28, +16)` | `28.77457810035783470455` | **`28.77450261627168216271`** | **`-7.548409e-05`** | `-5.430050e-06` | `18x9(n=800)` + `10x18(NEW)` + `lpslack` + `pslp` |
| **849** | `(29, +8)` | `29.63688171337221016936` | **`29.63688154086398901216`** | **`-1.725082e-07`** | `-1.204383e-08` | `12x16(n=849)` + `12x16(n=849)` + `lpslack` + `pslp` |
| **970** | `(31, +9)` | `31.63768686499829017102` | **`31.63768686184327810742`** | **`-3.155012e-09`** | `-2.058088e-10` | `16x11(g=+4, n=490)` + `20x11(n=970s6)` + `lpslack` + `pslp` |
| **538** | `(23, +9)` | `23.70588235542379963937` | **`23.70588235377292491020`** | **`-1.650875e-09`** | `-1.454849e-10` | `5x13(n=492)` + `9x18(g=+6, n=538)` + `lpslack` + `pslp` |

**Total across 4 strict PR3 wins**: `d(L^2/n) = -5.442445e-06` (prior PR1/PR2 wins `{585, 686, 739, 795, 796, 852}` are preserved/improved in `origin/main` `merged.csv`).

## Certified PR2 Wins (vs `origin/main` `ccdac8f`, `literal == pandas`)

| `n` | `(m, r)` | Incumbent `L` (`merged.csv`) | New `L` | `dL` | `d(L^2/n)` | Brick Pair & Method |
|---:|:---:|---:|---:|---:|---:|:---|
| **795** | `(28, +11)` | `28.69912621005546637760` | **`28.69887184145230235083`** | **`-2.543686e-04`** | `-1.836509e-05` | `11x20(g=+6, n=852 peel_top)` + `9x16(g=+5, n=492)` + `pslp` + `lpslack` |
| **686** | `(26, +10)` | `26.69261044029394836792` | **`26.69260621000717565734`** | **`-4.230287e-06`** | `-3.292052e-07` | `16x9(g=+5, n=492)` + `9x17(g=+5, n=852 peel_top)` + `pslp` + `lpslack` |
| **852** | `(29, +11)` | `29.68414360939792026625` | **`29.68414360669388105569`** | **`-2.704039e-09`** | `-1.884204e-10` | `11x20(g=+6, n=852 peel_top)` + `17x9(g=+5, n=852)` + `lpslack` + `pslp` |
| **739** | `(27, +10)` | `27.68414360894269309823` | **`27.68414360686820275581`** | **`-2.074490e-09`** | `-1.554276e-10` | `10x17(g=+5)` + `17x9(g=+5)` + `lpslack` + `pslp` |

**Total across 4 strict wins**: `d(L^2/n) = -1.869464e-05` (PR1 wins `n=585` and `n=796` are preserved/improved in `origin/main:merged.csv`).

## Certified PR1 Wins (Merged in PR #104)

- `n=585`: `24.7044709524` -> **`24.7012979870`** (`dL = -3.172965e-03`)
- `n=686`: `26.6967111747` -> **`26.6926104403`** (`dL = -4.100734e-03`)
- `n=795`: `28.6997797865` -> **`28.6991262101`** (`dL = -6.535764e-04`)
- `n=796`: `28.7058823758` -> **`28.7058823712`** (`dL = -4.639665e-09`)

## Reproduction & Verification

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest poc/swarm_vstrip492/tests/test_vstrip492.py -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python poc/swarm_vstrip492/poc/run_synth_bricks.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python poc/swarm_vstrip492/poc/run_phase2.py
BASE_SHA=$(git merge-base origin/main HEAD) HEAD_SHA=$(git rev-parse HEAD) python .github/scripts/check_final_csv.py
```
