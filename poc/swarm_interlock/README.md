# H35: Interlocking Corner-Shared Sub-Brick Pinwheel + Seam Melt (`poc/swarm_interlock/`)

## Mechanism
When placing two rectangular sub-bricks `b1` (`w1 x h1`, excess `g1`, slack `f1`) at `TL` and `b2` (`w2 x h2`, excess `g2`, slack `f2`) at `BR` inside `[0, m + f]^2` with `g1 + g2 = r = n - m^2`:
- Previous disjoint placement (`H17`) required `w1 + w2 <= m` or `h1 + h2 <= m`.
- H35 allows corner overlap `kx = w1 + w2 - m >= 0`, `ky = h1 + h2 - m >= 0` (`kx + ky >= 1`), deduplicating `|I| = kx * ky` inner-corner `1x1` axis-aligned unit squares (`drop_from in {b1, b2, regrid}`), filling the complementary `BL` and `TR` unit blocks with optional `shift_delta = 0.5 * |f2 - f1|`, and running `run_cpp_masked_melt` around `I` + `relax_lbfgs` + `joint_xytheta_slp_tight` + `fixed_angle_lp_tighten`.

## Certified Win (`poc/swarm_interlock/final.csv`)
Exported in `poc/swarm_interlock/final.csv` (`literal == pandas` verified by `poc/auto_union/poc/union.py`):

| `n` | `m, r` | Pair (`kx, ky`) | `origin/main` (`b67cd44`) `L_inc` | H35 `L_lit = L_pd` | `dL` vs `b67cd44` |
|---:|---|---|---:|---:|---:|
| **851** | `29, +10` | `B4_9x19 + B6_21x10` (`kx=1, ky=0`) | `29.6841436063593355` | **`29.6661920494731863`** | **`-1.795156e-02`** |

Progression on `n=851`:
1. Gate (`run_gate.py`): `29.684143606 -> 29.675415917322384` (`dL = -8.727689e-03`) via `B4_9x19+B6_21x10_s03_b1_center_umelt_1`.
2. Post-sweep polish (`polish_and_retry.py`): `contact_jacobian` + `soft_modes` mechwalk (`29.675416 -> 29.667933`) followed by targeted unit-seam `run_cpp_masked_melt` (`-> 29.6661920494731863`, `p851_9`).

## 15-Minute Gate Results (`poc/swarm_interlock/docs/gate_report.json`)
- **`n = 851`**: **WIN** (`L = 29.675415917`, polished to `29.666192049`).
- **`n = 795`**: `OK` (`L = 28.806110466`, `dL = +0.102707`; note `B6_10x21 + B5_17x9` has `w1 + w2 = 27 < 28`, i.e., `kx = -1`).
- **`n = 684`**: `OK` (`L = 26.645957945`, `dL = +0.003388` via `B4_11x16+B4_16x10` `kx=0, ky=1`; higher-effort retry `dL = +0.016802`).

## Scale-Up Sweep (`poc/swarm_interlock/docs/scale.log`, `docs/scale_ckpt.json`)
Completed 12 targets in `2783.4s` before stopping automatically at `12` consecutive non-wins per protocol:
- `n = 794`: `dL = +0.008838` (`L = 28.692981948`, `B4_9x19+B6_21x10`, `kx=2, ky=1`; higher-effort retry `dL = +0.026512`)
- `n = 852`: `dL = +0.029762` (`L = 29.728892497`, `B5_7x22+B6_24x8`, `kx=2, ky=1`)
- `n = 633`: `dL = +0.030327` (`L = 25.687180906`, `B4_9x19+B4_16x10`, `kx=0, ky=4`)
- `n = 585`: `dL = +0.041887` (`L = 24.747792059`)
- `n = 537`: `dL = +0.051271` (`L = 23.743659174`)
- `n = 584`: `dL = +0.051857` (`L = 24.729526117`)
- `n = 739`: `dL = +0.060507` (`L = 27.744650593`)
- `n = 741`: `dL = +0.064903` (`L = 27.792824836`)
- `n = 688`: `dL = +0.092703` (`L = 26.820625111`)
- `n = 685`: `dL = +0.261372` (`L = 26.945515201`)
- `n = 634`: `dL = +0.273620` (`L = 25.962005953`)
- `n = 738`: `dL = +0.285270` (`L = 27.963917372`)

### Geometric Bottleneck Observed
When `kx + ky = 1` (`|I| = 0`, 1-axis overhang as in `n=851`), unit-seam melting successfully relaxes into a deep strict win. When `kx >= 1` and `ky >= 1` (`|I| >= 1`) or `kx, ky >= 3`, local masked SA around `I` consistently stalled at `feas=0/24` because relieving `O(1)` unit-corner overhang requires global phase rearrangement across the adjacent `1x1` grid rows/columns.
