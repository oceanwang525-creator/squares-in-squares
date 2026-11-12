# `poc/swarm_angles`: Non-45° Geometry Behind $D=1$ Roots ($n = k^2 + 1$, $k = 20..31$)

## Target Scope & Incumbents
Disjoint target $n$ (`D=1` roots $k = 20..31$):
- Initial focus: `401, 442, 530`
- Extended pool: `485, 577, 626, 677, 730, 785, 842, 901, 962`

Observed frontier curve in `poc/frontier_nov12/docs/per_n_gap.csv` vs current `merged.csv`:
- `k <= 19` (`n <= 362`): tied at `f = L - k = 0.535534` ($45^\circ$).
- `k = 20` (`401`): `20.535534` vs frontier `20.526135` (`gap = 0.009399`)
- `k = 21` (`442`): `21.535534` vs frontier `21.522923` (`gap = 0.012611`)
- `k = 22` (`485`): `22.535534` vs frontier `22.522903` (essentially `k=21` + upright L-border)
- `k = 23` (`530`): `23.535534` vs frontier `23.517305` (`gap = 0.018229`)
- `k = 24` (`577`): `24.535534` vs frontier `24.517305` (`k=23` + upright L-border)
- `k = 25` (`626`): `25.520815` (`17x17` $45^\circ$ diamond + 4 upright staircases) vs frontier `25.511049`
- `k = 26..28` (`677, 730, 785`): `f_front = 0.510279, 0.509315, 0.505935`
- `k = 29..31` (`842, 901, 962`): `f_front = 0.502153` (`k=29` + upright L-borders)

Structural clues from existing `merged.csv` non-$45^\circ$ packings:
- `n=50` (`7.571429 = 7 + 4/7`): $4 \times 4$ / two-cluster `36.87°` (3-4-5) core + 4 upright staircases (`6+11+6+11`).
- `n=123, 146, 227, 258` ($D=2$): diagonal band of squares tilted at `40.97° -> 39.65° -> 33.88° -> 33.77°` (`w=3` columns sheared transversely along $v_{45}$ inside a $45^\circ$ upright staircase channel).
- `n=626` (`25.520815`): $17 \times 17$ $45^\circ$ central diamond (`289`) + 4 upright staircases (`337`).
- `n=908` (`30.634071`): two decoupled tapered sheared pockets (`~24.5°`) embedded inside the upright grid with an intact upright diagonal corridor.

---

## Hypothesis Log (Logged Before Testing)

### Idea 1: Tilted Central Block (`m1 x m2` / Octagonal Core at $\theta \ne 45^\circ$) + 4 Corner-Anchored Upright Staircases
- **Hypothesis**: Just as `n=626` uses a $17 \times 17$ $45^\circ$ core + 4 upright corner staircases (`L = 25.520815`) and `n=50` uses a `36.87°` core + 4 corner staircases (`L = 7.571429`), rotating a central `m1 x m2` core (or slightly chamfered diamond / 2-block symmetric pair like `n=37`) to an angle $\theta \in (0, 45^\circ)$ allows a larger core `m` or tighter corner packing to beat `f = 0.535534` at `n = 401, 442, 530` (and `0.520815` at `626`).
- **Cheap test (<5 min)**: Exact geometric capacity scan over `(m1, m2, theta, cx, cy)` with 4 corner-anchored upright lattices (resolved by non-overlapping corner assignment + MIS / greedy conflict resolution) at target `L <= L_ours - 0.002` for `n = 401, 442, 530` (and `626` sanity calibration).
- **Result (NEGATIVE, `poc/idea1_tilted_core.py`, `docs/idea1_tilted_core.json`)**:
  - Calibration `n=626` (`17x17` @ $45^\circ$, `L=25.52081528`, `shift=(0.5, 0.0)`) reproduced `total=626`.
  - At `L_probe = L_ours - 0.002` on `n = 401, 442, 530, 626`, both single `m1 x m2` tilted blocks and symmetric 2-block cores + 4 corner upright fill maxed out at $k^2$ (`400, 441, 529, 625`, upright $5^\circ$ fallback; deficit $= 1$).

### Idea 2: Sheared Off-$45^\circ$ Diagonal Band ($\theta = 45^\circ - \delta$, Generalized `n=123/227/258`)
- **Hypothesis**: In `n=227` and `n=258`, columns of $w$ squares tilted at $\theta = 45^\circ - \delta$ ($\delta \approx 11.2^\circ$) shear transversely into the $45^\circ$ staircase grooves, achieving band width below $w$ with longitudinal pitch $\sec(\delta)$. At $k \ge 20$, wider sheared bands ($w \in [2..15]$) or partial/corrugated non-$45^\circ$ diagonal bands may reduce $L$ below $k + 0.535534$.
- **Cheap test (<5 min)**: Parametric scan over `(w, a1, a2, b1, b2, delta)` + continuous L-BFGS / LP relaxation on `n = 401, 442, 530`.
- **Result (NEGATIVE for rigid collinear `w`-columns, `poc/idea2_sheared_diag.py`, `docs/idea2_sheared_diag.json`)**:
  - First scan (`th=15..45°`, `b1 >= 1`) and widened scan (`th=1..45°`, `b1 >= 0`) both capped at $k^2 - 1$ (`399, 440, 528`, deficit $= 2$) at `L_probe = L_ours - 0.002`, with best found config collapsing to `th_deg=1.0`, `w=2`, `b1=0, b2=0`.
  - Note: Calibration on `n=227` (`a=12,11 b=2,2 w=3 @ 33.88°`) also only reached `201/227` with rigid perpendicular `(-s, c)` columns at step `u += 0.85`, meaning `n=227` actually uses sheared/tilted sub-columns or individually packed diagonal teeth rather than rigid orthogonal `w`-columns. Worth inspecting `n=227` and `n=908` coordinates directly before designing Idea 3.

### Idea 3: Embedded Tapered Sheared Pockets ($C_2$ / $C_4$ / Asymmetric Frame, Generalized `n=908` + Tapered Ends)
- **Hypothesis**: Unlike uniform wall-sheared L-pockets (which failed on $D=1$ in `hydrogen_brainstorm` with deficit 7 due to end-triangle waste $\frac{1}{2}W^2\tan\theta$), embedding 2 symmetric or 4 $C_4$-chiral tapered pockets (`theta in 15°..42°`) inside an upright grid (or along walls with tapered/upright-transition ends, or 4-edge pinched frame with $f_1 + f_2 = 1 + f$) avoids end-triangle waste and may unlock a bounded-width saving at $k \ge 20$.
- **Cheap test (<5 min)**: Construct & count tapered embedded pockets (NW/SE or 4-fold $C_4$ around central core / upright cross) + L-BFGS relaxation on `n = 401, 442, 530`.
- **Result (NEGATIVE for rectangular sub-pockets & rigid `dy=sec th` cols, `poc/idea3_tapered_pockets.py`, `docs/idea3_tapered_pockets.json`)**:
  - Sub-rectangle reduction test (`(a + 0.524) x (h + 0.524)` trying to pack `a*h + 1` across `a=2..12, h<=20`): `0` feasible (`best_pen = 0.0227` for `2x20+1 = 41` in `2.524 x 20.524`). Proves flat-walled sub-rectangles cannot beat `a*h` at `f=0.524`.
  - Rectangular `NW/SE` cavities with `dy = sec th, dx = cos th` columns: `cap = 389 (def=12)` on `401`, `430 (def=12)` on `442`, `518 (def=12)` on `530`. Flat upright walls (`sw_w x sw_h`) create tooth waste.

### Idea 4: De Novo Oriented-Domain Inflation ($a: 0.85 \to 1.00$) on `n=401`
- **Hypothesis**: Because `n=401` has only 401 squares and a large gap (`0.0094`, `20.5355 -> 20.5261`), de novo inflation from $a_0 = 0.85 \to 1.00$ inside fixed $L \in [20.5265, 20.5345]$ using diverse oriented priors ($C_4$ chiral swirl, 2-domain tilted, diagonal shear, upright frame + tilted interior) can directly discover the non-$45^\circ$ topology without assuming its exact discrete coordinates.
- **Cheap test (<10 min)**: Vectorized Numba SAT penalty + L-BFGS inflation across 7 CPU workers (`OMP_NUM_THREADS=1`) on `n=401`.
- **Result (INITIAL RUN NEGATIVE, `poc/idea4_inflation.py`, `docs/idea4_inflation.json`)**:
  - 42 seeds (21 free + 21 $C_4$ `1+4*100`) at `L_probe = 20.5315` completed in `1.3s` because `make_seed` initialized on a uniform `20x20` grid (`+1` at center) without `(0,0)/(L,0)/(0,L)/(L,L)` corner-aligned quadrant offsets or shake moves at `a < 1`, causing immediate `20x20` lattice lock-in. Best penalty was `6.28e-4` (`inc_shake`), best non-incumbent `0.0158` (`swirl_frame`).

### Idea 5: $D=1$ Multi-Offset Corner Lattices + Non-$45^\circ$ Sheared Central/Pocket Cavity & Pinched 4-Edge Frame
- **Hypothesis**: Inspecting `n=50` ($7^2+1$, `L = 7 + 4/7`, $36.87^\circ$ sheared central cluster of 16 + 4 corner upright staircases `6,11,6,11`), `n=37` ($6^2+1$, chevron-bent `47.9°/45°/42.1°` 15-square band + 4 corners), `n=626`, and `n=908` (`24.5°` vertically-aligned sheared pockets `dy = sec th, dx = cos th` inside offset `(0,0)/(L,L)` upright grids) reveals two missing mechanisms in Ideas 1–4:
  1. **Corner-aligned 4-quadrant offset (`A=(0,0), B=(L,L), M_NW=(0,L), M_SE=(L,0)`)** creates an internal sheared/stepped cavity (diagonal, cross, or two `NW/SE` pockets) where tilted squares sit at angle $\theta \ne 45^\circ$ with `n=50`-style sheared columns or `n=908`-style `dy = sec th` columns;
  2. **4-Edge Pinched Frame (`f_N + f_S = 1 + f`, `POC 3` tapered ends)** around an upright core splits the `0.5 + f` slack across opposite edges to embed tapered tilted strips (`W >= 1.5`) with reduced end-triangle waste;
  3. **Multi-offset inflation (`Idea 4b`)**: Seeding inflation (`a: 0.85 -> 1.00`) from 4-corner-aligned partial grids (`< k^2` upright with diagonal/corner/edge notches filled by tilted seeds + `a=0.88` shake/re-orient) avoids `20x20` grid jamming;
  4. **Sloped `(1:2 / 1:3)` stepped `NW/SE` boundaries** (true `n=908` boundary matching) + **$C_4$ `(2m+1)^2 + 4Q = 401` core-and-4-pocket** inflation (`m=5..8`) + **chevron-bending of `n=401` `45°` pockets**.
- **Cheap test (<8 min)**: Build & optimize parameterized `Idea 5` (`poc/idea5_sheared_cluster.py`) on `n = 401` (then `442, 530` on hits) with cavity-first then joint L-BFGS + `fixed_angle_lp_tighten`.
- **Result (NEGATIVE for local carved-cavity L-BFGS inflation, `poc/idea5_sheared_cluster.py`, `docs/idea5_sheared_cluster.json`)**:
  - 120 carved `(sx, sy)` 4-quadrant offset cavity jobs (`diamond`, `diag_band`, `nw_se_pockets`), 24 `C_4` `(2m+1)^2 + 4Q` jobs, and 24 `n=401` pocket-expansion jobs ran in `2.9s` at `L_probe = 20.5315`.
  - Best non-incumbent `5A` penalty was `0.0160` (`diamond 49`, `(sx,sy)=(9,11)`); best `5D` was `0.0352` (`19^2 + 4*10`); `5B` (`drop_k=0`) compressed `n=401` to `pen = 3.71e-4`. Continuous L-BFGS inflation alone gets jammed inside rigid integer cavities without discrete placement/angle topology.

### Idea 6: Chamfered Non-$45^\circ$ Lattice Core / Asymmetric Multi-Block + Exact MILP 4-Corner Upright Fill
- **Hypothesis**: Inspecting `idea1_tilted_core.py` and `n=401`'s exact upright partition (`SW=100, NW=74, SE=77, NE=100`, where `NW` and `SE` extend into `(0..18)` pushing `NE` to `(3..15, 3..15)`) revealed that Idea 1's symmetric round-robin `d=i+j` corner filler could not even recover `n=401` (`5x5 + 5x5 @ 45°`) itself, and Idea 1 never tested **chamfered/octagonal tilted cores** (like `n=598`'s `400` tilted squares @ `~18°` with chamfered corners + upright corners) or **sheared `n=50`/`n=227` multi-block / rotated clusters** paired with globally optimal 4-corner upright selection.
- **Cheap test (<8 min)**: Use `scipy.optimize.milp` (exact 4-corner Maximum Independent Set on upright grid sites clearing any tilted set `T`) — first calibrating on `n=401`, `n=50`, `n=598`, `n=626` — then scan chamfered rotated lattices (`theta in 5°..44°`, including `30°` / `arctan(3/7)` / `arctan(3/4)` / `arctan(1/2)`), chamfered diamonds, and asymmetric 2-block tilted cores on `n = 401, 442, 530` + L-BFGS/LP polish.

### Idea 7: $n=123/146/227/327$-Calibrated Jogging Sheared Diagonal & Off-Diagonal Bands + End-Angle Taper + MILP Corner Fill
- **Hypothesis**: Inspecting `n=123, 146, 227, 258, 327` in `merged.csv` revealed why Idea 2 failed calibration (`201/227` on `n=227`): the real sheared bands (a) jog in $v_{45}$ column-by-column to interlock with `NW`/`SE` upright staircase teeth (`n=123` uses alternating `4,3,4,3` column lengths; `n=227` uses `a1=13, a2=12` with `v45` jogging and `50.18°/55.76°` bent end columns), and (b) can sit off-diagonal (`a1 != a2`, e.g., `n=327` embeds a shifted `n=146`-band + smaller corner band). Furthermore, `per_n_gap.csv` shows the $D=2$ sheared-band curve dropping continuously to `k=24..30` (`0.5348 -> 0.5141`) in tandem with the $D=1$ curve (`k=20..29`: `0.5261 -> 0.5022`).
- **Cheap test (<5 min)**: First calibrate the $v_{45}$-jogging, alternating `(w, w-1)` column packer + `milp_upright_fill` + `relax_lbfgs` on `n=227` (`L=15.563626`) until it reaches `>= 226..227`, then scan `(a1, a2, w, th)` (both diagonal `a1 ~ a2` and off-diagonal `a1 != a2`, `th in 15°..45°`) + L-BFGS + LP on `n = 401, 442, 530`.

### Idea 6 Results (Chamfered Cores & Multi-Blocks + Exact MILP, 2026-11-12)
- `milp_upright_fill` calibrated cleanly on `merged.csv` (`n=50 -> 50`, `n=401 -> 402` at `20.535534`, `n=626 -> 626`).
- Fast-proxy (`fast_upright_count` -> top 18 MILP at `min_gap=-0.002`) on `n=401, 442, 530` at `L_probe = L_ours - 0.002`:
  - `6A` (chamfered core): hit `401/401, 442/442, 530/530` ONLY with `45°` `U=V=2.5` (`n_tilt=25`) exploiting the `-0.002` screening tolerance; L-BFGS+LP failed to clear `L_probe` (`rpen = 1.8e-4..2.3e-4, lp_ok=False`), confirming `45°` `5x5` cannot shrink below `0.535534`.
  - `6B` (`four_c4`): `4` blocks of `2x2` at `atan(1/3)=18.435°` around `(rx=5.5, ry=1.5)` reached `cap=441/442` (`def=1`) and `529/530` (`def=1`) with `-0.002` tolerance (still needs tight `min_gap >= 0` & wider `C4` pocket scan).

### Idea 7 Initial Result + Idea 7B (Sheared Jogging Diagonal Band — Calibrating to 227/227)
- Initial Idea 7 (`pack_jogging_band` with rigid column normals `shear=0`, `v=median`, `u_cur += 0.88`, `du<=0.45`) calibrated only `nt=45/54` (`tot=220/227`) on `n=227` and defaulted to `45°` (`def=5..6` on `401, 442, 530`).
- **Diagnostics on `n=227` (`L=15.563626`)**: `NW=78, SE=91` (`a1=13, a2=12`), `SW=3, NE=1`, `nt=54` (`18` cols of `w=3`, bent ends `-39.82° / -34.24°`, interior `-33.88°`) with: (a) **intra-column normal shear** (`-0.02..-0.05` along `e2`), (b) **periodic 3/4-col `v`-jogging** (`+0.075` vs `-0.193`) interlocking with staircase teeth, and (c) `u45` spanning `2.48..20.20` (`du ~ 0.98..1.06`).
- **Idea 7B**: Update `pack_jogging_band` to search `(u_lo, shear, v_jog_mode)` + end-column bend + exact `min_gap=0` MILP so `calibrate_227()` reaches `227/227` (`nt >= 52`), then re-scan `401, 442, 530` (including `w=1,2,3,4` and shifted `(a1, a2)` up to `k`).

### Idea 8 (Planned Before Testing: Bounded-Width Non-45° Wall/Corner/Pinched-Frame Pockets & 30° Bicrystal)
- Motivated by `merged.csv` `n=598` (`L=24.903160`: `18.9°` & `71.1°` bounded-width `dy = sec th` wall pockets pinching `NE=301`, `SW=44`), `n=908` (`L=30.634071`: `24.5°` stepped `NW/SE` `101`-sq pockets inside upright border), and `per_n_gap.csv` (`k=20..31` dropping `0.526135 -> 0.502153` in step with `D=2, D=3`, with `k=23,24` matching `6*sqrt(3)-9.875 = 0.51730485` at `30°`):
  - **8A (`n=598 / 908` Wall & Corner `sec th` Pockets + Exact MILP)**: Test 1 or 2 `w`-column non-45° pockets (`th = 10°..42°`, `dy = sec th, dx = cos th` + slight shear, clipped to `[0, L]` or a stepped `1:2` / `1:3` corner boundary) along ONE/TWO walls or NW/SE pockets + exact `milp_upright_fill(min_gap=0)`!
  - **8B (Expanded `C4` & Chiral Pinched Frame + `30°` `±30°` / `60°` Gadgets)**: Scan `C4` 4-block / 4-pocket `w x h` strips (`th = 12°..42°`) pinching an inner upright/tilted core, plus `±30°` symmetric chevrons/cores, strictly checked at `min_gap >= 0` + `relax_lbfgs` + `fixed_angle_lp_tighten`.
