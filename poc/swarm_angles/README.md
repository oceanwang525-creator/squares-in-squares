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

### Idea 7B & Idea 8 Results (C++/OpenMP Fast Search, `poc/search_angles.cpp`, `docs/cpp_search_7b_8.json`)
- Calibrated `Grid4C::solve_mis_exact` on `merged.csv`: `n=598 -> 595/595`, `n=626 -> 626/626`, `n=401 -> 391/401` (`up=341/351` — missing `10` because `n=401` uses an interlocking **pinwheel** `NW/SE/SW/NE` border-strip partition rather than `max(i,j)` corner blocks!). `pack_jogging_band` calibrated to `42/54` (`214/227`) on `n=227`.
- Screened `n=401, 442, 530` at `L_probe = L_ours - 0.001` in `20.7s` total (`OMP_NUM_THREADS=7`):
  - `n=401` (`L_probe=20.534534`): `best_cap=400 (def=1)` [`7B a=(20,18) w=1 th=30 sh=-0.04 nt=19 exact_up=381`] (`5.6s`)
  - `n=442` (`L_probe=21.534534`): `best_cap=441 (def=1)` [`7B a=(21,19) w=1 th=36.87 sh=0 nt=20 exact_up=421`] (`6.4s`)
  - `n=530` (`L_probe=23.534534`): `best_cap=529 (def=1)` [`7B a=(21,23) w=1 th=56.12 sh=-0.04 nt=22 exact_up=507`] (`8.7s`)
- None of `8A/8B/8C` beat `k^2` at `L_probe`, and `Idea 7B` hit the `k^2` ceiling (`def=1`) on all three. Notice `n=364` (`L=19+4/7`, `3-4-5`) and `n=404` (`L=20.591378`, `40.974°`) in `merged.csv` use a diagonal sheared band separating big `SW+NE` triangles (`114+165`) + thin `NW/SE` border strips, and `n=401` uses a pinwheel upright border that `solve_mis_exact` undercounted by `10`.

### Idea 9 (Planned Before Testing: Pinwheel/Tabu MIS + Small-Angle/Pinched-Frame + Spatial-Hash SA Shrinker)
- **Hypothesis**: Three concrete gaps explain why Ideas 6–8 stopped at `k^2` (`def=1`):
  1. **9A (Pinwheel + Tabu MIS for `Grid4C`)**: Adding pinwheel border-strip orders (`ci` vs `cj` CW/CCW + wall strips) and Tabu local search (`1-for-1` plateau + `k`-kick + greedy refill) so `ExactMIS` hits `401/401` on `n=401` (`351/351` upright), then re-testing `Idea 7B` (expanded to `n=364/404`-type sheared bands + `n=227` exact reproduction) & `Idea 8` top candidates (`def=1` -> `def=0`?).
  2. **9B (Small-Angle `1°..20°` Rotated Rectangular Cores + Chiral 4-Edge Pinched Upright Frame)**: Test `(U x V)` tilted rectangular cores at small/medium angles (`th = 2°..20°`), shifted/pinched 4-corner frames (`(dx, dy)` offset upright corner blocks squeezing 4 edge pockets/strips along walls), and off-diagonal `1`-to-`3` corner non-45° pockets (`n=442, 530` plateaus).
  3. **9C (Unbiased C++/OpenMP Spatial-Hash SA + Adam `L`-Shrinker)**: Initialize from `merged.csv` (`n=401`), perturbed `k^2` grids with `+1` inserted square, and top `def=1` seeds (+1 square in largest gap) shrunk to `[L_front, L_ours - 0.002]`, running fast `O(1)` spatial-hash SA with single-square moves + **`2x2..5x5` block rotation/shear moves** (`th` free in `0°..45°`) + **Adam & L-BFGS `L`-shrinkage** (`L_pen <= 1e-12`) to discover non-lattice pockets/cores without template bias.
- **Cheap test (<5 min)**: Compile `poc/idea9_sa_and_frames.cpp` (`-O3 -fopenmp`, max 7 threads) and run on `n = 401, 442, 530`.
- **Results (`docs/idea9_results.json`, 18.3s total)**:
  - **9A (`Grid4C` Pinwheel+Tabu MIS)**: Calibrated `n=401` -> **`tilt=50, up=352, tot=402`** (+1 over `n=401` itself at `min_gap=-1e-6`), `n=598 -> 595/595`, `n=626 -> 626/626`!
  - **9B & 9C**: Even with exact pinwheel MIS (`9B`) and 42-worker multi-start spatial-hash SA (`9C`), `401/442/530` peaked at `def=1` (`9B_rect` / `9B_pkt`: `400, 441, 529`), and SA reached `min_sa_E = 7.16e-5, 9.42e-5, 1.76e-4` without hitting `pen == 0` below `L_ours`.

### Idea 9D (Planned Before Testing: Wide `P = 8..k-1` Sheared Slabs + Tapered Sawtooth-End SA + Asymmetric 4-Edge Frame)
- **Hypothesis**: Earlier `Idea 8A` (`w <= 10`) and `9B-3` (`w <= 4`) NEVER tested wide sheared slabs **`P = 8..k-1` (`u = P + 1` columns at `th ~ 18°..26°` in width `W = P + f`)** or **interior `K_x x K_y` upright cores** pinched by 4 sheared strips! Yet direct calculation shows `P = 12, u = 13` (`th = 20.65°`) on `n = 530` (`L_front = 23.5173`) gives `273 + 253 = 526` (`def = 4`!) EVEN WITH RAW CLIPPED SAWTOOTH ENDS (`bot_gaps` / `top_gaps ~ [0.43, 0.78, 0.06]`), where `22` squares/column only exceed `23.5173` by `0.215` (less than the `0.288` height reduction from flattening top/bottom square angles toward `0°`!). Similarly, `n=401` (`P=11: def=5`) and `n=842` (`P=12: 347 vs 349` in slab!) start within `4–5` squares before filling/tapering the end gaps or coupling to an L-pocket (`K_x x K_y` upright + two sheared arms) or 4-edge pinched frame (`W_S, W_E, W_N, W_W`).
- **Cheap test (<3 min)**:
  1. Seed `SpatialSA + Adam` directly from wide `P = 9..k-2` sheared slabs (`+` upright complement), inserting missing `def` squares into deepest `bot_gaps`/`top_gaps` sawtooth voids inside `[0, W] x [0, L]` (`W = P + f`) and relaxing end rows (`20° -> 0°`) at `[L_front, L_ours - 0.001]`.
  2. Re-test **2-block tilted pairs** (symmetric & off-diagonal `(m1 x m2)` at `th = 18°..45°` including sheared `n=50/227` patches, modeled on `n=401`'s two-diamond topology) and **chamfered octagonal cores (`U x V` minus `1..3` corner tips)** using the calibrated **`9A Pinwheel+Tabu Grid4C` MIS** (`402/401` exact), plus test dropping `1` square from `9A`'s `402`-square `n=401` solution (`L_probe` polish).
- **Results (`poc/swarm_angles/docs/idea9d_results.json`, `task-600`, `20.4s` on 7 threads)**:
  - `9D-1`: `Grid4C` recovered **402 squares** (`50` tilted + `352` upright) at `ov=0.0` (note `-1e-6` placement tolerance inside `Grid4C` had no violation after `polish`), polishing to exact `L = 20.535533906`; dropping any single square `0..401` and polishing stayed at `20.535533906` (`0` improvement).
  - `9D-2`: `min_slab_E = 0.148` (`401`), `0.109` (`442`), `0.254` (`530`) — wide slabs + SA end-fillers did not converge to zero overlap.
  - `9D-3`: `9D_MIS` best caps `397/401` (`5x5 c=2`), `438/442` (`7x7 c=1`), `526/530` (`12x9 c=1`) — no win `< L_ours`.

### Idea 10 (Planned Before Testing: `a`-Homotopy Liquid-to-Jamming Inflation, `n=971` Oblique-Cut Reflected Wall Pockets, Bent Bands + Sliding Borders, & `n=626/842` Testing)
- **Diagnostic clues from inspecting `merged.csv` + `per_n_gap.csv`**:
  - `n=626` in `merged.csv` already achieves `L = 25.520815280171 = (27 + 17*sqrt(2))/2` using a `17 x 17` (`289`) `45°` diamond + `337` upright squares using half-offset `(f, f/2)` (`78+78+90+90+1`) — proving `Grid4C` (`{0, f}^2`) missed `(f/2)`!
  - Meanwhile `per_n_gap.csv` `D=1` (`0.526135 -> 0.502153`) drops at `k=20 (14*sqrt(2) = 19.799)`, `k=21 (15*sqrt(2) = 21.213)`, `k=23 (16*sqrt(2) = 22.627)`, `k=25 (17*sqrt(2) = 24.042)`, saturating at `k=29..31 (0.502153)`, while `n=17, 37, 364, 404, 908, 971` all use **sliding non-lattice border strips (`33..219` fractional modes)**, **curved/bent bands** (`n=37: 42.09°-45°-47.91°`), or **oblique-cut wall pockets reflected across `x=y` (`n=971`: `25.2°` along bottom + `64.8°` along left, `ux >= u0` slanted boundary)**.
- **Four targeted attacks in `poc/idea10_homotopy_and_pockets.cpp` (`-O3 -fopenmp`, max 7 threads)**:
  - **10A (`a`-Homotopy Inflation `a: 0.91 -> 1.00` + `n=626` Perturbation)**: Unlike `9C` (stuck in `95%`-density jamming at `a=1.0`), inflate `n = 401, 442, 530, 626, 842` from `a = 0.91..0.96 -> 1.00` inside `L_target in {L_front + 0.002, L_ours - 0.001}` using multi-domain/bent/perturbed-`n=626`/perturbed-`n=401` seeds, spatial-hash MC + cluster rotations + row/col slides + Adam polish per `a` stage.
  - **10B (Bent/Jogging Diagonal Bands + `9A` MIS + Sliding Border Polish)**: Re-run `Idea 7B` (which hit `def=1` even with broken MIS) using calibrated `9A Pinwheel+Tabu MIS`, end-bend `dth in {0, +-3°, +-6°}` (`n=37/227/364/404` style), and sliding border row relaxation.
  - **10C (`n=971` Oblique-Cut Reflected `th & 90°-th` Wall Pockets & Chiral 4-Pocket Frame)**: Test `(x, y, th) <-> (y, x, 90°-th)` and `(L-y, L-x, 90°-th)` slanted-cut wall strips (`W_p = 1.5..7.5`, `th = 14°..43°`) + 4-arm C4 chiral wall pockets around `(k-a) x (k-a)` core, filled with `Grid9` (`{0, f/2, f}^2`) + Tabu MIS + sliding strip refill.
  - **10D (`n=626, 842` Non-45° & `m=14..21` Tilted/Octagonal Cores on `Grid9`)**: Check `n=626` (`25.520815 -> 25.511049`) and `n=842` (`29.520815 -> 29.502153`) where `k=25, 29` unlock larger cavities (`m=17..21` at `30°..45°`).
- **Results (`poc/swarm_angles/docs/idea10_results.json`, `task-694`, `14.8s` on 7 threads)**:
  - Calibration exact: `L-prop` (`n=626 -> 842`) verified `pen=6.66e-16` at `L=29.520815280`, and `Grid9` recovered exact `n=626` (`289 tilted + 337 upright = 626`).
  - `10A` (`75k` SA sweeps/stage) over-melted initial lattices (`min_10A_E = 2.41e-3..7.78e-3`) without recrystallizing; `10B/10C/10D` `Grid9` MIS peaked at `400/401`, `440/442`, `528/530`, `624/626`, `840/842` (`def = 1..2`) without `< L_base`. Drop tests (`task-713`: `403->401`, `532->530`, `404->401`) also stayed rigid (`20.5649`, `23.5624`, `20.5914`).

### Idea 11 (Planned Before Testing: Frozen-Core Cavity `a`-Homotopy, Soft-Interface `Grid9` + `Adam` Relaxation, `n=847` `45°`+Pocket Hybrids & Clipped `971` Pockets, and DP Jogging Band)
- **Diagnostic findings from `merged.csv` (`401, 403, 532, 626, 847, 908, 971`) & why `10A/B/C/D` stalled**:
  - Inspecting `merged.csv` in radians->degrees revealed three concrete structural gaps in Idea 10:
    1. **`n=847` (`29.6066`) hybrid**: combines an off-center `15x15` (`224 @ 45°`) block in `[7.4..27.2] x [1.4..21.2]` WITH **one small `14`-square `37.4°` pocket** in the upper-left corner `[1.55..4.02] x [23.46..28.17]` inside an upright outer frame. Idea 10 never tested `45°` core + non-`45°` pocket hybrids!
    2. **`n=971` (`31.6643`) & `n=908` (`30.6341`)**: the `25.2°`/`64.8°` reflected pockets are clipped on **BOTH** ends (`x in [10.70..28.35], y in [0.67..7.04]` in `971`) and optionally inset behind an upright border (`[2.66..11.02] x [15.91..28.31]` in `908`), PLUS rely on `24..266` slightly shifted upright boundary squares (`n=626` itself has `294` `(fx,fy)` upright modes!). Strict `min_gap = 0.0` `Grid9` MIS drops those shifted interface squares (`def = 1..2`).
    3. **`10A` global melting**: annealing all `n = 401..842` squares at `T0 = 0.04` for `75k` sweeps turned upright lattices into amorphous glass (`~90` moves/square). Freezing an upright crystal complement and annealing ONLY `60..240` cavity/corridor squares gives `30x–100x` more moves per active square without losing crystalline long-range order.
- **Three targeted attacks in `poc/idea11_cavity_homotopy.cpp` (`-O3 -fopenmp`, max 7 threads) on `n in {401, 626, 842, 442, 530}`**:
  - **11A (Frozen-Core Cavity / L-Corridor `a`-Homotopy + `n=626/401` Cavity Melt)**: Freeze known upright sub-lattices (`k^2 - W^2` corner complement, `(k-W)x(k-W)` or `(k-W)xk` frame complement, `SW+NE` triangles around diagonal band, or `n=626/842/401` upright frames / inner diamond) and inflate ONLY `60..240` active squares from `a = 0.88 -> 1.00` over `400k–800k` cavity-only sweeps inside `L_probe = L_base - 0.001` & `L_fprobe = L_front + 0.002`, then **unfreeze ALL `n` squares** for joint `Adam` (`freeze_angles=false`) + `polish_and_shrink`.
  - **11B (Soft-Interface `min_gap in {0, -0.06, -0.09}` `Grid9` MIS + Joint `Adam` & Per-Family Diagnostics)**: Test dual-clipped/inset `971/908` pockets, `n=847`-style `45°` diamond + small `20°..43°` corner/wall pocket(s), and chamfered `U x V` cores (`10..21`). Solve `Grid9` MIS at both `0.0` and soft `min_gap in {-0.06, -0.09}`; whenever `cap >= n`, run joint `Adam` + `polish_and_shrink` to recover true zero-overlap equilibrium. Track best `cap` per family separately.
  - **11C (Calibrated Perpendicular DP Jogging Band)**: Fix perpendicular band offset, calibrate against `n=403 @ 20.564936` (`75` tilted at `~34.6°`) and `n=532 @ 23.562385` (`87` at `~33.0°`), and scan DP per-column `v`-jogging bands (`w = 1..7`, `th = 16°..45°`, end-bend `{0, +-3°}`) with soft `Grid9` MIS + `Adam`.
- **Results (`poc/swarm_angles/docs/idea11_results.json`, `task-766`, `32.0s` on 7 threads) & Critical MIS Calibration Discovery**:
  - `11A_cav` reached `min_E = 3.59e-6 .. 4.13e-6` only on `cav_seed` (frozen shrunk `base_seed`), masking individual cavity family energies; `11B_core` peaked `400/401, 440/442, 528/530, 621/626, 837/842`; `11C_dp` only reached `401/403` on `403` calibration and `cap0 < n-5` skipped soft `min_gap_tilt` on `11B_971/847/11C_dp`!
  - **Root-cause calibration test (`task-803` + `check_grid9_calib`)**: Even when handed the **exact** tilted squares of `merged.csv`, our heuristic `Grid9::solve_mis_tabu` ONLY recovered **`392/401`** on `n=401` (all `351` upright squares WERE in `Grid9`, yet heuristic MIS missed `10`!) whereas exact **HiGHS MILP clique-MIS** (`[xv, xv+1) x [yv, yv+1)` box cliques) solved exact `Grid9` MIS in `0.16s` getting **`402/401`**, **`403/403`** on `n=403`, **`846/847`** on `n=847`, **`904/908`** on `n=908`, and **`961/971`** on `n=971`. Moreover, inspecting the missing upright squares of `n=971` revealed exact `sec(25.2°) = 1.1052`-spaced boundary columns (`x=8.1643`), proving why fixed `Grid9` alone cannot score tilted pockets without interface sliding/fine points.

### Idea 12 (Planned Before Testing: Pure Cavity Capacity & Boundary-Buffered Homotopy Probes + Deep `n=401` SA, and Exact HiGHS Clique-MIS with Interface Sliding Points)
- **Two clean, orthogonal tests addressing both root causes**:
  - **12A (`poc/idea12_cavity_probes.cpp`, `-O3 -fopenmp`, 7 threads)**:
    1. **Pure geometric capacity tests** (zero-overlap feasibility via `a`-homotopy `0.86 -> 1.00`, `400k-600k` moves/stage):
       - **1-edge rectangular slabs** `(W + f) x (k + f)` (`k=20`, `f=0.5261`, `W = 2..8`, holding `W*k + 1` squares) — determines directly if ANY rectangular edge strip beats lattice density.
       - **L-rim corridors** `[0, L]^2 \ [W+f, L]^2` (`W = 2..6`, holding `2Wk - W^2 + 1` squares) and **4-edge pinched frames** `[0, L]^2 \ [W+0.5*f, L-W-0.5*f]^2` (`W = 1..4`, holding `4Wk - 4W^2 + 1` squares), BOTH with **hard inner boundaries** AND with **2-layer unfrozen upright inner buffers** (allowing `sec(th)` sliding!).
    2. **Unmasked per-family cavity homotopy & deep `n=401` global homotopy**: Run `1.6M` sweeps/stage on `n=401` (`L=20.5345`, `L=20.5300`) and buffer-unlocked cavities on `401, 626, 842`, reporting EVERY cavity type's own `min_E` separately.
  - **12B (`poc/idea12_highs_search.py`, 7 `spawn` workers using exact `scipy.optimize.milp` clique-cover + NumPy/C++ `Adam` polish)**:
    - Augment `Grid9` near tilted regions (`dist < 3.0`) with fine `{0, 0.25f, 0.5f, 0.75f, f}` + `sec(th)` wall-aligned points (exact clique constraints cover ALL candidate pairs in `0.2-0.4s`/solve), and evaluate across `401, 626, 842` at `L_probe = L_base - 0.001`:
      - Asymmetric `1`-block / `2`-block / `3`-block / `4`-block non-`45°` & `45°` diamond sets (with C4 / D2 / D1 symmetry),
      - `n=847`-style `45°` core + non-`45°` pocket(s),
      - True oblique-cut `n=971/908` 1/2/4-pocket chiral wall frames (`u >= u_cut`),
      - Soft `min_gap_tilt = -0.07` + `Adam` + L-shrink whenever `cap >= n`.
- **12A results (`task-822`, `30.6s`, 7 threads) — negative**: every probe stayed at `min_E ~ 1e-2` (slabs `W=2..8`: `1.004e-2..1.021e-2`; L-rim hard/buffered `W=3..5`: `9.86e-3..2.98e-2`; 4-frame `W=2,3`: `7.68e-3..9.87e-3`; diag buffer `w=3..5`: `9.91e-3..1.99e-2`; deep global `n=401` (28 workers): `9.85e-3`, no win). The near-constant `~1e-2` floor is about one square's worth of penetration spread over the cavity, i.e. SA does not find the extra square; caveat: the `nth -> 0` alignment move (25% of snaps) biases toward the upright lattice, so this is weak evidence, not a capacity proof.
- **12B results (`poc/swarm_angles/docs/idea12b_results.json`, `task-969`, `360.1s`, 7 workers) — 0 wins (`def = 4..10`):**
  - Note: adding `time_limit = 0.55s` (`1.5s` for `aug0`/`soft8`) prevented multi-minute branch-and-bound hangs, though on large open `n=626/842` grids `0.55s` returned suboptimal MILP bounds (`971`: `507/626` with `nt=1`; `dp` `842`: `776 -> 832` with `1.5s` `soft8`).
  - `n=401 (L_probe=20.534534)`: `4pin=392`, `847=389`, `971=390`, `dp=396` (`#987_dp_w5_d34.58_a16_u0.96`); top-25 `aug0/soft8` hit **`397/401`** (`def=4`).
  - `n=626 (L_probe=25.519815)`: `4pin=609`, `847=610`, `971=507`, `dp=620` (`#1005_dp_w2_d39.0_a23_u0.0`); fine+soft best **`620/626`** (`def=6`).
  - `n=842 (L_probe=29.519815)`: `4pin=715`, `847=818`, `971=657`, `dp=776`; fine+soft best **`832/842`** (`def=10`, `#972_dp_w5_d34.58_a24_u0.96|soft8`).

### Idea 12C (Planned Before Testing: No-Snap `a`-Homotopy on `f_W = 0.75` Slabs, `C_4`-Reduced Pinched Frames `W = P + 0.76`, and Unsnapped `4frame_buf` / Diagonal-Pocket Melts)
- **Diagnostic motivation**:
  1. In `12A`, `RectSpatialSA::run_sa` had `if (r2 < 0.25) nth = 0.0` in `allow_align`, forcefully snapping tilted seeds into jammed `0°` lattices (`min_E = 9.85e-3 = (20.5345-21)^2/22`). Remarkably, `4frame_buf_W3` STILL broke that floor (`7.683e-3`).
  2. `12A` also only tested `W_strip = W + 0.267` (`W=2, 3`), missing the true **4-edge pinched frame** width `W_strip = P + (1 + f)/2 ≈ P + 0.763` (`f_W ≈ 0.763`, `P=3..6` around `(k - 2P - 1)^2` odd upright cores `M in {15, 13, 11, 9, 7, 1}` on `n=401`), where `(P + 1)*cos(th) + sin(th) <= P + 0.75` yields net positive excess `+0.21..+0.36` squares/unit-height (`break_even_H ≈ 65.8` at `P=5, th=28.5°`, right near `4 * (20.53 - 5.76) = 59.1` and well below `4 * (29.50 - 5.75) = 95.0` for `k=29`!).
- **Three fast C++ (`-O3 -fopenmp`, 7 threads, NO `nth -> 0` snap on tilted zones) experiments (`poc/idea12c_tilted_cavities.cpp`)**:
  1. **12C-1 Pure `f_W = 0.75` Slab Capacity + End-Taper**: Test `(P + 0.75) x (H + 0.25)` (`P = 3..8`, `H = 12..24`) holding `P*H + 1` and `P*H + 2` squares (plus cyclic-L corner turn `(2*H_arm - W) * W` holding `2*P*H_arm - P^2 + 1`) to measure exact break-even arm length `H` of `f_W = 0.75` sheared+tapered strips.
  2. **12C-2 Exact `C_4`-Symmetry Pinched 4-Edge Frame on `n = 401` (`k=20`, `L = 20.5345 & 20.5300`)**: Lock odd central core `M^2` (`M in {15, 13, 11, 9, 7, 5, 1}`) with `1`-layer unfrozen buffer + `4` `C_4`-coupled arms (`Q = (401 - M^2)/4 = 44..100` independent squares) seeded with `th in [18°, 43°]` sheared rows & tapered ends, NO `0°` snap on tilted squares (`|sin(2*th)| >= 0.30` until `a >= 0.992`). Also test tilted `C_4` central diamonds (`M x M @ 45°`, `M = 7, 9, 11`) + `C_4` tilted border pockets.
  3. **12C-3 Unsnapped `4frame_buf_W3..6`, `Lrim_buf_W4..8`, & `n=401` Central-Corridor / `n=626` Corner Melts**: Re-run `12A`'s buffered cavities and melted `merged.csv` corridors (`120..220` active squares) with tilted-angle preservation + joint full-box `Adam` + `polish_and_shrink`.
- **12C results (`task-923`, `297.7s`, 7 threads) — 0 wins, crisp algebraic & empirical falsification**:
  - **12C-1 (`Slab75` & `LCorner75`)**: `Slab75 P=3 (3.750x22.250, need 67)` dropped cleanly to `min_E = 2.556e-05` (down from `1.098e-02` at `H=14`, `8.113e-03` at `H=18`), while `P=4..7` stayed at `5.756e-03..2.379e-02` and `LCorner75` stayed at `6.012e-01..1.484e+00`. However, `67/22.25 = 3.011 < 3.5`: algebraic density check confirms `(P+1)*cos(th) > P + 0.5` with `(P+1)*cos(th) + sin(th) <= P + 0.76` is **impossible for all `P <= 13`** (only turns positive at `P >= 14`, `W >= 14.75`, needing `H >> 200`), proving why symmetric `f_W = 0.75` pinched 4-edge frames cannot beat `k^2 + 1`.
  - **12C-2 (`C4` exact 4-fold pinched frames & diamonds on `n=401`)**: `C4 upright` (`M=15..1`) hit `8.685e-02..1.767e-01` and `C4 45deg` (`M=11,9,7,5`) hit `1.277e-01..2.862e-01` — far worse than the `~1e-2` floor, directly confirming the density bound above.
  - **12C-3 (Unsnapped `4frame_ns` & seam melts)**: `4frame_ns_W3..6` stayed `2.267e-02..3.643e-02` (proving `12A`'s `7.68e-3` on `4frame_buf_W3` came from `0°` snap compressing back toward an upright grid), `401_seam_melt` stayed `1.254e-02`, and `626_seam_melt` (`L=25.5198`) hit `2.018e-03` (0 wins).

### Idea 13 (Planned Before Testing: Warm-Box `L = k + 0.56 -> shrink` No-Snap Multi-Domain Crystallization + `n=37` Hinged Chevron & `n=50` Edge-Slip Diagonal Bands on `401, 442, 530`)
- **Diagnostic findings from `n=37, 50, 403, 965` (`merged.csv`) & why `12A/B/C` stalled**:
  1. Direct inspection of **`n=37` (`L=6.598620 < 6.621320`, `D=1`)** reveals a **3-segment hinged diagonal band** (`6 @ 47.91°` / `3 @ 45.00°` / `6 @ 42.09°`, `SW=3, NE=3, NW=6, SE=10`, all on `Grid4`) saving `-0.0227` over uniform `45°`. Meanwhile **`n=50` (`L=7.571429`)** uses a `5`-slice **`3, 3, 4, 3, 3` band at `36.87°`** with **`du = +-0.19` edge-square longitudinal slip** and `dv = +0.20..+0.30` jogging — which rigid-slice `make_dp_jog_band` missed (`387/403` on `n=403`).
  2. All prior global `a`-homotopy runs (`12A, 12C`) annealed directly inside `L_probe = k + 0.5345`, where `E_upright = 9.85e-3` provides only `~0.04%` clearance over the target crystal and `a -> 1` crushes misoriented nuclei into glass/upright defects before grain boundaries align. Running `a: 0.84 -> 1.00` inside a **warm box `L_warm = k + 0.560` / `k + 0.575`** (with zero `0°` alignment bias on tilted regions + local nematic neighbor alignment + rigid cluster translate/rotate moves) lets clean zero-overlap non-`45°` crystals form first, after which **`polish_and_shrink_sq` (`Adam` + step-shrink) + LP tighten** reveals the exact minimum `L*` of each basin.
- **Two fast experiments (<6 min total on 7 CPU cores)**:
  - **13A (`poc/idea13_warm_discovery.cpp`, `-O3 -fopenmp`, 7 threads)**:
    1. **Warm-box de novo & perturbed crystal discovery on `n = 401, 442, 530`** at `L_warm in {k+0.555, k+0.570}` (`63` diverse seeds/target: `n=37`-style 3-segment hinged bands `w=3..7`, `n=50`-style variable-width edge-slipped sheared bands, 2-block `45° +- d` diamonds, small-tilt `4°..22°` multi-grain grids, asymmetric `45°`+corner pockets, and multi-grain chiral Voronoi seeds) with **NO `0°` snap**, local-radius nematic alignment (`r < 2.2`), and `polish_and_shrink_sq` on EVERY `max_pen <= 1e-10` solution (logging the lowest achieved `L*` per family even if `> L_base`, so we see exact basin `L*` gaps!).
    2. **Direct `n=401` / `n=530` / `n=626` hinge + asymmetric quadrant perturbation shrink**: deform `401`/`626` tilted blocks by `+-d` (`d = 1°..15°`), vary `(SW, NE)` counts, anneal at `L_warm`, and shrink via `Adam`.
  - **13B (`poc/idea13_chevron_highs.py`, `time_limit=2.5s`, ~60s on `n=401, 442`)**: Test `n=37`-style 3-segment hinged diagonal bands (`w=3..7`, `d=2°..14°`, `a1, a2, a3`), `n=50`-style alternating `(w, w-1)` + `du_edge` sheared diagonal bands (`20°..44°`), and `2-block` unequal `U x V @ (45° +- d)` via exact HiGHS (`Grid9` + fine interface) + `utils.relax_lbfgs` + `fixed_angle_lp_tighten`.
- **Idea 13 Results (FAIL, 0 wins; one strong lead `def=2`!)**:
  - **13A (`52.2s`, 224 jobs on `n=401, 442, 530, 626`)**: `feas=0/224` across all warm-box families (`L_warm = k+0.555, k+0.605`). Best `min_E`: `n=401 small_tilt=5.30e-03, voronoi=1.30e-02, 2blk_hinge=1.53e-02, slip_n50=1.56e-02, chev_n37=2.21e-02`; `n=442 small_tilt=6.84e-03..slip=2.65e-02`; `n=530 small_tilt=7.94e-03..slip=3.37e-02`; `n=626 hinge=3.64e-02`. Even at `L_warm`, `SA` noise at `a=1.0` jammed before relaxing to zero overlap.
  - **13B (`160.4s`, `docs/idea13b_results.json`)**: Bug check showed `make_chevron_band` (`u_rel` overlap at angle hinge) and `2blk` (`cx1, cy1` outside box) produced 0 candidates (`only fam=slip ran`), and `Calib n=50 slipped band` failed (`0/50`) because `make_slipped_band` used monotonic `dv_step` rather than `n=50`'s symmetric `(0, +0.2, 0, -0.2, 0)` profile. Despite that, `fam=slip` hit `exact_Grid9 = 397/401` (`#363_slip_w4_n5_d40.5_dv-0.15_s0.0 nt=21 up=376`) and **`399/401` (`soft8`, `def=2`)** on `n=401 (20.534534)`, and `436/442` exact / **`440/442` (`soft8`, `def=2`)** on `n=442 (21.534534)`!

### Idea 14 (Planned Before Testing: Calibrated Cluster/Multi-Diamond HiGHS + Pure L-BFGS/LP Warm Compression on `n=401, 442`)
- **Why**:
  1. Follow-up diagnostics (`task-1041`, `task-1049`, `task-1062`) showed `diam M=12 @ 45°` hits **`400/401`** (`144+256`, `def=1`) on `n=401`, `13B` bugs suppressed `chev` and `2blk` completely (`0` candidates), and `SA` (`12C`, `13A`) jammed even at `L_warm=20.605` whereas deterministic `L-BFGS` + `fixed_angle_lp_tighten` cleanly relaxes `401` squares without thermal noise.
  2. Board `#35/#36` highlighted that `n=364 (19+4/7)` uses 3 anti-diagonal `(3,4,5)` `36.87°` diamond clusters in a stepped `Grid4` lattice (`n=401` in `merged.csv` uses 2 anti-diagonal `5x5 @ 45°` blocks shifted toward `SW` with a 3-unit `NE` `L`-border!).
- **14A (`poc/idea14_clusters_highs.py`, exact HiGHS `tl=3.0s`, `n=401, 442`)**:
  - **Calibrated `n=37` chevron** (`>=37/37` verified first): step slices along each segment's own `(cos(th), sin(th))`, test single & 2-cluster anti-diagonal chevrons (`w=3..7`, `d=2°..14°`) + `(dx, dy)` shifts.
  - **Calibrated `n=50` bell-slipped & `(p,q)` multi-diamond clusters** (`>=50/50` verified first): symmetric bell `v`-profile + `du` edge slip + Pythagorean angles (`36.87°`, `22.62°`, `28.07°`, `40.5°`), `1`, `2`, `3`, and `4` anti-diagonal clusters + `(dx, dy)` shifts / `L`-border corner offsets.
  - **Fixed asymmetric `2blk` (`(c1, c2)` + `(c2, c1)` with `SW`-shifted `L`-border offsets) & `4pin` chiral `A x B @ th` frames** (`th = 1°..42°`) around central core, checked via `Grid9` (`aug=True` for top leads).
- **14B (`poc/idea14_lbfgs_compress.py`, pure `L-BFGS` + `fixed_angle_lp_tighten`, ZERO `SA`, `n=401, 442`)**:
  - Fix Numba `utils` import path (`PYTHONPATH=.:poc/swarm_angles/poc` + unpacking 4-tuple `(rx, ry, rth, pen)`!).
  - Launch multi-start `L_warm` (`k+0.62 -> k+0.525`) pure `L-BFGS` (`maxiter=600`) + `LP` continuation from: (a) top `14A` near-misses (`400/401` `diam12`, `399/401` `#363_slip`) with `1..2` hole-inserted squares; (b) `4`-arm chiral `A x B @ theta` (`theta = 1°..15°`) pinwheels around tilted/upright core; (c) global small-tilt (`0.5°..8°`) lattices + `1` inserted square; (d) `n=401` (`merged.csv`) macro-angle swaps (`45° -> 25°..44°` or quadrant chiral twists).
  - Scale via `L`-borders (`D=1` pool `442..962`) immediately on any `< L_base` win verified via `utils.verify_official`.
- **Idea 14 Results (FAIL, 0 wins; first high-feasibility pure L-BFGS+LP basins!)**:
  - **14A (`369.1s`, `docs/idea14a_results.json`)**: `Calib n=50` hit exact **`50/50`** (`16+34`); `Calib n=37` hit `31/37`. On `n=401 (20.534534)` (`892` cands): `bell=398/401` (`#146_bell_w3_n3_d16.2602 nt=10 up=388`), `chev=395`, `2blk=394`, `4pin=392`, `mcl=391`, `soft8=402` (`5x5@45`). On `n=442 (21.534534)` (`896` cands): `bell=439/442` (`nt=10`), `chev=436`, `2blk=435`, `4pin=433`, `mcl=433`, `soft8=443`.
  - **14B (`19.9s`, `docs/idea14b_results.json`)**: Pure deterministic `L-BFGS` + `fixed_angle_lp_tighten` (no `SA`) achieved **`25/49` feasible on `n=401`** (best `pin_A16_B4_d1.5`: `20.537812`, `dL=+0.002278`; `pin_A14_B6_d0.0`: `20.538000`; `pert_quad1.0`: `20.538000`) and **`27/45` feasible on `n=442`** (best `pin_A15_B6_d1.5`: `21.539063`, `dL=+0.003529`). Notice `warm_sched` jumped directly from `k+0.538` to `L_base-0.001` with `550` iters, stranding `diam12_ins`, `glob_d`, and `bell_ins` before converging.

### Idea 15 (Planned Before Testing: Fine 14-Step L-BFGS+LP Continuation + `n=530, 842` Tilted-Core & Pinwheel Grid9)
- **Why**:
  1. `14B` ran in `19.9s` and lifted feasibility from `0%` (SA) to `>50%`, getting within `+0.002278` of `L_base`, but skipped intermediate `L` steps between `k+0.56 -> k+0.538 -> k+0.5345` and only ran `maxiter=550`.
  2. `14A` only searched `n=401, 442`, missing **`n=530` (`k=23`, `gap=0.0182`)** and **`n=842` (`k=29`, `L_base=29.520815 -> L_front=29.502153`, `gap=0.0187`)** where larger-span rotated cores (`M=11..25 @ 5°..44°`) or 3–5 cluster diagonal chains could unlock strictly below `L_base`.
- **15A (`poc/idea15_fine_compress.py`, `n=401, 530, 842`)**:
  - Run `16`-step fine `L`-continuation (`L_base + 0.08 -> -0.012`, `maxiter=1100`) with `L-BFGS` + `fixed_angle_lp_tighten` on: `pin_A_B` chiral pinwheels, `diam12_ins`, `#146_bell_ins`, `#363_slip_ins`, `glob_d`, and multi-angle perturbation kicks (`0.08..0.25 rad` on `merged.csv` tilted + neighbor pockets).
- **15B (`poc/idea15_large_k_grid9.py`, `n=530, 842`, `tl=2.0s`)**:
  - Exact `Grid9` HiGHS (`L_probe = L_base - 0.001`) on `n=530` (`23.5345`) and `n=842` (`29.5198`) across: rotated `A x B` blocks (`8..25 @ 5°..44°`, including shifted centers), `2`-`5` anti-diagonal `(p, q)` Pythagorean clusters, calibrated `bell`/`chev`, and `4pin` frames. Any `< L_base` win verified via `utils.verify_official` and propagated by `L`-borders.
- **Idea 15 Results (FAIL, 0 wins; uncovered `Grid9` MILP timeout artifact on large `k`)**:
  - **15A (`80.3s`, `docs/idea15a_results.json`)**: `23/29` (`n=401`), `22/24` (`n=530`), `22/24` (`n=842`) feasible; best `n=401 pin_A16_B4_d1.5` reached `20.536595` (`+0.001061`), `n=530 pin_A19_B4_d1.5` `23.538082` (`+0.002548`), `n=842 kick_s0.08_1` `29.530193` (`+0.009377`), hitting the `45°`-core `L_base` floor.
  - **15B (`271.3s`, `docs/idea15b_results.json`)**: On `n=530`: `rot=528`, `bell=525`, `chev=524`, `mcl=521`, `4pin=511`. On `n=842 (29.519815)`: `rot 17x17 @ 45°` hit `842/842` only with `mg=-0.0015` / `soft8` (matching `L_base=29.520815`), while small/medium tilted clusters (`bell=691`, `chev=694`, `mcl=811`) suffered massive `2.0s` HiGHS MILP timeouts on `~7500`-var full `Grid9` (`29x29`). Follow-up benchmark proved `Grid4` (`offs=[0, f]`) solves `n=842` `#239_bell` to `status=0` global optimality (`835/842`) in **`0.12s`**!

### Idea 16 (Planned Before Testing: Zero-Timeout `Grid4 + Local-f/2` Optimal HiGHS on `n=401, 530, 626, 842`)
- **Why**: Full `Grid9` (`[0, f/2, f]` everywhere) blew up MILP variables on `k >= 23`, causing `12B` (`n=626, 842`) and `15B` (`n=842` `bell/chev/mcl/rot`) to time out `100+` squares below true optimum. Switching to `Grid4` (`[0, f]`) globally + `f/2` ONLY within `2.5` units of tilted centers (`~0.1-0.4s`, `status=0` optimal!) gives true exact counts across ALL families.
- **Plan (`poc/idea16_grid4_local_highs.py`, `n=401, 530, 626, 842`, `tl=3.0s`)**:
  - Evaluate `rot` (`M=10..24`, `A x B` with `|A-B|<=4`, `theta=5°..45°`, `(dx, dy)` shifts), `bell` (`w=3..8, n_sl=3..13`, Pythagorean + `40.5°` + shifted centers), `chev` (`w=3..7, n_end=2..7`), `mcl` (`2..5` anti-diagonal clusters + `L`-border offsets), `2blk`, and `4pin` at `L_probe = L_base - 0.0005` using `Grid4 + local-f/2`.
  - Polish any `tot >= n` hits via `relax_lbfgs` + `fixed_angle_lp_tighten` + `verify_official`, saving `win_n<n>.csv` and propagating via `L`-borders if `< L_base`.
- **Idea 16 Results (FAIL, 0 wins; `135.2s`, `docs/idea16_results.json`)**:
  - Benchmark caught that adding `local-f/2` in coarse MILP caused `3.38s` fractional relaxation timeouts on `n=842`, so coarse used pure `Grid4` (`local_r=0.0`, `0/521` timeouts on `401`, `2/556` on `530`, `16/609` on `626`, `79/648` on `842`) followed by `g4_0` / `loc_0` (`local_r=2.0, mg=0.0`) fine checks on top 4 per family.
  - True `status=0` optimal counts across all `n`: `401 -> rot=400, bell=398, 2blk=396, chev=395, mcl=390, 4pin=384`; `530 -> rot=528, bell=527, 2blk=525, chev=524, mcl=521, 4pin=513`; `626 -> rot=626 (17x17@45_-0.5 at mg=-0.001; exact0 best 624), bell=623, chev=620, 2blk=617, mcl=615, 4pin=609`; `842 -> rot=842 (17x17@45_-0.5 at mg=-0.001; exact0 best 840), bell=839, chev=836, 2blk=833, mcl=831, 4pin=825`. Across `k=20..29`, all tested `Grid4` tilted-core/cluster/pinwheel families cap strictly at or below `k^2` (`L >= L_base`), confirming no finite-template `Grid4` unlock exists for `D=1` below `20.535534` / `25.520815`.
  - Status: genuinely out of viable `D=1` non-`45°` template ideas after Ideas 1–16 (`0` wins); ready for `@coordinator` reassignment or shutdown.





