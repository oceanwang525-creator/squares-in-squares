# Swarm Inflation: From-Scratch GPU Multi-Grain Inflation (`0.88 -> 1.00`)

Target: discover strictly lower-$L$, zero-overlap, non-grid packings from scratch for assigned top-10 gap `n` (`367, 910, 971` first; extended pool `971, 367, 910, 972, 791, 969, 687, 851, 796, 333`), never seeding from incumbent coordinates.

## Target Table (Assigned Pool)

| `n` | `k` | `D` | `L_ours` (`merged.csv`) | `L_frontier` | Gap `dL` | Gap Score |
|---:|---:|---:|---:|---:|---:|---:|
| 367 | 19 | 6 | 19.6923881554 | 19.6708567970 | 0.021531 | 0.001124 |
| 910 | 30 | 10 | 30.6791494646 | 30.6464577019 | 0.032692 | 0.001084 |
| 972 | 31 | 11 | 31.6839902146 | 31.6514012987 | 0.032589 | 0.001045 |
| 791 | 28 | 7 | 28.6340707516 | 28.6058845085 | 0.028186 | 0.001002 |
| 969 | 31 | 8 | 31.6339825014 | 31.6055708986 | 0.028412 | 0.000913 |
| 971 | 31 | 10 | 31.6643183252 | 31.6367893146 | 0.027529 | 0.000883 |
| 851 | 29 | 10 | 29.6841436064 | 29.6626447172 | 0.021499 | 0.000737 |
| 796 | 28 | 12 | 28.7215400023 | 28.7012840810 | 0.020256 | 0.000718 |
| 687 | 26 | 11 | 26.7232714974 | 26.7048349736 | 0.018437 | 0.000703 |
| 333 | 18 | 9 | 18.7726341135 | 18.7600141531 | 0.012620 | 0.000692 |

---

## Idea Log

### Idea 1 — Oriented Multi-Grain Seeding + Batched CUDA Inflation (`0.88 -> 1.00`) + Shake/Swap/Alignment + LP/SLP Polish

- **Status:** Logged before testing.
- **Hypothesis:** Previous from-scratch searches (`poc/soft_to_rigid`, `poc/global_search_midn`, Loop 28 block seeds) jammed because they either started from amorphous random angles at `a=1.0` (where orientational frustration caps density near ~0.85) or placed rigid `a=1.0` blocks with 3–4 missing squares that could not penetrate the rigid block-pocket interface. Starting all `n` squares from scratch inside `[0, L_target]^2` at scale `a = 0.88` (effective area density ~0.75) using **oriented multi-grain / catalog-inspired topologies** gives squares 12% linear slack to reorganize pocket ends, corner miters, and core-pocket zipper teeth during `a in [0.88, 0.96]` before jam lock-in.
- **Planned Seed Families (strictly from scratch, zero incumbent reads):**
  1. `sheared_l_pocket`: NE upright block `Kx x Ky` + West/South zero-sawtooth sheared pockets (`w*cos(th) + sin(th) = W`) + SW corner block/miter + excess squares placed in largest clearances along pocket seams/ends.
  2. `pinched_4edge_frame`: Interior upright core surrounded by 4 asymmetric sheared boundary strips (West, East, South, North) splitting `(L - Kx)` and `(L - Ky)` so every edge gets larger fractional slack `f_i in [0.70, 0.95]`.
  3. `chiral_tilted_core`: Tilted `m x m` interior core (including Pythagorean slopes `3-4-5`, `5-12-13`, `8-15-17`, `7-24-25`) + 4 congruent rotated wings.
  4. `bicrystal_chevron`: Symmetric tilt `(+alpha, -alpha)` multi-grain / V-chevron sheared pockets cutting end-wedge waste.
  5. `catalog_corner_lift`: `K x K` upright frame with an `m x m` corner replaced by a tilted/Pythagorean/45-degree corner gadget (`m^2 + D` squares) without artificial hard walls, allowing seam teeth to interlock during inflation.
- **Planned Optimizer & Moves (RTX 4070 Super PyTorch CUDA):**
  - Batched neighbor-list exact SAT quadratic penetration + wall violation energy with analytical gradient (`float32` for `a < 0.99`, `float64` for `a >= 0.99 -> 1.00`).
  - During `a in [0.88, 0.96]`: Adam descent + targeted Langevin shake on stressed squares + high-stress-to-void relocation + local orientation alignment / swap moves (`sin^2(2*(th_i - th_j))` soft alignment on half the batch, decayed to 0 before `a = 0.98`).
  - Post-inflation (`a = 1.00`): minimal uniform expansion `lambda` to clear any tiny residual overlap, followed by unbounded fixed-angle LP (`alln_fixed_lp` pattern) + Lipschitz-safe free-angle SLP (`lip_slp` pattern), then strict dual certification (`squares.core.certify` literal + `kaggle_roundtrip`).
- **Initial Fast Test Scope (<10 min):** `n in {367, 910, 971}` across target `L_target` values between `L_frontier` and `L_ours - 0.001`.

### Results Log (Idea 1)

- **Scope & Runtime:** `--batch 160 --top-k 12 --seed 1 --preset fast` on RTX 4070 SUPER (`torch 2.13.0+cu130` via `vllm-gemma4`) + CPU HiGHS LP/Lip-SLP polish + dual certification (literal `squares.core.certify` & `pandas.read_csv/to_csv`).
  - `n=367`: GPU `18.29s`, total `33.52s`. Zero overlap (`min_max_viol=0.000e+00`) held all the way through `a=0.9820`; jammed at `a=0.9980..1.0000` (`min_max_viol=1.830e-02`).
    - Pre-LP best (`lam=1.018819`): `20.047593` (`sheared_l_pocket(K=11x14,w=9/6,upright)`)
    - Best certified (LP+SLP): `L_lit = L_pd = 19.913271896102616` vs `inc_L = 19.692388155425202` (`+0.22088`, **not a win**)
    - Family `best_L_cert` pre-LP: `sheared_l_pocket` `20.0476`, `catalog_lift` `20.0719`, `chiral_core` `20.1224`, `bicrystal_2grain` `20.2649`, `pinched_4edge` `20.3640`, `v_chevron` `20.5602`.
  - `n=910`: GPU `53.87s`, total `122.4s`. Zero overlap through `a=0.9820`; jammed at `a=0.9980..1.0000` (`min_max_viol=1.536e-02`).
    - Pre-LP best (`lam=1.015783`): `31.118428` (`sheared_l_pocket(K=24x21,w=7/10,sheared_ext)`)
    - Best certified (LP+SLP): `L_lit = L_pd = 30.793072708818634` (`sheared_l_pocket(K=21x25,w=10/6,upright)`, dropped `31.1255 -> 30.7931` under fixed-angle LP) vs `inc_L = 30.679149464635975` (`+0.11392`, **not a win**)
    - Family `best_L_cert` pre-LP: `sheared_l_pocket` `31.1184`, `catalog_lift` `31.2488`, `pinched_4edge` `31.4146`, `v_chevron` `31.5385`, `chiral_core` `31.6425`, `bicrystal_2grain` `32.5872`.
  - `n=971`: GPU `58.90s`, total `142.82s`. Zero overlap through `a=0.9820`; jammed at `a=0.9980..1.0000` (`min_max_viol=1.399e-02`).
    - Pre-LP best (`lam=1.013294`): `32.072814` (`sheared_l_pocket(K=25x25,w=7/7,upright)`)
    - Best certified (LP+SLP): `L_lit = L_pd = 31.789411346959417` (`sheared_l_pocket(K=21x26,w=11/6,sheared_ext)`, dropped `32.2935 -> 31.7894` under fixed-angle LP) vs `inc_L = 31.664318325205181` (`+0.12509`, **not a win**)
    - Family `best_L_cert` pre-LP: `sheared_l_pocket` `32.0728`, `catalog_lift` `32.2442`, `chiral_core` `32.4225`, `pinched_4edge` `32.5861`, `bicrystal_2grain` `32.6009`, `v_chevron` `32.8039`.
- **Diagnosis of Failure:**
  1. **Seed capacity / geometry gaps before inflation:** `make_sheared_strip` leaves diagonal triangle waste at strip ends, `sw_mode="upright"` wastes a `~0.67`-wide dead corridor around the SW block (`floor(W)` vs `W`) while colliding `west` and `south` at `[p1, W1] x [p2, W2]`, and `complete_to_n` injects/deletes arbitrary void squares instead of choosing exact-capacity `(Kx, Ky, w1, w2)` / diagonal-45° hybrid topologies (critical for `n=367` where `45°`+upright dominates!).
  2. **Uniform dilation & stiff contact network at `a >= 0.985`:** Adam descent on `pen^2` inside fixed `box_L` cannot propagate macroscopic block/grain translations across `367..971` squares (`L_pre_lp -> L_lp1` dropped `0.33..0.50` in a single fixed-angle LP solve, proving huge unexploited slack remained!), while `minimal_feasible_dilation` inflates the entire box because of 1–2 pinned pairs.
  3. **Fixed `n_moves=3` local swaps at `a <= 0.962` cannot fix topological deficits** where an extra square needs a row/column fault-line glide or corner re-tiling.

---

### Idea 2 — Custom Shared-Memory `nvcc` CUDA Physics (`sm_89`) + Elastic-`L` / Cluster Moves + Exact-Capacity Multi-Grain & 45°-Hybrid Seeds

- **Status:** Logged before coding/testing.
- **Hypothesis:**
  1. Eliminating the `~7-15` square dead corners/overlaps in seeds via **exact-capacity (`|C - n| <= 3`) non-overlapping geometric pre-screening** across 6 improved topological families (tapered sheared L-pocket with wedge-filled ends & zero SW dead corridor, `45°`-upright diagonal bands / rotated core / corner triangles, 2-tier stepped sheared pockets, clipped pinwheel chiral cores, Pythagorean corner blocks, and clipped symmetric-tilt bicrystals) will start GPU inflation much closer to target density.
  2. Replacing PyTorch eager kernel launches with a **shared-memory spatial-hash `nvcc` (`sm_89`) CUDA solver** (`sat_lmin.cu`, ~50–100x faster: `B=320` chains × `20k–30k` steps in ~10s) enables:
     - **Parallel 256-thread void relocation & neighbor angle-snap moves** inside every trajectory block during `a in [0.88, 0.97]`,
     - **Cluster / grain rigid-translation moves** (summing net SAT forces on orientation clusters & spatial slabs to glide entire `100–500`-square grains in one move instead of stalling in Adam),
     - **Deflate-reinflate annealing pulses** (`a: 0.88 -> 0.97 -> 0.94 -> 0.99 -> 0.975 -> 1.00`) to melt pinned end-wedges, and
     - **Direct continuous `(x0, y0, L)` elastic-box compression** (`w_L * L + w_pen * (sum pen^2 + wall^2)`) with ramping penalty `w_pen` + final `float64` all-pairs expansion & top-24 SciPy HiGHS fixed-angle LP + Lip-SLP polish + dual certification.
- **Planned Verification & Fast Scope (<10 min):**
  - Unit-test `sat_lmin_cuda` exact SAT energy/gradient & all-pairs `L_cert` against `sat_cuda.py` / finite differences (`pytest`).
  - Test fast on `n in {367, 910, 971}` first; evaluate both raw seed `L`, GPU `L_cert`, and post-LP/SLP `L_lit`/`L_pd`.

### Results Log (Idea 2 — Initial Buggy Run)
- Ran `run_inflation_v2.py --n-list 367,910,971 --batch 336 --top-k 28 --seed 2 --fast-mode 1` (`sat_lmin.cu` compiled clean with `nvcc -O3 -arch=sm_89`).
- **Observed timing:** `n=367` GPU `1.75s` (total `38.68s`), `n=910` GPU `3.40s` (total `17.34s`), `n=971` GPU `3.65s` (total `26.25s`) — **10x–16x faster than Idea 1 PyTorch**, confirming raw `sm_89` shared-memory kernel speed.
- **Observed bug failure (`best=None, win=False` on all 3 `n`):**
  1. CUDA `gpu_L_cert` reported `20.668` (`lam=1.00547` on `n=367`), `31.930` (`lam=1.00561` on `n=910`), `32.610` (`lam=1.00797` on `n=971`), yet Python `minimal_feasible_dilation` rejected candidates (`py_lam = 1e9`, `min_dist2 = 0.0` at indices `96,97`). Diagnostic on `B=4, N=367` proved a **missing `__syncthreads()` WAR hazard after `red_buf[0]` / `ired_buf[0]` reads** in `polish_and_cert_fp64` (`xmin_all`, `ymin_all`, `lam_all`), `block_reduce_sum`, and `void_relocate` (`w_i`): for `N > 32` (warps 1..7), thread 0 overwrote `red_buf[0]` before warps 1..7 read `xmin_all` / `lam_all`, collapsing higher-index squares onto each other (`N=5` unit test stayed inside warp 0 and masked the race).
  2. Elastic `sh_L` Adam ratchet (`w_L = 1.0` vs `w_pen = 40..120000` -> `sh_v_L ~ w_pen^2`) expanded `sh_L` outward `+1.0..+1.2` (`19.7 -> 20.67`, `30.7 -> 31.93`, `31.7 -> 32.61`) during early overlap spikes and poisoned `sqrt(sh_v_L)` so `sh_L` could never compress back.

---

### Idea 2b — Reduction Barrier Fix (`N > 32` Verified) + Dimensionless Wall-Kiss `L` Controller + Tight Target-`L` Sweep

- **Status:** Logged before coding/testing.
- **Hypothesis:**
  1. Encapsulating every block reduction in `__device__` helpers (`block_reduce_sum/min/max` for `f32`/`f64`) with a mandatory post-broadcast `__syncthreads()` and fixing `void_relocate` energy consistency eliminates all inter-warp WAR races for `N <= 1024`, making `gpu_lam == py_lam` to `< 1e-6` even at `N=971`.
  2. Replacing Adam `sh_v_L` on `(sh_x0, sh_y0, sh_L)` with a **dimensionless wall-overhang controller** (`W_v = sum(vL + vR + vB + vT)` driving `sh_L` gently toward decaying `target_w`, tightly bounded above at `L_init + {0.005, 0.03, 0.08}` across batch slices) prevents box blowup, keeps boundary pressure forcing interior grain rearrangement during `a: 0.88 -> 1.00`, and enables `use_elastic_L=0` vs `1` comparison.
- **Planned Verification & Fast Scope (<5 min):**
  - Add regression test in `test_inflation.py` verifying `N=367` (>256 threads) matches Python `minimal_feasible_dilation` `lam` within `1e-5` and satisfies `sat_max_penetration <= 0`.
  - Run `run_inflation_v2.py` on `{367, 910, 971}` with `--batch 336 --top-k 32 --fast-mode 1` (and compare `elastic-L 1` vs `0`).

### Results Log (Idea 2b)
- Verified `N=367 > 256` multi-warp barrier fix: `diag_v2.py` (`B=4, N=367`) returned `py_lam = 1.0`, `min_dist2 >= 1.000`, `gpu_L = 20.077..21.160`.
- Built standalone sandbox C++/OpenMP + CUDA pipeline (`poc/swarm_inflation/poc/cpp_pipeline.cpp`, 7 OpenMP threads + `/tmp/sat_lmin_cuda` + `float64` Augmented-Lagrangian L-BFGS squeezer + 17-digit CSV self-check) because `.venv/bin/python` unsandboxed runs stalled on approval timeout after restart (`#16`).
- Ran `cpp_pipeline --n-list 367,910,971 --batch 336 --top-k 32 --seed 2 --fast-mode 1` for both `--elastic-L 1` (`task-408`, **14.4s total** across all 3 `n`) and `--elastic-L 0` (`task-420`):
  - `n=367` (`inc_L = 19.692388155425`, `L_front = 19.670856797022`):
    - `elastic=1` (`2.76s`): `gpu_L=19.808442 -> AL_L=19.771311410262` (`+0.0789` vs `inc_L`, `-0.142` vs Idea 1!) (`tapered_L(K=13x14,w=7/6,Wfull)`)
    - `elastic=0` (`2.64s`): `gpu_L=19.852443 -> AL_L=19.783357071088`
  - `n=910` (`inc_L = 30.679149464636`, `L_front = 30.646457701947`):
    - `elastic=1` (`5.64s`): `gpu_L=30.777416 -> AL_L=30.762085137680` (`+0.0829` vs `inc_L`, `-0.031` vs Idea 1!) (`tapered_L(K=22x25,w=9/6,Sfull)`)
    - `elastic=0` (`5.44s`): `gpu_L=30.859839 -> AL_L=30.808910444995`
  - `n=971` (`inc_L = 31.664318325205`, `L_front = 31.636789314648`):
    - `elastic=1` (`6.03s`): `gpu_L=31.774126 -> AL_L=31.755274981602` (`+0.0910` vs `inc_L`, `-0.034` vs Idea 1!) (`tapered_L(K=24x25,w=8/7,Sfull)`)
    - `elastic=0` (`5.75s`): `gpu_L=31.854548 -> AL_L=31.803349296010`
- **Diagnosis of Remaining `+0.079 .. +0.091` Gap (Proved Mathematically & Numerically):**
  1. **Strict Geometric Capacity Ceiling on 2-Strip L-Pockets:** Sweeping *all* `(Kx, Ky) in [0, floor(L)-1]^2` for standard 2-strip sheared L-pockets at `L = 19.690 / 30.675 / 31.660` proved maximum un-fanned capacity is strictly **below `n`**: `-3` on `367`, `-5` on `910`, `-4` on `971`! Why: (a) `seeds_v2` restricted `Kx, Ky <= k - 2` and `W >= 1.6`, skipping `Ky = floor(L)` (`1`-strip configurations where `w*sin(th)` waste only occurs on ONE strip instead of two!), and (b) flat-cut sheared strips waste `~W^2 * tan(th)` at their two diagonal ends (`max_hole_r = 0.865` at strip ends in `cand_v2_n367`).
  2. **Bulk Waste Identity (`Waste / H = sin(th)`):** Narrower pockets (`w=6..8, th=24..29 deg`) waste `sin(th) ~ 0.42..0.48` per unit strip length, whereas wide single strips (`w=10..17, th=15..21 deg`) halve `sin(th) ~ 0.26..0.36` and cut total strip length in half (`1` strip instead of `2`!).
  3. **Exact Smectic End-Fan Proof at `a=0.89`:** Unfolding end rows `0 deg -> ... -> th` on a single wide sheared strip alongside `K_x x floor(L)` upright ALREADY achieves **`diff = +1` (`368/367`) at `L=19.690`**, **`diff = 0` (`910/910`) at `L=30.675`**, and **`diff = 0` (`971/971`) at `L=31.660`** — strictly inside `inc_L`!

---

### Idea 3 — Dual-End Smectic-Fanned Single/Split Strips + Unconstrained `K` Sweep + Multi-Grain Low-Jitter CUDA Inflation (`a = 0.89 -> 1.00`)

- **Status:** Logged before coding/testing.
- **Hypothesis:**
  1. Adding **dual-end smectic fans** (`0 deg -> ... -> th` at BOTH bottom and top ends of sheared strips, plus multi-row slanted/staggered interface rows) and removing the artificial `Kx, Ky <= k - 2` ceiling so `Ky = floor(L)` (`1`-strip, `K in [4, floor(L)-2]`, `w in [6, 22]`) and **2-strip split-column/pinched-frame** configurations can compete lets our GPU seeds start at **zero capacity deficit (`>= n`) inside `L <= inc_L`**.
  2. Using **low-jitter seeding (`0.001..0.005`)** for pre-aligned fanned crystal regions plus **adaptive angle-preserving / gentle CUDA anneal (`a: 0.90 -> 1.00`, `fast_mode=0` on top seeds)** and **local swap/re-angle shake inside AL-LBFGS** allows the smectic fan to lock smoothly at `a = 1.00` without shattering the `w`-strip pitch (`w*cos(th)+sin(th)`).
  3. If `{367, 910, 971}` beats `inc_L` (or closes the gap sharply), scale Idea 3 immediately to all 10 assigned targets `{971, 367, 910, 972, 791, 969, 687, 851, 796, 333}`.
- **Planned Verification & Fast Scope (<5 min):**
  - Implement Idea 3 seed families + `--mode` sweep in `cpp_pipeline.cpp` (and update `seeds_v2.py` once verified), test fast on `{367, 910, 971}` first (`~15–30s`), inspect gaps, refine if needed, and scale winners to the full 10-target pool.

### Results Log (Idea 3)
- Cleaned & executed Idea 3 in `cpp_pipeline.cpp` + `sat_lmin.cu` (`--idea 3 --batch 448 --top-k 48 --seed 3 --fast-mode 1 --elastic-L 1`, `task-483`, 20.6s total):
  - `n=367` (`3.82s`): `gpu_L=19.806389 -> AL_L=19.781595647061` (`+0.0892` vs `inc_L`, `smectic1(K=3x19,w=17,nf=2)`).
  - `n=910` (`7.90s`): `gpu_L=30.736812 -> AL_L=30.733022811846` (**`+0.0538` vs `inc_L`, `-0.0291` better than Idea 2b!**, `lam=1.000278`, **`hybrid45_band(wb=6)@L=30.65790`** -> `677` upright + `221` curved `39°..47°` diagonal squares!).
  - `n=971` (`8.92s`): `gpu_L=31.829494 -> AL_L=31.785084704612` (`+0.1208` vs `inc_L`, `hybrid45_band(wb=6)@L=31.65661`).
- **Diagnostic Audit (`a=1.000` Exact Capacities & Geometry Breakdown):**
  1. Note all 10 assigned targets are `n = k^2 + d` (`d in 6..12`) with `L - k in [0.606, 0.773]` — classically governed by diagonal/tilted bands (`hybrid45_band`) or single/L-pocket wide sheared strips.
  2. Exact `a=1.000` rigid capacity tests (`task-511`, `task-520`, `task-524`, `task-528`, `task-532`) ruled out:
     - Pinched 4-frame (`-29` on `367`, `-35` on `910`, `-31` on `971`) — 4 internal corners destroy density.
     - V-bent chevron strip (`-13` on `367`, `-16` on `910`, `-15` on `971`) — central apex seam loses an entire column.
     - 4-corner 45° frame (`-6`, `-10`, `-10`) and central 45° diamond (`-10`, `-17`, `-18`).
  3. Meanwhile **three families hit `-3 .. -5` at exact `a=1.000`** before any relaxation:
     - **Offset `45°` band (`45band`)**: `364/367 (-3)` (`wb=3, du=-3.9`), `905/910 (-5)` (`wb=3, du=-2.5`), `966/971 (-5)` (`wb=3, du=-3.2`).
     - **Single wide sheared strip (`smectic1`)**: `364/367 (-3)` (`K=10, w=10@21.09°`), `905/910 (-5)` (`K=6, w=25@11.82°`), `967/971 (-4)` (`K=15, w=17@15.32°`).
     - **2-strip L-pocket**: `967/971 (-4)` (`K=21x21, w=11/11@20.33°`).
  4. **Why Idea 3 missed these `-3 .. -5` basins:**
     - `seed_smectic_1strip` ranked `(Kx, w)` by `a=0.895` count (`K=3, w=17` on `367` instead of `K=10, w=10`), breaking early before testing the optimal `a=1.000` strip widths.
     - `hybrid45_band` received only ~1.25% of the batch, used `0.985` upright spacing, locked both upright sides to one shared `(phx, phy)` phase instead of corner-anchored `(0.5+ix)` and `(L-0.5-ix)` grids, and never swept `wb, u0` by `a=1.000` capacity — yet STILL won on `910` and `971`.
     - `sat_lmin.cu` shrank ordered `-3..-5` seeds down to `a=0.880` (`lr=0.016`) + `void_relocate` teleportation + allowed `sh_L` to expand `+0.075` (`max_L_b`), melting crisp grain boundaries and locking in inflated boxes; and Idea 3 used only AL-LBFGS without SciPy HiGHS LP / Lip-SLP polish (`.venv` restored in `#19`).

---

### Idea 4 — `a=1.000`-Ranked Multi-Grain Basins (`45band_exact` + `smectic1_unit` + `tapered_L_unit`) + Gentle-Compression CUDA (`fast_mode=2`, `a=0.945 -> 1.000`) + Full LP/Lip-SLP Polish

- **Status:** Logged before coding/testing.
- **Hypothesis:**
  1. Generating initial basins strictly from top **`a=1.000`** rigid configurations (`-3..-5` from `n`) across **50% `45band_exact`** (fine `u0` offset, independent corner-anchored low/high `0°` lattices, tilted `40°..45°` band angles & pre-curved `39°` corner-flares mimicking `cand_v3_n910`, plus `hybrid45_split` / rotated edge strips), **30% `smectic1_unit`** (exact `(Kx, w)` sorted by `a=1.000` count + 1–2 end transition rows), and **20% `tapered_L_unit` / `sandwich_unit`**, then inserting only the remaining `3..6` squares into largest interface/edge voids at `a=0.94..0.96` places trajectories directly in the winning basins.
  2. Adding **`fast_mode=2` (`gentle_warm`)** in `sat_lmin.cu` (`a: 0.945 -> 1.000` in 10 smooth stages, `lr: 0.0055 -> 0.00004`, zero `void_relocate` teleportation, tight `max_L_b = sh_L_init + {0.002, 0.010, 0.025}`, lower `target_w` so `sh_L` stays below `inc_L`) preserves ordered grain structure during compression instead of melting it at `a=0.880` (A/B tested against `fast_mode=1` with tightened `max_L_b`).
  3. Post-polishing top GPU + AL-LBFGS candidates through exact **HiGHS LP (`fixed_angle_lp_polish`) + Lip-SLP (`lip_slp_polish`)** in `.venv/bin/python` (`polish_candidates`) recovers any rigid sliding slack left by penalty methods.
- **Planned Verification & Fast Scope (<5 min):**
  - Run `pytest -q poc/baseline/poc/tests/ poc/swarm_inflation/tests/`.
  - Compile updated `sat_lmin.cu` & `cpp_pipeline.cpp` (and sync Python runners/seeds for `polish_and_certify.py`), benchmark A/B (`fast_mode=2` vs `1`) on `{367, 910, 971}` + full LP/Lip-SLP polish (`<2 min` total).
  - If `{367, 910, 971}` yields wins (`win_n<n>.csv`), verify `squares.core.certify` + `pandas` roundtrip and hand off to `@verifier`, then scale immediately to all 10 targets `{971, 367, 910, 972, 791, 969, 687, 851, 796, 333}`.

### Results Log (Idea 4)
- Fixed a critical `sat_lmin.cu` clamp discovered by `pytest`: `min_L_b = sh_L_init * 0.992f` blocked `sh_L` from shrinking >0.8%, causing `test_cuda_v2_synthetic_5_compresses_to_optimum` (`3.0 -> <2.71`) to stall at `2.9658`. Changed `min_L_b` to `max(1.0, sh_L_init * 0.75)` and added `fast_mode=2` (`a: 0.945 -> 1.000`, zero `void_relocate`, tight `max_L_b`); all `7/7` tests in `poc/swarm_inflation/tests/` passed (`19.80s`).
- Ran A/B test (`batch=480, top-k=48, elastic-L=1`, `{367, 910, 971}`):
  - **`fast_mode=2` (`a=0.945 -> 1.000`, `task-575`, 23.5s total):** `367 -> 19.787425`, `910 -> 30.840073`, `971 -> 31.845026` — **worse on all 3** (pre-jamming at `0.945` prevents grain adjustment).
  - **`fast_mode=1` (`a=0.880 -> 1.000`, tightened `max_L_b`, `task-577`, 25.0s total) + HiGHS LP & Lip-SLP polish (`.venv`, `task-593`):**
    - `n=367`: `gpu_L=19.768623 -> AL_L=19.755795 -> LP/SLP=19.755485` (**`+0.0631` vs `inc_L` (~2.5 squares!), best so far on `367`** from `tapered_L(K=14x13,w=6/7,Sfull)`).
    - `n=910` (incl. `cand_v3_n910`): `AL_L=30.733023 -> LP/SLP=30.730561` (**`+0.0514` vs `inc_L` (~3.1 squares!), best so far on `910`** from `hybrid45_band(wb=6)`). Idea 4 `fast_mode=1` best alone was `30.770170`.
    - `n=971`: `gpu_L=31.839918 -> AL_L=31.789149 -> LP/SLP=31.730153` (**`-0.0590` drop in Lip-SLP! `+0.0658` vs `inc_L` (~4.1 squares!), best so far on `971`** from `smec1u(K=13,w=19,nf=0)`).
- **Spatial & Diagnostic Findings (Only ~2.5–4.1 Squares Left!):**
  1. **Why `band45u` failed in Idea 4:** `top_m=12` sorted by rigid `a=1.000` count only selected `wb=2..3` locked tightly at `0.995` spacing, starving the `wb=5..10` curved-band basin. Meanwhile `v3_910_h45` (`30.730561`) formed a symmetric **S-arched `50.8° -> 45° -> 38.8°` diagonal band (`wb=6`, `band=232`)** between two triangular numbers **`T(24)=300` and `T(27)=378`**, leaving zero large interior voids (`d <= 0.79`) except two `0.816` gaps at the band corners `(0, L)` & `(L, 0)`!
  2. **Why `smec1u` & `tapered_L` lost ~3–4 squares:** Direct `x < 1.2` inspection of `cand_v4_n971` and `cand_v4_n367` shows periodic **`0.86–0.90` empty holes spaced `1 / sin(th)` (~3.8–3.9 units) along `x=0.5`** where sheared-strip ends meet flat walls.
  3. **Why C++ `AL-LBFGS` missed top LP basins on `971`:** `Lip-SLP` cut `31.789 -> 31.730` (`-0.0590`), meaning C++ `AL-LBFGS` ranking discarded strong candidates before `.venv` LP/Lip-SLP polish.

---

### Idea 5 — S-Arched Diagonal Bands (`50°->45°->39°`, `wb=3..11`), Wall-Notch-Repaired `tapered_L` & `smec1u`, Deeper GPU (`fast_mode=0`) + Top-32 LP/Lip-SLP Polish Across Full 10-Target Pool

- **Status:** Logged before coding/testing.
- **Hypothesis:**
  1. Expanding **`hybrid45_band_v5`** to ~40% of the batch across `wb in [3, 12]`, `u_mid in [0.35, 0.65]`, independent `s in {0.982, 0.988, 0.994}` upright triangle slack, **antisymmetric corner arch `th(x,y) = 45° + alpha * (y - x) / L` (`alpha in [0, 7°]`)**, and explicit `(0,L)/(L,0)` corner-tip plugs directly explores the `T(m1) + T(m2) + arched_wb` family that hit `+0.051` on `910` from only ~11 random samples.
  2. Expanding **`tapered_L`** (which hit `+0.063` on `367` with 5% of batch) and **`smec1_v5`** (`30%` each) with `0.985..0.996` axial spacing + `1/sin(th)` wall-notch upright/tapered plugs fills the `0.86–0.90` end-wall holes seen in `367` and `971`.
  3. Exporting **top-32 diverse GPU+AL candidates** directly to `.venv` HiGHS LP (`lp_rounds=10`) + **Lip-SLP (`slp_top_k=12, slp_rounds=10`)** and running **`fast_mode=0` (`1.8x` GPU steps)** prevents `AL-LBFGS` bottlenecks from masking `-0.059` SLP improvements.
  4. Sweeping all 10 assigned `n` (`971, 367, 910, 972, 791, 969, 687, 851, 796, 333` — notably high-`f` sizes `333 (f=0.773), 687 (0.723), 796 (0.722)` plus wide-gap `972, 969, 791`) after quick validation on `{367, 910, 971}` maximizes immediate win probability.
- **Planned Verification & Fast Scope (<5 min):**
  - Implement Idea 5 builders (`seed_hybrid45_v5`, `seed_tapered_L_v5`, `seed_smec1_v5`, `--export-top 32`) in `cpp_pipeline.cpp`.
  - Test `{367, 910, 971}` (`batch=600, fast_mode=1` -> ~25s, then `fast_mode=0` -> ~40s) + `.venv` LP/Lip-SLP on top-32 (~30s).
  - Run Idea 5 across all 10 targets `{971, 367, 910, 972, 791, 969, 687, 851, 796, 333}`.

### Results Log (Idea 5)
- **Initial 3 (`367, 910, 971`, `--batch 600 --top-k 64 --fast-mode 1`, `task-613` + HiGHS LP/Lip-SLP `pos[:20], slp_top_k=8` `task-622`):**
  - `n=367`: `AL_L = 19.742064`, **`pol_L = 19.742057290312`** vs `inc_L = 19.692388155425` (`+0.049669`, `~1.95` squares short, `win=False`, `h45v5(wb=6,ar=3,diff=0)`).
  - `n=910`: `AL_L = 30.741742`, `pol_L = 30.736575323148` vs `inc_L = 30.679149464636` (`+0.057426`, `win=False`, `h45v5(wb=6,ar=3,diff=0)`; Idea 3+LP best remains `30.730560765514`, `+0.05141`).
  - `n=971`: `AL_L = 31.713203`, **`pol_L = 31.713185882324`** vs `inc_L = 31.664318325205` (`+0.048868`, **`-0.0719` jump vs Idea 3/4**, `~3.0` squares short, `win=False`, `h45v5(wb=6,ar=7,diff=0)`).
- **Remaining 7 (`--batch 480 --top-k 48 --fast-mode 1`, `task-630` + HiGHS LP/Lip-SLP `task-653` partial):**
  - `n=333`: `AL_L = 18.829704024454`, `pol_pd = 18.829646158887` vs `inc_L = 18.772634113467` (`+0.057012`, `win=False`, `h45v5(wb=4,ar=0,diff=0)`).
  - `n=687`: `AL_L = 26.790324755313`, `pol_pd = 26.779022091324` (`-0.0113` LP/SLP drop) vs `inc_L = 26.723271497397` (`+0.055751`, `win=False`, `tapLv5(K=20x21,w=7/6,Wfull)`).
  - `n=796`: `AL_L = 28.803146858124`, `pol_pd = 28.786884988604` (`-0.0163` LP/SLP drop) vs `inc_L = 28.721540002259` (`+0.065345`, `win=False`, `tapLv5(K=23x22,w=6/7,Sfull)`).
  - `n=851`: `AL_L = 29.745970`, `pol_pd = 29.741920534806` vs `inc_L = 29.684143606359` (`+0.057777`, `win=False`).
  - `n=791`: `AL_L = 28.715610`, `pol_pd = 28.701807792337` vs `inc_L = 28.634070751579` (`+0.067737`, `win=False`).
  - `n=969`: `AL_L = 31.728278`, `pol_pd = 31.683966324931` (**`-0.0443` LP/SLP drop on export idx 1**) vs `inc_L = 31.633982501360` (`+0.049984`, `win=False`, `smec1v5`).
  - `n=972`: `AL_L = 31.763916`, `pol_pd = 31.742615602892` vs `inc_L = 31.683990214555` (`+0.058625`, `win=False`).
  - **Takeaway:** LP/Lip-SLP can drop `0.01..0.044` beyond AL; top-k export matters (969 winner was export idx 1, not idx 0).
- **Structural Void & Lattice Diagnosis (Key Discovery):**
  - On `n=971`, `h45v5(wb=6,ar=7)` spontaneously collapsed its 237-square diagonal band to a **uniform `-38.200°` sheared lattice** with **`de1 = -0.1645` brick shear** (`tri1=409`, `tri2=325=T(25)`), cutting the gap from `+0.121` to `+0.0489`.
  - Void rasterization shows `971_h45v5` wastes **`6.11` area at the two `(0,0)/(L,L)` corner band ends (`3.49 + 2.62`)** plus `0.85` holes strictly along ONE (`tri1`) seam, out of `34.73` total empty area (need `31.63` at `inc_L`). Similarly `cand_v5_n367` wastes `2.91` at `(0,0)/(L,L)` + `0.84` holes along ONE seam; `cand_v5_n333` wastes `5.57` at `(0,L)/(L,0)`; `cand_v5_n851` (`tapLv5`) wastes `~20` area in an underfilled bottom arm (`2.59+2.78` corners) while its left arm is dense.

## Idea 6: Explicitly Sheared 35°–52° Bands, Corner-Clearing `u_mid` & V-Bent Bands, Phase-Wrapped Dual-Angle `L`-Strips & Basin Ruin-and-Recreate (`fast_mode=0`)
- **Status:** Logged before coding/testing.
- **Hypothesis:**
  1. **`seed_sheared_band_v6` (Direct Sheared Lattice + Full `u_mid` + Corner-Pocket Plugs):** Instead of starting at unsheared `45°` (`dth_g in ±2.3°`) and relying on `arch=7°` to accidentally shear on GPU, explicitly construct the valid brick-sheared rotated lattice (`th_b = -pi/4 + alpha` for `alpha in [-10°, +10°]` covering `35°–55°` including `(3,4,5) = 36.87°` and `38.2°`, with `v_par = (1,1)/s2 / cos(alpha)` and `v_row = e1 + sh_row * e2`, `wb in [2, 10]`), sweep **wide `u_mid in [0.12, 0.88] * L * s2`** (unlocking 1-corner-clipping bands that avoid one corner completely and leave a massive upright pentagon), allow independent `(tri1, tri2)` grid offsets/tilts (`±1.5°`, `sp in [0.985, 1.002]`), and explicitly fill `(0,0)/(L,L)` corner wedges (`±15°, ±26.5°, ±36.9°, ±45°, 0°`) + seam notches at `a=0.875`.
  2. **`seed_bent_band_v6` (Corner-Avoiding Bent `V`/`Z` Elbow Band & Twin-Corner `45°`/`38°` Pockets):** Eliminate the `3..6` dead area at `(0,0)/(L,L)` by routing a tilted `wb=3..7` band from wall `(0, y_c)` to `(x_m, y_m)` at `+38°..+45°` and bending to `(x_c, L)` or `(L, y_c)` at `-38°..-45°` (or testing symmetric 2-corner pockets at `(0,0)+(L,L)`), keeping upright walls clean.
  3. **`seed_wrapped_dual_L_v6` (SAT-Verified Phase-Wrapped Dual-Angle `L`-Strip):** Fix `tapLv5`'s underfilled second arm (`cand_v5_n851`) by filling both `Arm 1 (L x W_b)` and `Arm 2 (W_l x Ky)` using the exact SAT-verified phase-wrapped rectangular tilted lattice (`v_c=(1/cos th, 0)`, `v_r=(-sin th + k/cos th, cos th)` -> area=1.000/sq, zero parallelogram lean) independently optimized over `th1, th2 in [15°, 45°]`, plus explicit `[0, W_l] x [0, W_b]` corner block (`0°/45°/th`) and sawtooth seam plugs.
  4. **Target-Capped Basin Ruin-and-Recreate (`--warm-start-dir` + `fast_mode=0`):** Load top `v5` scratch basins (`top_n<n>_fm1.bin` — never `merged.csv`), scale linearly to `L_target = inc_L - 0.0025 .. inc_L` (`a_start = 0.90..0.95` breathing room), apply targeted ruin moves (hole-neighborhood `3.2`-radius angle flips/pair swaps, uniform band angle shifts `Δth in ±2°..5°`, high-strain square -> largest corner void relocation, upright-triangle seam sliding `±0.25`), and run `sat_lmin` at **`fast_mode=0`** (`1.8x` stages + 1800 fp64 polish steps), followed by `.venv` HiGHS LP (`lp_rounds=8`) + Lip-SLP (`slp_rounds=8, slp_top_k=6`).
- **Planned Verification & Fast Scope (<10 min):**
  - Implement Idea 6 builders + `--warm-start-dir` in `poc/swarm_inflation/poc/cpp_pipeline.cpp`, compile with `-O3 -fopenmp`.
  - Benchmark on `{367, 910, 971}` FIRST (`batch=768, fast_mode=0`, fresh + warm-start ruin-and-recreate, `<3 min` GPU+AL + `<1 min` LP/SLP polish).
  - If promising (`< +0.03` or `WIN`), scale across all 10 targets `{971, 367, 910, 972, 791, 969, 687, 851, 796, 333}`.

### Results Log (Idea 6)
- **Pre-check (exact `a=1.000` rigid capacity at `L = inc_L - 0.0005`):** exact sheared 35–55° band + 2 integer upright triangles: `367: -5`, `910: -5`, `971: -8`; SAT-verified phase-wrapped tilted strips: 1-strip `-5/-5/-5`, L-strip `-7/-9/-10`. Rigid lattice constructions alone cannot win — compliance/relaxation is required (as expected).
- **Run (`--idea 6 --batch 768 --top-k 64 --seed 6 --fast-mode 0 --warm-start-dir /tmp/v5_top`, `task-727`, ~50s GPU+AL) + HiGHS LP/Lip-SLP top-16 (`task-734`):**
  - `n=367`: `AL_L = 19.714356`, **`pol_pd = 19.714342361422`** vs `inc_L = 19.692388155425` (**`+0.021954`**, was `+0.0497`; `ruin6_btilt` warm basin; fresh `shb6(wb=6,a=40,uf=49)` reached `19.714529` independently). Band re-tilted to `50.38°` (std `0.88°`), `234 upright + 133 band`.
  - `n=910`: `AL_L = 30.724373`, **`pol_pd = 30.723210888981`** vs `inc_L = 30.679149464636` (`+0.044061`, was `+0.0514`; fresh `shb6(wb=6,a=40,uf=45,diff=0)` GPU best `30.727599`).
  - `n=971`: `AL_L = 31.713987`, **`pol_pd = 31.708784817800`** vs `inc_L = 31.664318325205` (`+0.044466`, was `+0.0489`; `ruin6_btilt`).
  - **Takeaway:** coherent band-angle tilt (`ruin6_btilt`) + explicit sheared 40° band (`shb6`) are the productive moves; `fast_mode=0` helps. Iterating ruin-and-recreate (v6 basins -> v7) is the next step. No win yet.

---

### Idea 7 — LP Dual Force-Chain Surgery + Alternating LP/SLP ↔ CUDA Ruin-and-Recreate (`fast_mode=0`) & 10-Target Expansion
- **Status:** Logged before coding/testing.
- **Diagnosis of Idea 6 Jamming:**
  - Exact HiGHS LP dual analysis on `cand_v6_n367.csv` (`19.714342`, `+0.02195`, 0.86 squares short) reveals that only **34 active pair duals + 4 wall duals** hold the entire 367-square box open via a single 6-square cross-band pinch (`298 -> 117 -> 98 -> 77 -> 54 -> 29 -> 4 -> 280 -> ... -> 137`), while 21 rattlers (`deg<=1`) and `0.85`-radius voids sit untouched elsewhere.
  - Similarly `cand_v6_n910.csv` is held up by a 6-square `42.4°` cross-band pinch (`656 -> 159..4 -> 528 -> 527 -> 526`, 108 active pairs, 184 rattlers!), and `cand_v6_n971.csv` is held open by a `13.8°` wall-to-wall tilted row chain (`38 -> 43..51`).
  - Blind random hole teleportation (`ruin6_hole`) misses the dual bottleneck chain >90% of the time, and `lip_slp_polish` moves only small local steps (`<= 0.04` rad) without evicting or flipping the pinning bridge squares.
- **Hypothesis:**
  1. **LP Dual Chain-Break Surgery (`idea7_chain_ruin.py`):** Solve fixed-angle LP + Lip-SLP on top distinct `v5`/`v6` scratch basins (`top_n*_fm*.bin` & `cand_v*_n*.csv`, never `merged.csv`) to extract exact dual shadow-price load `sq_load[i]` and top interior/corner voids `(hx, hy)`, then generate targeted surgical perturbations:
     - **(Op A) Chain-Evict Teleport:** Pick one of the top-12 highest-dual-load squares `p` (plus optionally 1 chain neighbor), move `p` into a top-3 void `(hx, hy)` at candidate grain angles, and melt (`±0.18` pos, `±0.40` rad) `r < 2.6` around both the vacated bridge site and the void destination.
     - **(Op B) Bottleneck Cluster Flip:** Select the 3–9 connected highest-load squares on the force chain, coherently flip/rotate their angles (`±15°..45°` or snap to `0°/±39°/45°`), and soften neighboring `r < 2.2` seam squares.
     - **(Op C) Cut-Plane Strip Compression:** Across a cut bisecting the highest-load bridge, pull one half inward by `Δ = -0.025..-0.060` toward `inc_L` and soften the `2.8`-wide seam strip.
     - **(Op D) Deep Band-Tilt + Dual Melt:** Scale to `L_target in [inc_L - 0.012, inc_L - 0.001]`, shift band angles `Δth in [-7°, +7°]`, and melt all `sq_load > 0.05` squares before full `0.88 -> 1.00` CUDA re-inflation (`fast_mode=0`).
  2. **Fresh Scratch Fine-Sweep (`shb7`):** Keep ~50% of every batch as pure scratch (`wb in [4, 8]`, `band_deg in [36°, 43.5°] ∪ [46.5°, 54°]`, `wrapped_dual_L_v6`, `smec1_v5`) so new basins keep entering.
  3. **Scale Idea 6+7 Across All 10 Targets:** Run `{367, 910, 971}` first (`<4 min`); iterate `v7 -> v8` on near-misses and expand `shb6/shb7 + ruin` across all 10 assigned `n` (`333, 687, 791, 796, 851, 969, 972`).
- **Planned Verification & Fast Scope (<6 min initial check):**
  - Implement `poc/swarm_inflation/poc/idea7_chain_ruin.py` + `--idea 7 --surg-dir` in `cpp_pipeline.cpp`.
  - Run `{367, 910, 971}` (`batch=1024, fast_mode=0`, `512` surgical + `512` fresh scratch) + top-16 HiGHS LP/Lip-SLP polish (`<4 min`).

### Results Log (Idea 7)
Tested `--idea 7 --batch 1024 --fast-mode 0 --elastic-L 1` (512 LP-dual surgical warm starts + 512 fresh `v6/v5` seeds) + HiGHS LP & Lip-SLP polish (`pos[:20], slp_top_k=8, lp_rounds=10, slp_rounds=10`) on `{367, 910, 971}`:
- **`n=367`**: `AL=19.713335114836`, **`pol_pd=19.713328807893`** vs `inc=19.692388155425` (**`+0.020941`**, `surg7_cut@L=19.69159`, down `-0.00101` from Idea 6; ~0.82 squares short).
- **`n=910`**: `AL=30.714951466300`, **`pol_pd=30.714908827300`** vs `inc=30.679149464636` (**`+0.035759`**, `surg7_btilt@L=30.67835`, down **`-0.00830`** from Idea 6).
- **`n=971`**: `AL=31.709461857377`, `pol_pd=31.708923519430` vs `inc=31.664318325205` (`+0.044605`, `surg7_cut@L=31.66352`).

---

## Idea 7b: Scale Idea 6 + Idea 7 Pipeline to Remaining 7 Assigned Targets (`{333, 687, 969, 851, 972, 791, 796}`)
- **Hypothesis:** Targets `{333, 687, 969, 851, 972, 791, 796}` have ONLY been tested with `Idea 5` (`fast_mode=1`, `batch=480`), where `969` (`31^2+8`) already reached `+0.04998` and `333` (`18^2+9`), `687` (`26^2+11`), `851` (`29^2+10`) are small/mid `k` where `shb6` (`sheared_band_v6`) + `wrapped_dual_L_v6` + `bent_band_v6` + `idea7_chain_ruin` surgical starts + `fast_mode=0` LP/Lip-SLP polish can make large jumps (or win outright).
- **Plan:** Run `idea7_chain_ruin.py --bin-dirs /tmp/v5_top --count 384 --polish-rounds 3` -> `/tmp/cpp_pipeline --idea 7 --batch 768 --top-k 48 --fast-mode 0 --elastic-L 1 --surg-dir /tmp/v7_warm --export-top /tmp/v7_top` -> `polish_and_certify.py` on `{333, 687, 969}` first (~3 min), then `{851, 972, 791, 796}`.
- **Criteria:** Any certified `pol_pd < inc_L` -> save `win_n<n>.csv` and hand off to `@verifier`; otherwise update baseline across all 10 assigned `n` before Idea 8.

### Results Log (Idea 7b)
*(All 7 extended-pool targets completed through CUDA + AL + HiGHS LP/Lip-SLP polish; `0` wins vs incumbents, large drops across all 7 vs Idea 5)*:
- **`n=333`**: `AL=18.818520188878`, **`pol_pd=18.818486582345`** vs `inc=18.772634113467` (`+0.045852`, `-0.01116` drop vs Idea 5; `-14.05°` `O`-framed tilted core + `0°` wall sleeves).
- **`n=687`**: `AL=26.783416786482`, `pol_pd=26.782835412327` vs `inc=26.723271497397` (`+0.059564`, `-24.49°`).
- **`n=969`**: `AL=31.681078919761`, **`pol_pd=31.681042669670`** vs `inc=31.633982501359` (**`+0.047060`**, `-0.00292` drop vs Idea 5; `-16.98°` `smec1` + side sleeves).
- **`n=972`**: `AL=31.718152855693`, **`pol_pd=31.718147261340`** vs `inc=31.683990214555` (**`+0.034157`**, `-0.02447` drop vs Idea 5; `+40.98°` sheared band).
- **`n=851`**: `AL=29.724026673799`, **`pol_pd=29.723992821712`** vs `inc=29.684143606359` (**`+0.039849`**, `-0.01793` drop vs Idea 5; `-41.10°` sheared band).
- **`n=791`**: `AL=28.679588859551`, **`pol_pd=28.679353424318`** vs `inc=28.634070751579` (`+0.045283`, `-0.02246` drop vs Idea 5; `+24.85°` tilted interior).
- **`n=796`**: `AL=28.768579078832`, **`pol_pd=28.766259427809`** vs `inc=28.721540002259` (`+0.044719`, `-0.02063` drop vs Idea 5; `+25.08°` tilted interior).

---

## Idea 8: Corner-Framed & Asymmetric-Plane Bands + Seam Phase-Conversion Surgery + Tight-Target Multi-Temp CUDA (`fast_mode=3`)
- **Diagnosis of Idea 7 Spatial & Optimizer Bottlenecks:**
  1. **Spatial vacant-area map on `cand_v7_n367.csv` (`+0.0209`, 0.825 sq short of `inc=19.692388`):** Interior of `t_low` (`81`), `t_hi` (`153=T(17)`), and `band` (`133 @ -38.86°`) has `0.00` waste. All `21.616` waste lives in: (a) blunt band tips hitting `x=0` (`~4.0` area) & `y=0` (`~3.05` area) corner-first, and (b) upper seam `band <-> t_hi` (`~7.5` area, `~0.95` per `2x2` cell, twice the lower seam `~4.5`) locked by one single `6`-square bridge `sq#358 (8.5,3.5) -> sq#231 (13.21,9.21)`. Same blunt band-tip signature appears on `n=910` (`+0.0358`) and `n=972` (`+0.0342`, `++///` defects at top-right).
  2. **Missed geometric families in `cpp_pipeline.cpp`:** `seed_sheared_band_v6` hardcoded `band_ang ≈ 45° ± 2°` and `th_b ≈ -45° ± 9.5°`, never generating asymmetric band planes `psi in {26.57°, 33.7°, 38°}` + rational slopes `±18.43°, ±26.57°, ±36.87°` (`psi=26.57°, th=-36.87°` packs `371` at `a=0.96`), never tapering `wb` or putting `0°` wall-border frames around band tips to stop corner wedge waste, and never explicitly seeding the `0°`-sleeved tilted core (`O`/`U`/`L`-frame) that `n=333` (`-14.1°`), `n=969` (`-17.0°`), and `n=971` (`+26.6°`) converged to from scratch.
  3. **CUDA `sat_lmin.cu` lock-in:** `max_L_b` allowed `2/3` of the batch to expand `+0.015..+0.040` past `sh_L_init` instead of forcing packing below `inc_L`, while `void_relocate` probed only `256` random points × `3` angles (`{0°, 45°, sth[i]}`), missing `0.85`-radius seam/corner holes on `L=20..32` (`~90%` miss rate) and never trying `±18.4°, ±26.6°, ±38.8°`.
- **Proposed Mechanisms (Idea 8):**
  1. **`sat_lmin.cu` `fast_mode=3` (Tight-Target Multi-Temp + 1024-Probe Void Relocate):**
     - Add intermediate deflate-reheat cycles (`0.955 -> 0.978 -> 0.964 -> 0.988 -> 0.976 -> 0.994 -> 0.998 -> 1.000`) with `2.2x` step budget.
     - Tighten `max_L_b = sh_L_init + {-0.0005, +0.001, +0.003, +0.008}` (`b % 4`) with lower `target_w` (`0.025 -> 0.0001`) so trajectories stay packed under `inc_L`.
     - Upgrade `void_relocate` (`scale_a <= 0.992f`, up to `4` moves/stage): `4` spatial probes per thread (`1024` probes/move) across `7` candidate angles (`0°, ±22.5°, ±26.57°, 45°, sth[w_i]`), including targeted corner/wall-bias probes (`25%` along walls/corners where `cand_v7` holes concentrate).
  2. **New Scratch Topologies in `cpp_pipeline.cpp` (`idea == 8`):**
     - `seed_framed_band_v8`: `0°` wall-flush border arms along `(0, L)` / `(L, 0)` (eliminating tilted corner-tip sawtooth waste) + tapered-width sheared band (`wb -> wb-2 -> 2` near corners) across `th_b in [-54°, -18°]`.
     - `seed_asym_band_v8`: asymmetric band plane `psi in {26.57°, 31°, 35°, 40°, 45°, 50°, 55°, 63.43°}` × rational/sheared `th_b in {±14°, ±18.43°, ±26.57°, ±33.69°, ±36.87°, ±41°, 45°}` + dual-anchored upright triangles.
     - `seed_framed_tilted_v8`: explicit `O`/`U`/`L`-frame of `fB, fT, fL, fR` upright wall layers enclosing a sheared smectic tilted interior (`11°..30°`), tailored for `333, 687, 791, 796, 969, 971`.
  3. **Targeted Seam & Corner-Tip Refold (`poc/swarm_inflation/poc/idea8_seam_surgery.py`):**
     - **Op E (Seam Phase Slide & Gap Closure):** Lock `t_low`, tangentially slide `t_hi` along `(±s, ∓s)` (`s in [-0.7, +0.7]`) AND shift inward `(-d, -d)` (`d in [0.05, 0.35]`) into the low-density seam channel, breaking single pinch bridges and scanning all inter-tooth phases.
     - **Op F (Corner-Tip Strip Refold):** Strip tilted squares in blunt corner pockets (`x < 2.2` or `y < 2.2` or corners) and re-seed flush `0°` wall columns/rows (plus `±45°`/`±26.6°` corner wedges) directly inside `L_target in [inc_L - 0.008, inc_L - 0.0008]`.
     - **Op G (Seam Stoichiometry Shift `band <-> up`):** Convert `2..8` high-dual seam squares between `band` angle and `0°` (widening/narrowing `band` by 1 row locally without leaving `inc_L`).
- **Verification & Fast Scope (<5 min):**
  - Build updated `sat_lmin.cu` + `cpp_pipeline.cpp`; validate on `{367, 910, 972}` first (`batch=1024`, `fast_mode=3`, `512` seam-surgical + `512` new `v8` scratch) + HiGHS LP & Lip-SLP polish.
  - Scale winning `v8` pipeline across best remaining targets (`971, 851, 969, 333, 791, 796, 687`).

### Results Log (Idea 8)
*(Completed across all 9 surg-enabled targets (`batch=1024` on `{367,910,972}`, `batch=768` on `{851,971,969,333,791,796}` + scratch `n=687`) with `fast_mode=3` + HiGHS LP/Lip-SLP polish; `0` wins vs incumbents, `6` targets improved over Idea 7b led by a `-0.03185` drop on `n=969`)*:
- **`n=969`**: `AL=31.649203828139`, **`pol_pd=31.649192182516`** vs `inc=31.633982501359` (**`+0.015210`**, **`-0.03185` drop** vs Idea 7b `31.681043`! Grown from scratch via `asym8(p=45,t=-27,w=4,diff=0)` into `823 up + 146 @ -26.07°` — note `31.649192` is already smaller than `inc(971)=31.664318` and `inc(972)=31.683990`, with remaining gap concentrated in period-3 seam voids `min_d <= 0.86` and a `0.649` bottom-wall column split).
- **`n=367`**: `AL=19.713231251926`, **`pol_pd=19.713223586114`** vs `inc=19.692388155425` (**`+0.020835`**, `-0.00011` drop vs Idea 7b; `surg8_stoich`).
- **`n=851`**: `AL=29.721064913408`, **`pol_pd=29.717105816792`** vs `inc=29.684143606359` (`+0.032962`, **`-0.00689` drop** vs Idea 7b; `surg8_dualtilt`).
- **`n=972`**: `AL=31.722641075952`, **`pol_pd=31.717267107638`** vs `inc=31.683990214555` (`+0.033277`, `-0.00088` drop vs Idea 7b; `surg8_stoich`).
- **`n=796`**: `AL=28.766314888467`, **`pol_pd=28.761586835840`** vs `inc=28.721540002259` (`+0.040047`, **`-0.00467` drop** vs Idea 7b; `surg8_stoich`).
- **`n=791`**: `AL=28.693022751970`, **`pol_pd=28.678921495261`** vs `inc=28.634070751579` (`+0.044851`, `-0.00043` drop vs Idea 7b).
- Unimproved vs Idea 7b: `n=910` (`pol_pd=30.716003127444`, vs `30.714909`), `n=971` (`pol_pd=31.713197338572`, vs `31.708924`), `n=333` (`pol_pd=18.829070649461`, vs `18.818487`), `n=687` (`AL=26.873380287724` scratch-only without `surg8`, vs `26.782835`).
- **Key Idea 8 Takeaways:**
  1. `fast_mode=3` multi-temp cycles + 1024-probe `void_relocate` + `asym8` unlocked a **new `~31.649` basin on `n=969`** (`-0.03185` better than `smec1`!) from **1 single lucky `(p=45, t=-27, w=4)` seed**, while most `(n, v_shift, w)` slices of `asym8` were skipped whenever `abs(diff)` missed `0` or landed far off-target.
  2. Tight `max_L_b = sh_L_init + ...` clamping in `fast_mode=3` helped `surg8` drop `n=851`/`796`/`972`/`367`, yet `n=969` still trapped period-3 Moire seam pockets (`0.81..0.86` void radius) because global inflation locks outer `0°` crystals early and cannot repack 40–180 seam squares collectively at `L < inc_L`.

---

## Idea 9: Fixed-Crystal GPU Sub-Pocket Inflation (`sat_sub.cu`) + Systematic Wall-Anchored `asym9` & Pythagorean Diamond Stepped Grids
- **Diagnosis of Bottlenecks in Idea 8 Top Basins (LP Dual Chains):**
  - HiGHS LP-dual inspection across our 4 tightest `cand_v8` basins shows only **8%–11% of squares carry any LP load**, all pinned along a single `4..6`-square tilted cross-bridge + wall tips:
    - `n=969` (`31.649192`, `+0.01521`): `60/969` active duals (`(5.5,6.15)->(10.15,3.5)` 4-sq `-26.9°` bridge), `0.649` column split in `t_lo`, period-3 seam voids `r=0.827..0.839`. Meanwhile `31.649192` already undercuts `inc(971)=31.664318` and `inc(972)=31.683990`!
    - `n=367` (`19.713224`, `+0.02084`): `35/367` active duals (`(6.21,3.5)->(0.5,8.21)` 6-sq `-38.7°` bridge) right above a `r=0.893` tip void at `(3.33, 1.29)`.
    - `n=851` (`29.717106`, `+0.03296`): `55/851` active duals (`6`-sq `+40.5°` bridge `(10.5,12.5)->(16.22,17.22)`).
    - `n=972` (`31.717267`, `+0.03328`): `62/972` active duals (`6`-sq `-40.7°` bridge `(3.5,25.51)->(8.22,31.22)`) beside a `r=0.881` corner void `(0.5, 31.22)`.
  - Meanwhile, rational approximation of `per_n_gap.csv` targets reveals Pythagorean denominators (`n=367: 19+9/13` `(5,12,13)`, `n=910: 30+11/17` & `n=687: 26+12/17` `(8,15,17)`, `n=333: 18+19/25` `(7,24,25)`, `n=796: 28+7/10` `(3,4,5)`), while `seed_asym_band_v8` anchored `t_lo`/`t_hi` inconsistently and sampled only `~20%` of scratch seeds (`~77` draws over ~3,920 `(psi, th, wb, u_frac)` slots).
- **Proposed Mechanisms (Idea 9):**
  1. **9A — Fixed-Crystal Sub-Pocket GPU Inflation (`poc/swarm_inflation/poc/sat_sub.cu` + `idea9_subpocket.py`):**
     - Freeze `N_fix = N - N_mob` flawless deep-interior `0°` crystal squares (shifting any `(L, L)`-anchored upper-right block cleanly by `ΔL = L_target - L_old`, checking zero frozen-frozen overlap), leaving `N_mob ~ 90..220` squares around the LP dual bridge, band, corner tips, and largest voids mobile.
     - Run `B = 1024..2048` parallel sub-pocket trajectories on RTX 4070 Super inflating ONLY `N_mob` from `a: 0.72 -> 1.00` (with 3 deflate-reheat cycles, free angle exploration `0°/±22.6°/±26.6°/±28.1°/±36.9°/±45°`, wall-tip & void teleportation, and tight fixed `L_target = inc_L - delta`).
     - Also run `+2`/`+3` from-scratch stoichiometry lifts on our `n=969` `31.649` basin (`969 -> 971` at `L < 31.664`, `969 -> 972` at `L < 31.684`).
     - Any sub-pocket run reaching `lam <= 1.0005` passes directly to AL-LBFGS + HiGHS LP & Lip-SLP certification (`polish_and_certify.py`).
  2. **9B — Systematic Wall-Anchored `asym9` + Pythagorean Stepped-Grid Diamond Clusters (`cpp_pipeline.cpp --idea 9`):**
     - Fix `seed_asym_band_v9` so `t_lo` anchors cleanly to `(0, 0)` and `t_hi` anchors cleanly to `(L, L)`, allocate 55% of scratch to `asym9`, and sweep rational/Pythagorean angles (`-14.04°, -16.26°, -18.43°, -22.62°, -26.57°, -28.07°, -30.96°, -33.69°, -36.87°, -40.5°`) × `psi in {45°, 36.87°, 31°, 26.57°}` × `wb in 2..7` × `u_frac`.
     - Add `seed_pyth_diamond_v9`: stepped `0°` grids partitioned by `1..6` anti-diagonal / diagonal clusters of Pythagorean `(3,4,5)`, `(5,12,13)`, `(8,15,17)`, `(7,24,25)`, `(1,2,sqrt5)` tilted squares (25% of scratch), keeping `fband8/ftilt8/shb6/wrapL6` (20%) + `v8` warm-starts.
- **Verification & Fast Scope (<5 min):**
  - Test **9B** first (`~15s/n`) on `{969, 971, 972, 367, 910, 851}` (`batch=1024, fast_mode=3, elastic_L=1`) + HiGHS LP/Lip-SLP.
  - Build & unit-test `sat_sub.cu` + `idea9_subpocket.py` on `{969, 367, 971, 972, 851, 910}` (including `969->971/972` lifts), then scale wins/improvements across remaining targets.

### Results Log (Idea 9)
- **Idea 9A (`sat_sub.cu` + `idea9_subpocket.py`)**:
  - Diagnosed and fixed two root causes in fixed-crystal sub-pocket inflation: (1) unnormalized `2*pen` gradient descent stalled at `pen ~ 0.061` (`lam ~ 1.064`), replaced with thread-local Adam (`b1=0.85, b2=0.99`, cosine decay) + `mode=0` gentle (`a: 0.960 -> 1.000`, `w_align=0`) and `mode=2` deep (`a: 0.895 -> 1.000`) schedules, dropping unperturbed `n=969` residual penetration 130x to `4.70e-4` (`lam=1.000471`); (2) added connected single-hole ball selection (`focus_hole >= 0`) and gentle stoichiometry-preserving perturbations.
  - Across `n in {969, 971, 972, 367, 851, 910, 796}` (`B=512` per config), freezing the upright crystal at `L_target < inc_L` gave `pen = |dL| / sqrt(2) = 0.0106..0.0115` (`eff_L = 31.97` on `969`, `19.85` on `367`, `29.95` on `851`) and `pen = 0.040` on `+2/+3` stoichiometry lifts (`969 -> 971/972`), proving that rigid frozen boundaries prevent global `0°` crystal sliding and seam re-partitioning (`sat_lmin_cuda --elastic-L 1` required).
- **Idea 9B (`cpp_pipeline.cpp --idea 9` + HiGHS LP/SLP polish `pol_v9b_1` & `pol_v9b_2`)**:
  - Fixed `slot_hint` stride (`gcd(288, 20) = 4` -> sequential `asym_slot = grp * 11 + fam` and `pyth_slot = grp * 5 + (fam - 11)`) so all 288 `(psi, th, w)` wall-anchored `asym9` configs and 24 `pyth9` configs are swept uniformly across `B=1024`.
  - **5 new all-time bests for our from-scratch pipeline** (verified via `squares.core.certify` and pandas roundtrip):
    - **`n=791`**: `AL=28.663066036371` -> **`pol_pd=28.658853445488`** vs `inc=28.634070751579` (**`+0.024783`**, **`-0.020068` drop** from Idea 8's `28.678921`, via `asym9(p=45,t=-28,w=4,diff=0)`)
    - **`n=972`**: `AL=31.716515784024` (scratch) / `31.720801802524` (surg) -> **`pol_pd=31.708429178278`** vs `inc=31.683990214555` (**`+0.024439`**, **`-0.008838` drop** from Idea 8's `31.717267`, via `surg9_stoich@L=31.68339`)
    - **`n=971`**: **`pol_pd=31.711910858759`** vs `inc=31.664318325205` (**`-0.001286` drop** from Idea 8's `31.713197`)
    - **`n=333`**: `AL=18.825859707092` -> **`pol_pd=18.822641723285`** vs `inc=18.772634113467` (**`-0.006429` drop** from Idea 8's `18.829071`, via `asym9(p=45,t=-18,w=6,diff=0)`)
    - **`n=687`**: `AL=26.806819439220` -> **`pol_pd=26.804722490408`** vs `inc=26.723271497397` (**`-0.068658` drop** from Idea 8's `26.873380`, via `asym9(p=45,t=-37,w=6,diff=0)`)
    - Unchanged prior bests: `n=969: 31.649192182516` (`+0.015210`), `n=367: 19.713223586114` (`+0.020835`), `n=851: 29.717105816792` (`+0.032962`), `n=910: 30.716003127444` (`+0.036854`), `n=796: 28.761586835840` (`+0.040047`).

---

## Idea 10: Gentle Elastic-L Full-Crystal Breathing (`fast_mode=2`, `a: 0.945 -> 1.000`) on Cross-`n` Stoichiometry Transfers (`969 <-> 971 <-> 972`) + Flush-Sheared (`du = tan(theta)`) Pythagorean Bicrystals

### Hypothesis & Motivation
1. **Why Idea 8/9 `fast_mode=3` (`a: 0.880 -> 1.000` + random void teleports) under-exploited `cand_v8_n969` (`31.649192 < inc(971)=31.664318 < inc(972)=31.683990`) and `cand_v9_n791` (`28.658853`):**
   - `cand_v8_n969.csv` (`31.649192`) was discovered at the end of Idea 8, AFTER `971` and `972` had already run, and contains 5 seam holes of radius `r = 0.850..0.866` along the `+45°` boundary between `822` upright squares and `147` `-26.07°` tilted squares.
   - When `fast_mode=3` melts a surgical/stoichiometric candidate down to `a = 0.880` (`12%` linear shrinkage) with `lr = 0.016` and random void jumps, it scrambles the pristine `822`-square `0°` crystal and `-26.07°` band. Conversely, `Idea 9A` (`sat_sub_cuda`) froze the `0°` crystal rigidly so `t_lo` and `t_hi` could not slide or breathe.
   - In `sat_lmin.cu`, `fast_mode=2` (`stages_gentle`: `a: 0.945 -> 0.962 -> 0.976 -> 0.986 -> 0.992 -> 0.996 -> 0.9985 -> 1.000`, `lr: 0.0055 -> 0.00005`, `opt_L=1` from stage 0, **zero random teleports**) lets the entire `0°` crystal and tilted band breathe and slide smoothly without losing crystal order!
2. **Flush-Sheared (`du = ±tan(theta)`) Tilted Bands & Stepped 4-Quadrant (`SW(0,0), NE(f,f), NW(0,f), SE(f,0)`) Pythagorean Seeds (`Idea 10B`):**
   - A tilted lattice at angle `theta` (`|tan(theta)| < 1`) sheared row-by-row by `du = +tan(theta)` (`dv = 1.0`) maintains exact `max(|du|, |dv|) = 1.0` non-overlap while making the step vector `(dx, dy) = (0, sec(theta))` **100% flush (`dx = 0`) against a vertical `0°` crystal wall** (eliminating staircase sawtooth waste!).
   - Furthermore, for rational targets (`687: 26+12/17`, `910: 30+11/17` from `(5,12,13)` with `a+b=17`; `333: 18+19/25` from `c=25`; `796: 28+7/10` from `(3,4,5)`), partitioning the `0°` background into 4 shifted quadrants `SW(0,0), NE(f,f), NW(0,f), SE(f,0)` leaves an exact width-`f` / `1+f` L-stepped channel filled by `du`-sheared tilted clusters.

### Plan & Validation
- **Track 10A (`poc/swarm_inflation/poc/idea10_gentle_transfer.py` + `cpp_pipeline --fast-mode 2`)**:
  - Build a dedicated 1024-trajectory surgical & cross-`n` transfer pool from our tightest polished basins (`cand_v8_n969: 31.649192`, `cand_v9_n972: 31.708429`, `cand_v9_n791: 28.658853`, `cand_v8_n367: 19.713224`, `cand_v8_n851: 29.717106`, `cand_v7_n910: 30.716003`, `cand_v8_n796: 28.761587`, `cand_v9_n333: 18.822642`, `cand_v9_n687: 26.804722`):
    1. **Cross-`n` seam-hole insertion (`969 -> 971` `+2`, `969 -> 972` `+3`, `971 -> 972` `+1`)** and **high-dual seam removal (`972 -> 971` `-1`, `972 -> 969` `-3`, `971 -> 969` `-2`)** placing new squares directly at the exact `r = 0.85..0.866` seam voids with local neighbor backoff.
    2. **Same-`n` LP-dual bridge slide, band-angle micro-rotation (`±0.4°..2.0°`), and `0°`-to-tilted boundary conversion** (converting 1–4 `0°` squares adjacent to `r > 0.82` seam holes into tilted band squares so `t_lo`/`t_hi` can step inward by `0.02..0.04`).
  - Run through `sat_lmin_cuda` under **`fast_mode=2` (`a: 0.945 -> 1.000` gentle elastic-L)** + C++ OpenMP AL polish + HiGHS LP/SLP polish.
- **Track 10B (`seed_twin_rect_pockets_v10` & `seed_flush_sheared_v10` in `cpp_pipeline.cpp --idea 10`)**:
  - Implement **Twin Off-Diagonal Rectangular Pockets (`NW` at `+theta`, `SE` at `-theta` or `+theta`, `theta in {22.62°, 24.5°, 25.5°, 26.57°, 28.07°, 31.0°, 36.87°, 40.5°}`)** carved out of a `SW + NE` corner-anchored `0°` crystal (with outer `0°` wall frame `m in {0, 1, 2}` and optional row shear `du_shear in {0, ±tan(theta), ±sin(theta)}`), eliminating the long `45°` diagonal sawtooth interface on `n=910, 969, 971, 972, 851, 791, 796, 687, 367, 333`.
  - Also implement flush-sheared (`du = ±tan(theta)`) diagonal bands and 50% surgical/transfer pool integration.

### Results Log (Idea 10)
- **Track 10B (`seed_twin_rect_pockets_v10` + `seed_flush_sheared_v10` under `fast_mode=3`, `batch=640`, `s=1010` + HiGHS LP/SLP `pol_v10b_1`)**:
  - Dropped **`n=367`** from scratch to `AL_L = 19.707263438429` -> **`pol_pd = 19.707251703133`** (`-0.005972` below Idea 8's `19.713224`, via `asym9(p=45,t=-37,w=2,diff=0)`).
  - Twin off-diagonal rectangular pockets (`twin10`) reached `gpu_L = 19.992` (`n=367`), `31.119` (`n=910`), `32.164` (`n=971`), confirming single diagonal bands remain tighter than 2-pocket split interfaces at these `n`.
- **Track 10A (`idea10_gentle_transfer.py` + `cpp_pipeline --idea 10 --fast-mode 2 --elastic-L 1` across 3 monotonic ratchet rounds `s=1020, 1025, 1030` + HiGHS LP/SLP polish)**:
  - Switching surgical/stoichiometric refinement from `fast_mode=3` (`a: 0.880 -> 1.000` with void teleports) to **`fast_mode=2` (`a: 0.945 -> 1.000`, `opt_L=1`, zero teleports)** + dual-grain inward slide (`op=4`) improved **ALL 10 ASSIGNED TARGETS simultaneously** (`cand_v10_n*.csv`, all passing literal `squares.core.certify` and `pandas` roundtrip with `0.0` overlap):
    - **`n=791`**: `28.658853` -> **`28.646799512923`** vs `inc = 28.634070751579` (**`+0.012729` gap**, **`-0.012054` drop** via `surg10_dual`)
    - **`n=969`**: `31.649192` -> **`31.647466974754`** vs `inc = 31.633982501359` (**`+0.013484` gap**, **`-0.001725` drop** via `surg10_dual`)
    - **`n=367`**: `19.713224` -> **`19.707112761322`** vs `inc = 19.692388155425` (**`+0.014725` gap**, **`-0.006111` drop** via `surg10_glide`)
    - **`n=972`**: `31.708429` -> **`31.707132714296`** vs `inc = 31.683990214555` (**`+0.023142` gap**, **`-0.001296` drop** via `surg10_dual` from `n=969`)
    - **`n=851`**: `29.717106` -> **`29.715497178580`** vs `inc = 29.684143606359` (**`+0.031354` gap**, **`-0.001609` drop** via `surg10_glide`)
    - **`n=910`**: `30.716003` -> **`30.714612203134`** vs `inc = 30.679149464636` (**`+0.035463` gap**, **`-0.001391` drop** via `surg10_dual`)
    - **`n=796`**: `28.761587` -> **`28.757132285967`** vs `inc = 28.721540002259` (**`+0.035592` gap**, **`-0.004455` drop** via `surg10_glide` + SLP)
    - **`n=971`**: `31.711911` -> **`31.707477769346`** vs `inc = 31.664318325205` (**`+0.043159` gap**, **`-0.004433` drop** via `surg10_glide` from `n=969`)
    - **`n=333`**: `18.822642` -> **`18.818682766685`** vs `inc = 18.772634113467` (**`+0.046049` gap**, **`-0.003959` drop** via `surg10_glide` + SLP)
    - **`n=687`**: `26.804722` -> **`26.776196108364`** vs `inc = 26.723271497397` (**`+0.052925` gap**, **`-0.028526` drop** via `surg10_dual` + SLP)

---

## Idea 11: Symmetric Box-Squeeze & Split-Half-Crystal CUDA Kernel (`fast_mode=4`, `a: 0.965 -> 1.000`) + Active LP-Dual Force-Chain Buckling & Slip-Line Surgery

### Hypothesis & Motivation
1. **Root Cause of the `+0.0127 .. +0.0147` Floor in `sat_lmin.cu` (`fast_mode=2` + `polish_and_cert_fp64`):**
   - Inspecting the Round 2/3 CUDA logs on `n=791` (`sh_L_init = 28.63357 < inc_L` -> `best_gpu_L = 28.647571`, `lam = 1.000023`, `pen = 2.30e-5`), `n=969` (`sh_L_init = 31.63348 < inc_L` -> `best_gpu_L = 31.648089`, `lam = 1.000023`), and `n=367` (`sh_L_init = 19.69189 < inc_L` -> `best_gpu_L = 19.707142`, `lam = 1.000003`) reveals why `best_gpu_L` consistently lands `+0.0140 .. +0.0152` above `sh_L_init` even when overlap `pen` is practically zero (`~3e-6 .. 2e-5`):
     1. **One-way upward box ratchet in `inflate_sat_kernel` (`line 581`)**: `err_w = (sum_wall - target_w) / fmaxf(target_w, 0.015f)`. In the late stages (`a >= 0.996`), `target_w` decays to `0.0001f` while the denominator stays `0.015f`. Thus when `sum_wall == 0` (the box is larger than the crystal!), `err_w = -0.0001 / 0.015 = -0.0067` (zero inward contraction), whereas any transient wall contact (`sum_wall = 0.015`) produces `err_w = +1.0` (full-speed expansion up to `max_L_b`)!
     2. **One-way upward box expansion in `polish_and_cert_fp64` (`lines 942–944`)**: `err_w = (swall - 1.0e-4) / 0.005` has a 50x upward asymmetry (`err_w = -0.02` when `swall == 0`, vs `+1.0` when `swall = 0.0051`) and allows `sh_L` to grow by up to `+0.015` (`0.4 * lr * 900 = +0.011`) during float64 polish!
     3. **Coupled `g_hi U g_lo` upright cluster in `sat_lmin.cu` collective moves (`line 700`)**: When `seed_idx` is an upright square (`th ~ 0`), `|cos(2*(sth[i] - seed_th))| > 0.96f` selects **both** `g_hi` (NW `0°` crystal) and `g_lo` (SE `0°` crystal) together; their opposite wall forces (`vL/vT` vs `vR/vB`) cancel out and prevent `g_hi` and `g_lo` from gliding inward toward each other!
2. **Active LP-Dual Force-Chain Buckling & Slip-Line Surgery (`idea11_force_chain.py`):**
   - At a local minimum of $L$, a sparse force chain of ~35–65 squares carries 100% of the nonzero LP dual multipliers (`sq_load > 0`) from wall to wall across the tilted seam band. Perturbing zero-dual rattlers cannot lower $L$, while melting to `a = 0.880` destroys the crystal.
   - Targeting the **active LP force chain (`sq_load > 0.02`)** with transverse buckling kicks, slip-line shear across the peak-dual bottleneck, and bottleneck-to-hole vacancy hopping — relaxed under `fast_mode=4` (`a: 0.965 -> 1.000` with symmetric inward wall pressure and split-half-crystal `g_hi`/`g_lo` glide) — directly attacks the load-bearing arches keeping `L > inc_L`.

### Plan & Validation
- **Track 11A (`fast_mode=4` in `poc/swarm_inflation/poc/sat_lmin.cu`)**:
  - Add `stages_micro` (`fast_mode == 4`, 12 stages: `a: 0.965 -> 0.975 -> 0.983 -> 0.989 -> 0.982 -> 0.991 -> 0.995 -> 0.9975 -> 0.999 -> 1.000 x 3`, `lr: 0.0038 -> 0.00004`, `opt_L = 1` from stage 0, no random teleports).
  - Fix the `sh_L` box-pressure controller in `fast_mode == 4` so `sum_wall < target_w` normalizes by `fmaxf(target_w, 1e-5f)` (giving full `-1.0` inward squeeze whenever `sum_wall == 0`), cap `max_L_b = sh_L_init + {-0.0002, +0.0008, +0.0025, +0.0060}`, and fix `polish_and_cert_fp64` so `swall < 1e-4` contracts inward (`denom = 2e-4`) and caps `sh_L <= sh_L_init + 0.0015`.
  - Split the upright `0°` cluster in `cm == 0` across the box diagonal (`(x - cx0) - (y - cy0)` or `(x - cx0) + (y - cy0)`) so `g_hi` and `g_lo` translate as independent rigid half-crystals along their net wall+seam forces.
- **Track 11B (`poc/swarm_inflation/poc/idea11_force_chain.py` + multi-round `fast_mode=4` & HiGHS LP/SLP ratchet)**:
  - Implement 5 force-chain & slip-line surgical operators (`op=0`: force-chain transverse buckle; `op=1`: slip-line shear through peak-dual bottleneck; `op=2`: max-dual bottleneck to max-void vacancy hop; `op=3`: dual-weighted anisotropic inward squeeze + rattler void drift; `op=4`: cross-`n` stoichiometry transfer + dual-grain glide) and run across all 10 targets (`791, 969, 367, 972, 971, 851, 910, 796, 333, 687`).

### Results Log (Idea 11)
- Ran fresh `fast_mode=4` (`task-1607`, `batch=1024`) + fixed surgical force-chain Rounds 1–2 (`s=1110, 1115`, `task-1638`) + 10-round HiGHS LP / 10-round SLP polish across all 10 targets (`0` wins, all 10 improved strictly over Idea 10, pushed right up against exact topology floors):
  - `791`: `28.646711798378` vs `inc=28.634070751579` (`gap=+0.012641`, `-0.000063`) — sitting on the `w=4, (1+f)^2+(3+f)^2>=16 -> f_min=sqrt(7)-2=0.645751` algebraic floor (`L_min=28.645751`).
  - `969`: `31.646763931217` vs `inc=31.633982501359` (`gap=+0.012781`, `-0.000703`) — sitting on the exact same `w=4, f_min=0.645751` floor (`L_min=31.645751`).
  - `367`: `19.707109910174` vs `inc=19.692388155425` (`gap=+0.014722`, `-0.000002`) — `w=1` `45°` band floor (`k + sqrt(2)/2 = 19.707107`).
  - `972`: `31.707117598066` vs `inc=31.683990214555` (`gap=+0.023127`, `-0.000007`) — `w=1` `45°` band floor (`31.707107`).
  - `851`: `29.707126768156` vs `inc=29.684143606359` (`gap=+0.022983`, **`-0.008364`**) — `w=1` `45°` band floor (`29.707107`).
  - `910`: `30.707136765186` vs `inc=30.679149464636` (`gap=+0.027987`, **`-0.007475`**) — `w=1` `45°` band floor (`30.707107`).
  - `971`: `31.707141323055` vs `inc=31.664318325205` (`gap=+0.042823`, `-0.000330`) — `w=1` `45°` band floor (`31.707107`).
  - `796`: `28.755514088565` vs `inc=28.721540002259` (`gap=+0.033974`, **`-0.001366`**)
  - `333`: `18.816719150203` vs `inc=18.772634113467` (`gap=+0.044085`, **`-0.001964`**)
  - `687`: `26.772431306882` vs `inc=26.723271497397` (`gap=+0.049160`, **`-0.003544`**)
- **Diagnostic Proof of Why Local Basin Search Cannot Beat `inc_L`:** Force-chain and active-wall audits on `cand_v11` proved `791`/`969` have a single `w=4` tilted bridge `(1+f, 3+f)` whose Cauchy-Schwarz bound `(1+f)^2+(3+f)^2 >= 16` forces `f >= sqrt(7)-2 = 0.645751 > f_inc (~0.634)`, while `367, 851, 910, 971, 972` hit the `w=1` `45°` band floor `sqrt(2)/2 = 0.707107 > f_inc`. Idea 12 must therefore change global topology cleanly via higher-width exact bridge seeds (`w=3/5/8/9`) and true flush sheared-row wall L-pockets (`W=P+f`).

---

## Idea 12: Mitered Flush Sheared-Row L-Pockets & Exact Algebraic Bridge Roots (`w=3/5/8/9`) (2026-11-13)
### Hypothesis
Idea 11 proved mathematically that local force-chain relaxations around our `v10`/`v11` basins are pinned against hard discrete floors (`w=4` -> `f_min = sqrt(7)-2 = 0.645751` on `791`/`969`, and `w=1` -> `0.707107` on `367, 851, 910, 971, 972`). Two global topologies directly bypass both floors at `L < inc_L`:
1. **Track 12A — Mitered & End-Tapered Flush Sheared-Row Wall L-Pockets (`seed_mitered_Lpockets_v12`)**:
   - Previous Idea 3–4 L-pocket attempts (`seed_smectic1_unit`, `seed_tapered_L_unit`) failed because they left West/South strips un-mitered at `(0, 0)` (losing `6..12` squares), inserted perpendicular `0°` end-fan rows (`nf >= 1`) that collided with sheared rows, tested only `w = floor(W) + 1`, packed with `fill_wedges_and_complete(..., 0.945)` junk fillers, and ran only under early `fast_mode=0/1`.
   - Exact Python SAT profiling at `L = inc_L - 0.002` confirms that sharing the SW corner along the diagonal miter `y/W2 = x/W1` with `s1 * s2 = -1` (aligned lattice orientation across West and South strips) and pre-compression `a_lat in {0.978, 0.988, 0.996}` yields `>= n` non-overlapping squares (`+1` to `+3` above `n` on `791, 971, 972` even before end-triangle wedge extensions!).
   - We therefore enumerate `(Kx, Ky, P1, w1, P2, w2, s1, s2, a_lat)` with `P1, P2 in [3..14]` and `w in {P+1, P+2}`, fill `(0, L)`, `(L, 0)`, and `(0, 0)` corner wedges using `0°` crystal extensions + strip-angle probes at `a_sat = 0.892`, rank by exact `raw_cnt` match to `n`, and relax via `fast_mode=3` (`0.880 -> 1.000` with neighbor-angle alignment and elastic `sh_L`).
2. **Track 12B — True-Axis Exact Algebraic Bridge Roots (`seed_exact_bridge_v12`)**:
   - `seed_asym_band_v9` stopped at `w <= 7`, never included `w=8, 9` roots (`w=8 (5,5) @ 45° -> f=0.65685 < f_inc(972,851,910,367)`; `w=8 (4,6) -> 0.56776`; `w=8 (2,7) -> 0.57445`; `w=9 (2,8) -> 0.61249`; `w=9 (4,7) -> 0.68466`; `w=5 (3,3) @ 45° -> 0.53553`; `w=3 (1,2) -> 0.56155`; `w=5 (1,4) -> 0.70156`), and stepped rows along band direction `psi` instead of the square lattice axis `th_b`.
   - `seed_exact_bridge_v12` directly targets all feasible `(a, b, w)` roots with `f_root < f_inc`, steps band rows along the true square axis `(cos th_b, sin th_b)`, pairs with slight normal-tilt offsets (`psi - th_b in [-4°, +4°]`), and fills sawtooth staircase seams with angle-matched tooth plugs at `a_sat = 0.885` before `fast_mode=3` + `fast_mode=4` + HiGHS LP/SLP polish.
- Validate first on 4 core targets (`791, 971, 367, 972`, `batch=1024`, `<10 min`) and check whether `cand_v12` breaks into lower-`f` basins (`< 0.6458` / `< 0.7071`) before scaling to all 10 targets.

### Results Log (Idea 12)
- **4-Target POC (`n in {791, 971, 367, 972}`, `batch=1024`, `fast_mode=3`, `elastic_L=1`, `seed=1200`) — FAILED (`0` wins; bloated `lam=1.003..1.007`):**
  - `791`: `cand_v12 = 28.701860530673` (10-round HiGHS LP/SLP -> `28.689196`) vs `inc = 28.634070751579` (`lpok12(P=6/6,w=7/7,a=0.992,pre=788)`)
  - `971`: `cand_v12 = 31.771630` (LP -> `31.740250`) vs `inc = 31.664318325152`
  - `367`: `cand_v12 = 19.782134` vs `inc = 19.692388155425`
  - `972`: `cand_v12 = 31.810909` vs `inc = 31.683990214555`
- **Root-Cause Diagnostics (`cand_v12` audit + exact SAT/area-budget sweeps):**
  1. **Area budget (`L^2 - n`) proves `P=7 (w=8)` L-pockets lose ~18.5 squares at un-tapered corners (`NW, SE, SW`):** At `P=7`, bulk strip waste already consumes `~23.7` of `31.6` total empty area (`n=971, 972`), leaving `< 8.6` area for all three corners. Naive `L`-strip clipping at `a=1.000` lost `5..7` squares (`965/972`, `964/971`, `360/367`), while stuffing missing squares via `0.882` random probes (`fill_wedges_and_complete`) jammed `1-2` boundary/interface outliers that bloated `lam=1.003..1.007`. Also `raw_est` favored `P=5, 6` (`rho=0.91..0.93`) over `P=7, 8, 9` (`rho=0.945..0.955`), and `s1=+1, s2=-1` was strictly inferior to `s1=-1, s2=+1`.
  2. **Bridge orientation mismatch:** `seed_exact_bridge_v12` aligned band tangent `psi = th_r +- 4°` instead of the `+-45°` diagonal corridor (`psi = +-45°, th = +-th_r`), whereas `cand_v11_n791`/`n969` (`0.6467`) and `cand_v11_n333` (`18.8167`) proved sheared `th_r` supercells inside an asymmetrically-shifted `+-45°` corridor are the tight seam geometry.
  3. **Untapped `0.7071` & `w >= 4` bridge headroom:** Every target has strictly sub-`inc_L` algebraic bridge roots (`971: (8,5,5) 0.65685, (4,1,3) 0.64575`; `972/851/910: (13,8,9) 0.67878, (11,2,10) 0.67083, (8,5,5) 0.65685`; `367: (9,4,7) 0.68466, (8,5,5) 0.65685`; `791/969: (9,2,8) 0.61249, (12,5,10) 0.60864, (15,10,10) 0.60660`; `796/687: w=1..4 @ 45° 0.70711, (5,1,4) 0.70156, (6,3,4) 0.71308`; `333: (14,5,12) 0.76013, (15,2,14) 0.74643`). Meanwhile `seed_asym_band_v9` stopped via `if (diff >= -1 && diff <= 1) break;` on `w=1` (`0.7071`) before ever generating `w >= 4` bands.

---

## Idea 13: Exact-Root `+-45°` Diagonal-Corridor Bridges (`w>=3` / `796,687` `0.7071`) & Corner-Fanned `P>=7` L-Pockets (2026-11-13)
### Hypothesis
Combining the root-cause findings of Idea 11 and Idea 12:
1. **Track 13A (`seed_asym_bridge_v13`, 65% of batch):**
   - Enumerates all exact algebraic bridge roots `(w_b, a, b)` (`w_b in [1..18]`, `(a+f)^2+(b+f)^2=w_b^2`) strictly below `inc_f - 1e-4`.
   - When `inc_f < 0.7071` (`367, 910, 971, 972, 851, 791, 969, 333`), explicitly excludes `w <= 2 @ 45°` so seeds never settle into the `0.707107` `w=1` trap. Conversely, for `796 (inc_f=0.72154)` and `687 (inc_f=0.72327)`, directly targets `w=1..5 @ 45°` (`0.70711`) and `(5,1,4)` (`0.70156`) / `(6,3,4)` (`0.71308`), which Idea 10/11 never generated cleanly after shifting away from `asym9`.
   - Places the `th_r` sheared crystal inside an asymmetrically shifted `+-45°` corridor (`psi = +-45° +- 8°`, plus `30%` true-axis `psi = th_r +- 3°`), supporting both 2-corner and 3-corner `0°` anchorings (`cand_v11_n969` style), sweeping `(shift_d, wb, sp_u, a_seed)` until `diff in [-2, +1]` without early `w=1` exit, followed by seam void filling.
2. **Track 13B (`seed_fanned_Lpockets_v13`, 35% of batch):**
   - Fixes `seed_mitered_Lpockets_v12` by restricting `P >= 7` (`k >= 29`, else `>= 5`) where `rho_shear >= 0.944`, prioritizing `(s1, s2) = (-1, +1)` and `(+1, -1)`, enforcing exact outer `hx = 0.5*(c+s)` wall margins, adding continuous corner fanning (`nf = 0, 3, 5`) + `0°` staircase wedge teeth before `0.892` corner completion.
- Test first on 4 targets (`796, 687, 971, 367`, `batch=1024`, `fast_mode=3`, `<5 min`) + Round 2 `Idea 11` surgery + `fast_mode=4` + 10-round HiGHS LP/SLP polish; scale to all 10 targets on promising results.

### Results Log (Idea 13)
- **Status:** `0` strict wins (`L_lit < inc_L`), `2` all-time `swarm-inflation` personal bests (`n=687`, `n=796`), committed on `swarm-inflation`.
- **Pipeline:** Added `seed_asym_bridge_v13` (strict sub-`inc_f` `(w,a,b)` roots with `45°` `0.7071` gated by `inc_f > 0.7072`, band-aligned shear `sh = -cot(th-psi)`, `nx/ny`-matched corner anchors) + `seed_fanned_Lpockets_v13` (`P >= 7` for `k >= 29`, `(s1,s2)` opposite-shear bias, exact `hx = 0.5*(c+s)` wall margins, miter corner fanning, `0°` end-wedge teeth).
- **Core 4 Targets (`796, 687, 971, 367` — Round 1 `fm3` `batch=1024` + `idea11_force_chain` surgery -> Round 2 `fm4`/`fm2` + 10-rd HiGHS LP/SLP):**
  - `n=687` (`inc=26.723271`): **NEW BEST `26.744554107875`** (`v13_r2:fm4` + SLP10, down `-0.0290` from `v11_surg` `26.773585`; gap `0.0213`). Structure: genuine `P=7, w=8` opposite-shear `(23°, 67°)` L-pocket (`382 @ 0°`, `137 @ 67°`, `126 @ 23°`).
  - `n=796` (`inc=28.721540`): **NEW BEST `28.748227708142`** (`v13_r2:fm4` + SLP10, down `-0.0078` from `v11_surg` `28.755991`; Round 1 `brg13(w=6,3_4)` hit `28.767601` and `fanL13` hit `28.766607 -> 28.751129`).
  - `n=971` (`inc=31.664318`): Round 1 `fanL13(P=8,9)` `31.741379 -> 31.717119` (SLP10), `brg13(w=4,1_3)` `31.761635`; Round 2 pulled back to `31.707131` (`45°` `v11_top` donor).
  - `n=367` (`inc=19.692388`): Round 1 `fanL13(P=6,6)` `19.740413 -> 19.737964` (SLP10), `brg13(w=4,1_3)` `19.749446`; Round 2 `19.707110` (`45°`).
- **Remaining 6 Targets (Round 1 `fm3` `batch=1024`):**
  - `n=791` (`inc=28.634071`): `28.703882` (`fanL13(P=9,6)`); `n=969` (`31.633983`): `31.714716` (`fanL13(P=9,9)`); `n=972` (`31.683990`): `31.735638` (`fanL13(P=9,7)`); `n=851` (`29.684144`): `29.737313` (`fanL13(P=9,7)`); `n=910` (`30.679149`): `30.759441` (`fanL13(P=8,8)`); `n=333` (`18.772634`): `18.806003` (`fanL13(P=5,5)`).
- **Key Bottleneck Diagnostics:**
  1. Void audit of `cand_v13_n687` & `cand_v13_n796` shows largest remaining voids (`r=0.79..0.82`) sit strictly at the `(0, L)`/`(L, 0)` outer L-pocket strip ends (`(4.46, 26.25)`, `(26.25, 3.47)`) and `(0,0)` corner miter (`(1.49, 2.48)`).
  2. `lip_slp_polish` in `polish_and_certify.py` uses a symmetric worst-case `+LIP * |dtheta|` Lipschitz penalty on every pair, which mathematically blocks cooperative small-angle rotation `(dx, dy, dtheta)` of locked bridge/pocket chains.

---

## Idea 14: Exact-Jacobian Joint `(x, y, dtheta)` HiGHS SLP + Gentle GPU Breathing (`0.94 -> 1.00`) + End-Matched `24.8°–26.6°` Twin/L-Pockets (2026-11-13)
- **Hypothesis:**
  1. Replacing `lip_slp_polish`'s symmetric `+LIP * |dtheta|` bound with **exact signed first-order SAT-margin derivatives `(dx, dy, dtheta_i, dtheta_j, dL)`** (analytic derivatives `dg/dtheta_i, dg/dtheta_j` of `direction * proj_k - rad_k` + exact `d(ext_i)/dtheta_i` + `O(ta^2)` curvature margin + trust-region halving / backtracking on exact `sat_max_penetration`) will unlock cooperative rotation of tilted chains inside our existing `cand_v13_n687`, `cand_v13_n796`, `cand_v11_n791`, `cand_v11_n969`, and exported `v13_r1`/`v13_r2` bins.
  2. Combining **gentle GPU breathing (`fast_mode=4/5`, `a: 0.94 -> 1.00`, zero artificial `w_align` penalty)** on `v13` / `v11` scratch basins + **end-matched curved/twin `24.8°–26.6°` L-pockets** (eliminating the `(0,L)/(L,0)` end-strip collision & corner miter holes `(4.46, 26.25)`, `(1.49, 2.48)` via tapered row length & miter curve) + `joint_xytheta_slp` on core targets `{687, 796, 367, 971}` (`<10` min pilot first) will close the remaining `0.021..0.027` gap.

### Results Log (Idea 14)
*(Pending execution)*
