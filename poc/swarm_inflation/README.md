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
