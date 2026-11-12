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
