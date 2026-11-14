# gadget_squeeze — squeeze the pockets, recompose, certify

Hypothesis **H2**: the tracked large-`n` packings are compositions of rectangular "gadgets".
A gadget `(w, h, +g)` is `w·h + g` unit squares packed in a `(w+f) × (h+f)` box; along the
anti-diagonal seam of a `K+f` box, `m` gadgets give `n = K² + Σg`, and the row's side is
`K + max f`. The frontier beats us by a **lower `f` for the same gadget shape**, so improving
`f` for a handful of small rectangles pays at many `n` at once. `poc/dsmall_band`,
`poc/faultline_dp2`, `poc/minusE_shear` attack neighbouring questions; this folder attacks
`f` directly plus the gadget *supply*.

Result in one line: **no binding pocket moved**, but the recomposition DP over a strictly
richer library certifies **5 strict wins** (`n = 442, 485, 530, 577, 585`, total score gain
`1.227e-06`) — dust, all in `final.csv`. The useful output is the *demand table*
(`docs/demand.txt`): it says exactly which gadget shapes the big-gap rows need.

## Ground truth

- Incumbent `poc/pell_corrugation/data/border-scan/merged.csv`, official score
  **22.153257205012313**; per-`n` `ours`/`frontier` from
  `poc/koboshchan_coordinator/data/gap_vs_frontier.csv`.
- Pocket library regenerated from **this** incumbent (`poc/gadlib.py`, log in
  `docs/gadlib_state.txt`): 889 gadgets, 334 proper sub-gadgets, **101 distinct
  `(w ≤ h, +g)` types**. This is *not* the November-13 library: in the current one
  `(13,16,+4) @ 0.634071` no longer exists (the row `n = 791` was re-built as a single
  interlocked 26.4°/24.5° structure whose clean cut is the whole packing), and the binding
  shapes are now `(9,15,+3) @ 0.608713` (n = 848, 968), `(10,6,+2) @ 0.623065` (n = 848),
  `(11,16,+4) @ 0.637029` (n = 970), `(20,17,+1) @ 0.533025` (n = 401), `(24,25,+1) @ 0.520815`
  (n = 626), `(28,30,+1) @ 0.520815` (n = 901).

## Method

Everything lives in `poc/`; every reported `f` is a *realised* packing, and every reported
win carries the official metric in **both** transports.

1. **`squeeze.py` — free-angle squeeze of one rectangle** (centres *and* angles free):
   - *seeds*: the extracted incumbent pocket + mirror/transpose/small-rotation variants,
     upright grids, tilted row-block lattices over 16 angles (rows along the angle, per-row
     slides = the "meshed end" freedom the tracked pockets use), random sequential adsorption;
   - *compress-and-relax*: drop the seed in a deliberately loose `(w+1.4)×(h+1.4)` box and
     minimise the repo's exact SAT overlap penalty (`sat.py` = `k2p10_fill2/poc/a_overlap.py`,
     vectorised separating-axis penetration² with analytic gradients) plus a rectangle wall
     term by L-BFGS, then step `f` down (side continuation), rescaling and re-relaxing with
     jittered retries;
   - *exact polish*: `pslp_np.py` (pocket_squeeze's sequential-LP solver with numba stripped)
     linearises every near-active separating axis and shrinks `f` in a trust region, ending
     with an infeasible-path restoration to ~1e-15;
   - a candidate counts only if the independent SAT check reports `max(pair, wall) ≤ 1e-12`
     in the `(w+f)×(h+f)` box **and** the bounding box really fits.
   - `L-BFGS`: scipy's `L-BFGS-B` in this build costs ~20 ms/iteration at *any* size
     (63 variables: 300 iterations, 6-19 s, energy still 2.0) and dense `BFGS` costs
     0.3 s/iteration at 636 variables, so `squeeze.py` ships its own numpy two-loop recursion
     (~2 ms/iteration at 636 variables, machine-precision energies) with an Armijo line search
     and kink restarts.
2. **`hop.py` — structured basin hopping** around an extracted pocket: six shake operators
   (`scale`, `row` (a band translated along its own direction), `block`, `corner`, `subset`
   (re-randomise 5-25 % of the squares), `rot`), each repaired by the SAT relaxation and then
   re-squeezed by the SLP; 20 chains in parallel, best certified `f` kept.
3. **`cutlib.py` — new gadget supply**: every clean sub-rectangle of every existing pocket is
   itself a legal gadget (the compostion only needs the pocket to sit inside its own box), so
   889 pockets become 1517 cut gadgets / 537 distinct `(w,h,g)`, of which **77 shapes are new**
   and none improves an existing shape's `f`.
4. **`dp.py` — composition DP** (same staircase rule as `gaddp.py` / `pocket_squeeze/compose.py`:
   `Σw ≤ K`, `Σh ≤ K`, `Σ(w+h+1) ≤ 2K+1`, both orientations), re-implemented numba-free with an
   *incremental* knapsack: items enter in increasing `f` and adding one unbounded item is one
   ascending sweep, so a whole `K`-frontier costs `O(#items)` sweeps instead of one DP per level
   (5 s for all `K ≤ 31` instead of 4 min).
5. **`demand.py`** — for the biggest-gap rows: the composition at the level just below the
   frontier's `f`, and the leftover seam budget.
6. **`build_cert.py`** — build each predicted cell (`compose.py`), gate it with the SAT module,
   then certify with `squares._official_metric.evaluate_single_puzzle` on the literal strings
   **and** after a real `pandas.read_csv → to_csv` round trip; writes `final.csv`.

## Exact results

### Certified wins (`final.csv`, 5 groups, total score gain 1.227e-06)

| n | ours | new (literal) | new (pandas) | dL literal | dL pandas | composition |
|---|---|---|---|---|---|---|
| 442 | 21.533030303 | 21.533024975175 | 21.533024975175 | +5.328e-06 | +5.328e-06 | 1× `(20,17,+1)` from n=401, transposed, f=0.533025 |
| 485 | 22.533031303 | 22.533024975177 | 22.533024975177 | +6.328e-06 | +6.328e-06 | same |
| 530 | 23.533032303 | 23.533024975179 | 23.533024975179 | +7.328e-06 | +7.328e-06 | same |
| 577 | 24.533032303 | 24.533024975181 | 24.533024975181 | +8.328e-06 | +8.328e-06 | same |
| 585 | 24.705887939 | 24.705887436260 | 24.705887436260 | +5.025e-07 | +5.025e-07 | `(11,18,+6)` + `(13,5,+3)`, both from n=585, f=0.705887 |

All five are strict wins in both modes (uniform `2e-12` dilation, no fallback), certified from
the exact bytes in `final.csv`; provenance in `data/wins.json`. Cause: the `j = 1` rows at
`K = 21…24` were built from `(21,21,+1)…(24,24,+1) @ 0.53303x`, while the library also holds
`(20,17,+1) @ 0.533025` from `n = 401`, which is a hair better and still fits the seam.

`poc/verify_final.py` re-reads `final.csv`, converts `deg → rad` at 80 digits and re-certifies
each group from the file bytes:

```
n=442: ours 21.533030303094 -> literal 21.533024975175367211… / pandas 21.533024975175367209…
       dL +5.328e-06 / +5.328e-06  STRICT WIN      (485, 530, 577 analogous with +6.33e-6,
                                                     +7.33e-6, +8.33e-6)
n=585: ours 24.705887938765 -> literal 24.705887436260158000… / pandas 24.705887436260158000…
       dL +5.025e-07 / +5.025e-07  STRICT WIN
total score gain from the final.csv bytes: 1.2265e-06
```

Nominal union with the incumbent: `22.153257205012313 − 1.2265e-06 = 22.1532560` (the five
groups replace only their own `n`, so nothing can regress).

### The squeeze found nothing (`docs/hop_run1.txt`, `docs/hop_run2.txt`, `docs/squeeze_far.txt`)

| shape | N | incumbent f | target to pay | best certified f found | search |
|---|---|---|---|---|---|
| (9,15,+3) | 138 | 0.608713 | < 0.605885 (D=7/8 rows) | **0.608713** (no change) | 20 chains × 22 rounds structured hop |
| (11,16,+4) | 180 | 0.637029 | any decrease | **0.637029** | 20 × 20 |
| (10,6,+2) | 62 | 0.623065 | any decrease | **0.623065** | 20 × 20 |
| (9,16,+4) | 148 | 0.642570 | any decrease | **0.642570** | 20 × 20 |
| (11,15,+3) | 168 | 0.637029 (current lib) | — | **0.626015** (imported from `poc/pocket_squeeze`, re-certified here) | — |
| (13,16,+4) | 212 | *gone from the current incumbent* | < 0.605885 | **1.000000** (a 14×17 box; from-scratch compress-and-relax) | 20 starts |
| (28,30,+1) | 841 | 0.520815 | < 0.502153 | **0.520815** (returns the incumbent to 1e-9; every from-scratch seed jams at f = 1.0) | 10 starts, seeded from n=901 |

Only 0-2 of 22 hop trials per chain even reach a *feasible* re-polish; every feasible one
returns the incumbent `f` to 1e-9. The imported `(11,15,+3) @ 0.626015` is a real 0.011
improvement on the shape — and it changes **no** row, because that shape is not binding
anywhere (it only re-enters as a member of the `n = 585` composition).

Independent negative controls for the penalty stage (`squeeze.py`, `--no-seed`): the known
`(4,5,+1) @ 0.621320` pocket (21 squares) is *not* reachable — the search jams on the upright
`4×5+1` configuration at `f = 1.000000` and reports 0/24 feasible starts even at a target 15 %
looser than the record (0.71, 0.68, 0.65, 0.63). This reproduces `rgad.py`'s note that
compress-and-relax reaches the records only up to 13 squares.

### The DP is closed under composition, even with the richer library

| library | gadgets | distinct (w,h,g) | n improved | score gain |
|---|---|---|---|---|
| `gadlib.json` (clusters of the current incumbent) | 889 | 101 + whole rows | **5** | 1.227e-06 |
| `lib_plus.json` (+5 improved pockets from `poc/pocket_squeeze`) | 894 | — | 5 | 1.227e-06 |
| `lib_cut.json` (+1517 clean sub-rectangle cuts, 77 new shapes) | 1431 | 537 cuts | 5 | 1.227e-06 |

The `K`-frontiers are identical in all three (`docs/dp_closure2.txt`; the `lib_plus` and
`lib_cut` predictions are kept as `data/pred_lib_plus.json` / `data/pred_lib_cut.json` and
contain the same five cells with the same `K` and `f`): adding 77 new shapes and 5 better
`f`s does not move a single level. So *supply* of shapes is not the bottleneck — gain
**density** is (see the demand table). This updates the `poc/rectangle_gadgets` closure
result to the current incumbent and a much richer library.

### What would pay (`docs/demand.txt`, 16 biggest gaps)

For each big-gap row: the best composition at the highest `f` level strictly below the
frontier's `f`, and the leftover seam budget. Two regimes:

- **Budget saturated** (`dw = 0` or `dh = 0` or `dsum < 5`): `n = 803` (whole `28×28+17`),
  `n = 492` (`22×22+7`), `n = 975` (`31×31+12`), `n = 908` (three `(10,9,+2)` fill the seam),
  `n = 328`, `n = 365`. No extra pocket can be added at all, so these rows can only move by
  *replacing* gadgets — i.e. by shapes with strictly better gain per seam unit.
- **Budget left but wrong shape**: `n = 791` (`K=28, j=7`) already achieves gain 5 at
  `f ≤ 0.605829` with `(12,17,+3) + (11,8,+2)ᵀ`, leaving `dw = 8, dh = 0, dsum = 7`; reaching
  the frontier's 0.605885 needs **2 more squares with `h = 0`** — impossible, so again a
  replacement shape is needed. Concretely: a `(w,h,+g)` with `w+h+1 ≤ 12`, `g ≥ 2` at
  `f ≤ 0.6059` (we have `(9,10,+2) @ 0.605829` and `(8,11,+2) @ 0.591378` and nothing better)
  would take `n = 791, 908, 969, 848` and the whole `j = 7, 8` column with them.
- `n = 367` (`K=19, j=6`) sits at gain 5 with `(12,12,+4) + (4,5,+1)` and needs +1 at
  `f ≤ 0.670857` with `dw=3, dh=2, dsum=4`: a `(3,2,+1)`-sized shape at `f ≤ 0.6709` would pay
  1.1e-3 of score — the single biggest per-row prize left in the composition picture.

These are the concrete shopping list for whoever builds new pockets: **no extra pocket fits
the top rows, they need better gain density at `f ≈ 0.60-0.67` in small footprints.**

## Why the squeeze cannot help (mechanism)

The tracked pockets are engineered interlocked structures: a 24-27° block whose row ends mesh
with the two corner-anchored grids, with the row length sequence doing the work (see
`poc/pocket_squeeze/README.md`). Cut from that context, the pocket is a *rigid jam*:
- any local move either keeps the mesh (same `f` to 1e-9) or breaks it (infeasible), which is
  what the 0-2/22 feasible re-polishes per chain measure;
- the penalty landscape has a floor around `E ≈ 1e-8` (`violation ≈ 1e-5`) because the
  separating axis flips between branches, so the relaxation cannot even certify a pocket
  without the SLP stage;
- from-scratch compress-and-relax jams at the grid configuration from 21 squares upward.

Together with `poc/pocket_squeeze` (337 pockets, plain squeeze, 40 basin-hopping rounds, seed
factory) and `poc/free_angle_slp` / `poc/alln_fixed_lp` (whole-portfolio local optimality),
the evidence is that isolated-`f` improvement is not the missing mechanism: the missing thing
is a pocket family with better gain **density** at `f ≈ 0.60`, i.e. a different *construction*.

## How to run

```bash
cd poc/gadget_squeeze/poc
PY=../../../.venv/bin/python          # (from this folder: /mnt/e/.../.venv/bin/python)

# 1. regenerate the pocket library from the incumbent (≈ 3 min)
$PY gadlib.py ../../koboshchan_coordinator/data/incumbent-22.15325720501.csv ../data/gadlib.json

# 2. best instance per gadget type (seed pockets) + the squeeze seeds file
$PY mk_targets.py                     # -> ../data/targets_all.json
$PY pick.py 9,15,3 28,30,1            # -> ../data/targets.json (only the named shapes)

# 3. squeeze / hop a shape (all free-angle, all certified)
$PY squeeze.py --shape 13,16,4 --no-seed --starts 20 --procs 20      # compress-and-relax
$PY hop.py --shape 9,15,3 --seed-file ../data/targets_all.json --rounds 22 --chains 20

# 4. enrich the library and re-run the composition DP
$PY cutlib.py ../data/lib_plus.json ../data/lib_cut.json --nmax 8    # clean sub-rectangle cuts
$PY dp.py ../data/lib_cut.json ../data/oursL.json --out ../data/pred_lib_cut.json

# 5. build, gate and certify every predicted cell (writes ../final.csv)
$PY build_cert.py ../data/pred_lib_cut.json --lib ../data/lib_cut.json

# 6. re-certify the deliverable from its own bytes, and ask what would pay
$PY verify_final.py
$PY demand.py ../data/lib_cut.json 16
```

Minimal path that reproduces `final.csv` exactly (all five groups use pockets that are in
`data/gadlib.json` alone):

```bash
$PY dp.py ../data/gadlib.json ../data/oursL.json --out ../data/pred_gadlib.json
$PY build_cert.py ../data/pred_gadlib.json --lib ../data/gadlib.json
```

`data/gadlib.json` (21 MB) is checked in because `build_cert.py` needs it; `lib_plus.json`,
`lib_cut.json` and `targets_all.json` (21/30/19 MB) are regenerable from it and are not kept.

## Files

| path | what |
|---|---|
| `poc/squeeze.py` | free-angle squeeze: seeds, SAT compress-and-relax, SLP polish, side continuation, certification gate |
| `poc/hop.py` | structured basin hopping around an extracted pocket |
| `poc/sat.py` | verbatim copy of `poc/k2p10_fill2/poc/a_overlap.py` (the repo's SAT overlap module) |
| `poc/pslp_np.py` | `poc/pocket_squeeze/poc/pslp.py` with numba removed (`feas` vectorised with numpy/cKDTree) |
| `poc/compose.py`, `poc/compose_ref.py` | seam builder (numba-free) and the untouched reference copy |
| `poc/dp.py` | incremental composition DP + frontier/level queries |
| `poc/gaddp_ref.py` | untouched reference copy of `poc/rectangle_gadgets/poc/gaddp.py` |
| `poc/cutlib.py` | clean sub-rectangle gadget enumeration |
| `poc/gadlib.py` | copy of the pocket extractor (library regeneration) |
| `poc/mk_targets.py`, `poc/pick.py` | seed-pocket selection |
| `poc/demand.py` | demand table for the biggest-gap rows |
| `poc/common.py` | incumbent loader, official two-transport certification, `final.csv` writer |
| `poc/build_cert.py` | build + SAT gate + official certification + `final.csv` |
| `poc/verify_final.py` | re-certify `final.csv` from its own bytes (`deg → rad` at 80 digits) |
| `data/gadlib.json` | 889 pockets of the current incumbent |
| `data/lib_plus.json`, `data/lib_cut.json` | + 5 improved old pockets / + 1517 cut gadgets |
| `data/pred_*.json` | DP predictions, `data/wins.json` the 5 certified groups with provenance |
| `docs/gadlib_state.txt`, `docs/dp_closure2.txt`, `docs/dp_plus.txt` | library + DP closure logs |
| `docs/hop_run1.txt`, `docs/hop_run2.txt`, `docs/squeeze_far.txt` | negative squeeze logs |
| `docs/demand.txt` | per-row seam budget / shopping list |
| `final.csv` | the 5 certified groups (`id,x,y,deg`, s-prefixed decimals) |

## Open

- The demand table says the money is a small-footprint pocket with `f ≈ 0.60-0.67` and
  `g ≥ 2`: `(w,h,+2/3)` with `w+h+1 ≤ 12`. That is a *construction* problem in a 10-14 square
  box, not a local-optimisation problem.
- `n = 803, 492, 975, 908, 861` use the whole row as one pocket (`28×28+17` etc.); their seams
  are exactly saturated, so they need a better whole-row construction (H1/H3 territory).
- `hop.py`'s `subset` shake is the only operator that changed the contact topology in
  `pocket_squeeze`-scale tests; a version that also re-assigns squares between a block and the
  adjacent grid (row-length swap) is the natural next move.
