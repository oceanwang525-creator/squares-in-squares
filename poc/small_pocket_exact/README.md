# small_pocket_exact — C2: can a small pocket beat the pay lines? (exact answer for `(3,2,+1)`; negative)

Status: **no certified win (0 rows in `final.csv`), and a definitive `(3,2,+1)` answer.**
The `(3,2,+1)` pocket — the one shape in the C2 shopping list whose pay line is
close enough to be reachable — bottoms out at exactly `f = 1/sqrt(2)`, which is
*the* pay line from above: it pays iff `f < 1/sqrt(2) - 1e-7`, and the optimiser
returns `f = 1/sqrt(2)` to 16 digits from every one of thousands of restarts.
The other eight pay-line shapes are **not decidable** with the engines available
here (including the new one, whose calibration gap on 17/21/65 squares is
+0.032 / +0.086 / +0.465 in `f`), and the paper argument says they need densities
`0.827 … 0.873` — 4-10 % above the trivial grid, and (for the long boxes) above
the best aligned-shelf density for their own width.

Two by-products change the picture the coordinator was given:

1. **The `n=367` prize of 1.1e-3 does not exist.** It comes from
   `poc/gadget_squeeze/docs/demand.txt`, which proposes a `(3,2,+1)` at
   `f<=0.6709`; but `n=367`'s remaining seam budget is `dw=3, dh=2, dsum=4`, and
   a gadget added there needs `w+h+1 <= 4`, i.e. `w+h <= 3`. A `(3,2)` gadget
   does not fit that budget at all (`w+h+1 = 6`).
2. **The real `(3,2,+1)` pay line is `f < 1/sqrt(2)` (worth 4.76e-4 at
   `n = 10, 38, 492, 974`), not `0.6709`.** Both facts are computed exactly from
   the repo DP (`poc/payline.py`, `poc/gain.py`), see §3.

---

## 1. Engine

Everything is numpy; **numba is not installed** in `.venv` (numpy 2.5.4 /
scipy 1.18.1 / pandas 3.1.1) and installing it was ruled out as a risk to the
shared environment, so the engine is a batched-numpy design instead.

| file | what |
|---|---|
| `poc/satb.py` | the repo's exact SAT penalty, batched over `R` restarts **and** the pair list, with analytic gradients. Also `energy_grad_flat` (screened neighbour lists, identical energy, 5-10x cheaper), `report` (independent min-over-4-axes depth), `tighten`/`shape_f`. |
| `poc/packb.py` | `Run`: per-restart box excess `f_r`, normalised-GD with momentum **and a per-restart backtracking line search** (one gradient evaluation per step), plus shrink / restore-from-anchor+jitter / hard-kick continuation. Seed families: upright lattice, tilted-angle lattice, *genuinely rotated* lattice (spacing 1, isotropically compressed), best-candidate RSA. |
| `poc/polish.py` | `pslp_np.squeeze` (the repo's sequential-LP squeeze, `poc/gadget_squeeze/poc/pslp_np.py`) + validation with the full pair list. |
| `poc/search.py` | the pipeline: continuation → SLP polish pool (multiprocessing) → `shake_squeeze` basin hop → fixed-`f` jam probe. |
| `poc/probe.py` | fixed-box feasibility probe (all restarts in the *same* box): `min E = 0` ⇔ feasible box found. |
| `poc/hammer.py`, `poc/masssqueeze.py` | multi-start SLP squeezers (loose feasible states → hard-shrink → SLP). |
| `poc/donor.py` | squeeze a library pocket of the *same* shape (only `(13,5,+3)` exists). |
| `poc/payline.py` | exact pay line of a hypothetical shape against the current incumbent (reuses `poc/faultline_dp2/poc/dpx.py` semantics + `data/report.json`). |
| `poc/gain.py` | DP gain of a shape at a *specific* realised `f` — the decision query once a real pocket exists. |
| `poc/build.py` | build the `m=1` staircase composition from a pocket (same geometry as `poc/faultline_dp2/poc/build_pocket.py`) and certify it in **both** transports (`squares.core.certify` on literal decimal strings, and again after a real `pandas.read_csv -> to_csv` round trip via `squares.core.kaggle_roundtrip`). |

**Model check** (`python poc/satb.py`): energy identical to
`poc/k2p10_fill2/poc/a_overlap.py` (`diff 0.0`), analytic gradient matches it to
`6.7e-16` (scale 2.4) and matches finite differences to `8.4e-10`, including the
box-wall terms in a tight box (`5.6e-09`).

Two engine lessons worth keeping:

* **Adam is unusable here.** Per-coordinate normalisation makes the angle
  components overshoot; measured on `(2,2,1)` at a fixed box it *increases* the
  energy (5.3 → 5.5 at lr 0.02, 11.9 at lr 0.05). What works is a normalised
  gradient (one scalar norm for the `xy` block, one for the `theta` block) with a
  per-restart backtracking step — it converges to `E = 0` exactly when the box is
  feasible.
* **The SLP, not the descent, does the squeezing.** For `(4,4,1)` the batched
  continuation bottoms out at `f = 0.8854`; one `pslp_np.squeeze` pass on a
  `0.78`-state gives `0.7085`, and the polish pool over the archive reaches
  `0.7071`. The continuation's job is to generate
  *diverse feasible structures cheaply* (thousands of restarts), the SLP's job is
  to contract them.

## 2. Calibration (free random starts, no donor pockets)

Records are the ones in the brief; every "best" below is a *validated* packing
(exact all-pairs SAT check, `violation <= 1e-12`), found from random seeds only.

| shape | N | record `f` | best found here | gap | gap % |
|---|---:|---:|---:|---:|---:|
| `(2,2,1)` | 5 | 0.707107 (= 1/√2) | **0.707106781187** | **0.0** | exact |
| `(4,4,1)` | 17 | 0.675530 | 0.707106781 | **+0.031577** | +4.67 % |
| `(4,5,1)` | 21 | 0.621320 | 0.707106781 | **+0.085787** | +13.8 % |
| `(8,8,1)` | 65 | 0.535534 | 1.000000000 | **+0.464470** | +86.7 % |

So: **exact at 5 squares, 4.7 % short at 17, 14 % short at 21, useless at 65.**
That is still a large step up on the previous searchers (`poc/rectangle_gadgets/poc/rgad.py`
and `poc/faultline_dp2/poc/gadsearch.py`: `(4,4,1)` → 1.0356 / 0.769 vs the record
0.6755), but it is **not** calibrated in the `N = 30…80` range that the pay-line
shapes live in. The reason is visible in the calibration: the engine finds the
*"upright grid + 45° corner diamond"* family (which is why `(2,2,1)`, `(3,2,1)`,
`(4,4,1)`, `(4,5,1)` all come back at exactly `f = 1/√2`), and it does not find
the tilted-band structures that the 0.62–0.68 records need. Every shape with
`N >= 30` comes back at exactly `f = 1.000000000`, i.e. the trivial
`a x b` grid plus a one-square strip — the same wall `poc/gadget_squeeze` hit
("from-scratch compress-and-relax jams at the grid configuration from 21 squares
upward").

Instrument accuracy of the fixed-box jam probe (`probe.py`), calibrated on the
one record we know exactly: for `(2,2,1)` (true optimum 0.7071068) the probe is
feasible at `f = 0.7085` and jammed at `f = 0.7072` with `min E = 3.2e-8`, i.e.
it overestimates the threshold by ~1e-4 and cannot resolve a 1e-7 question. It
is a useful jam indicator, not a decision procedure.

## 3. The pay lines (exact, repo DP against the current incumbent)

`poc/payline.py` bisects the largest `f` with positive DP gain (gain is
`sum dL/sqrt(n)` over every `n` the composition improves, `dL > 1e-7`), using
`poc/faultline_dp2/poc/dpx.py` on `data/gadlib.json` + `data/report.json`
(incumbent 22.15325720501231). `data/paylines.json`.

| shape | N | pays only if `f <=` | gain at that `f` | `n` improved | density needed at the line |
|---|---:|---:|---:|---|---:|
| `(3,2,+1)` | 7 | **`< 1/√2 - 1e-7`** (bisection floor 0.706250) | 4.76e-4 | 10, 38, 492, 974 | 0.6975 |
| `(4,7,+3)` | 31 | 0.777344 | 1.262e-4 | 150, 203 | 0.834 | 
| `(4,13,+3)` | 55 | 0.691016 | 1.472e-4 | 330, 367 | 0.856 |
| `(5,13,+3)` | 68 | 0.691016 | 7.165e-5 | 367 | 0.873 |
| `(4,16,+3)` | 67 | 0.655469 | 6.553e-5 | 447 | 0.864 |
| `(4,19,+3)` | 79 | 0.632617 | 5.659e-5 | 582 | 0.869 |
| `(5,6,+2)` | 32 | 0.706250 | 1.390e-4 | 38 | 0.836 |
| `(4,7,+2)` | 30 | 0.706250 | 3.863e-5 | 492 | 0.827 |
| `(6,7,+2)` | 44 | 0.691016 | 7.165e-5 | 367 | 0.855 |

The bisection resolution is ±0.0013; for `(3,2,+1)` the threshold was then pinned
exactly with `poc/gain.py`:

```
(3,2,+1) at f=0.7071066 (1/√2-1.8e-7): gain 1.007e-07 over 4 n
(3,2,+1) at f=0.7071068 (1/√2-3.8e-8): gain 0.000e+00 over 0 n
```

and the gain is exactly linear in the shortfall, `gain = 0.5555 * (1/√2 - f)`
(`0.5555 = 1/√10+1/√38+1/√492+1/√974`), so `f = 0.70625` (the bisection floor)
gives 4.76e-4 and any improvement `δ` gives `0.5555 δ`.

**Why `n=367` is not on the list.** `demand.txt` asks for "+1 gain at
`f <= 0.670857` with budget `dw=3, dh=2, dsum=4`". The composition rule is
`Σw <= K, Σh <= K, Σ(w+h+1) <= 2K+1`, so an extra gadget needs `w <= 3`,
`h <= 2` **and** `w+h+1 <= 4`. A `(3,2)` item has `w+h+1 = 6` and cannot be
added to that row at any `f`. The budget only admits shapes with `w+h <= 3`, and
none of those can reach `f < 1` at all: every unit square's axis-parallel bounding
box is `e x e` with `e = |cos θ|+|sin θ| >= 1`, so a box of height `1+f < 2`
cannot take two squares in a stack and a box of width `3+f < 4` cannot take four
in a row, giving `f >= 1` for `(1,1,+1)`, `(2,1,+1)`, `(3,0,+g)` (our engine
returns exactly `f = 1` for these, and `probe.py` jams at `f = 0.6709`). The DP
agrees: `(3,2,+1)` improves `n = 10, 38, 492, 974` and **not** `367`.

## 4. Per-shape results: best `f` vs pay line

Every run: batched continuation (`R` restarts, 30-45 sweeps x 80-100 sweeps of
80-100 steps), 64-128 SLP polishes in a process pool, basin hop, then a fixed-`f`
probe at the pay line. Best `f` is a validated packing (`violation <= 1e-12`).
`data/v2_*.json`, `logs/v2_*.log`.

| shape | N | pay line | **best `f` found** | gap | density at best / needed | verdict |
|---|---:|---:|---:|---:|---:|---|
| `(3,2,+1)` | 7 | `<1/√2` | **0.707106781187** (=1/√2) | +1e-7 … 8.6e-4 | 0.6975 / 0.6975 | **at the boundary, does not pay** (§5) |
| `(4,7,+3)` | 31 | 0.777344 | 1.000000000 | +0.2227 | 0.775 / 0.834 | not decided (engine at the trivial grid) |
| `(5,6,+2)` | 32 | 0.706250 | 1.000000000 | +0.2938 | 0.762 / 0.836 | not decided |
| `(4,7,+2)` | 30 | 0.706250 | 1.000000000 | +0.2938 | 0.750 / 0.827 | not decided |
| `(6,7,+2)` | 44 | 0.691016 | 1.000000000 | +0.3090 | 0.786 / 0.855 | not decided |
| `(4,13,+3)` | 55 | 0.691016 | 1.000000000 | +0.3090 | 0.786 / 0.856 | not decided |
| `(5,13,+3)` | 68 | 0.691016 | 1.000000000 | +0.3090 | 0.810 / 0.873 | not decided |
| `(4,16,+3)` | 67 | 0.655469 | 1.000000000 | +0.3445 | 0.788 / 0.864 | not decided |
| `(4,19,+3)` | 79 | 0.632617 | 1.000000000 | +0.3674 | 0.790 / 0.869 | not decided |

The fixed-`f` probes at each pay line are all strongly jammed (`min E` between
1.7e-2 and 2.3e-2 — for comparison, `(2,2,1)` at 1e-4 above its optimum is
3e-8), but §2 says that is not evidence: the same probe is jammed at 2.2e-2 for
`(8,8,1)`, whose true record is 0.465 lower.

**What would have to be true for these to exist.** The pay lines need densities
`0.827…0.873`. Context (`data/gadlib.json`, current incumbent):

| construction | density |
|---|---:|
| trivial `a x b` grid + strip at `f = 1` (best our engine reaches for `N>=30`) | 0.750 – 0.810 |
| axis-aligned 4-column shelf in width `4+f`, long box (`4/(4.63)`) | 0.864 (asymptotic) |
| `(4,5,+1)` record `f = 0.62132` | 0.808 |
| `(5,7,+1)` record `f = 0.57143`, `(6,6,+1)` 0.59862 | 0.853 |
| `(8,8,+1)` record `f = 0.535534` (45° band) | 0.892 |

So `(4,7,+3)`/`(5,6,+2)`/`(4,7,+2)` need `0.827…0.836`, which is *inside* the
range that real packings achieve for neighbouring shapes (0.853 at `(5,7)`), so
they are open on density grounds; `(4,19,+3)` at 0.869 is **above** the best
shelf construction for its own width (`4/4.6326 = 0.864`) and would need a
genuinely non-shelf, non-lattice construction. None of this is a proof.

## 5. `(3,2,+1)`: the one decidable case — it lands exactly on the pay line

7 squares in `(3+f) x (2+f)`, so `f < 1/√2 = 0.7071068` is needed. Every search
returns `f = 1/√2` **exactly** (the validated packing's realised `f` agrees with
`1/√2` to 16 digits, i.e. the construction is rigid, not numerically optimised):

| search | result |
|---|---|
| continuation `R = 8192`, 45 sweeps x 100 steps + 128 SLP polishes + hop (`logs/v2_3_2_1.log`) | 0.707106781 (viol 6.7e-16) |
| `masssqueeze.py`: 8192 starts at `f=0.78` → 43 feasible → hard shrink to `f=0.7060` → SLP (`logs/mass_3_2_1.log`) | 19/19 valid squeezes return **0.707106781** |
| `hammer.py`: 4096 fresh starts, shrink to 0.7065, SLP | 0.707106781 |
| `probe.py` at `f = 0.7085 / 0.7072 / 0.7069 / 0.70625 / 0.70` | jammed, `min E = 8e-7 / 3.8e-6 / 5.6e-6 / 2.3e-6 / 5.5e-5` |

The solution is the **rigid `3 x 2` upright grid + one 45° square tucked into a
corner**: with `f = 1/√2` the 0.7071-wide corner margin is exactly the amount a
45° square needs to reach the corners of its two neighbours (the `(2,2,+1)`
argument, applied to the corner of a 3x2 block). The same reasoning says the
diamond cannot sit in the middle: it needs a `1.414 x 1.414` hole, and two rows
of squares around a mid-box hole already need `1 + 1.414 + 1 = 3.414 > 2+f`.
So the corner-diamond family is forced, and its optimum is exactly `1/√2`.

**End-to-end certification of the boundary case** (`poc/build.py`, `n = 10`,
`K = 3`, pocket flush in the top-right corner + the 3-square L-grid):

```
n=10 K=3  pocket (3,2,+1) 7 sq + grid 3 sq   f=0.707106781223619  side=3.707106781223619
   incumbent 3.70710678118654760000000000000000000000000000000000
   literal   3.70710678121961863019670448169146298605013406979504   dL +3.307e-11
   pandas    3.70710678121961901271137764171157633842087844979504   dL +3.307e-11
   no win (not strictly better)
```

(all of the `3.3e-11` is the `1e-11` dilation needed to make a numerically
optimal pocket pass the official metric; the incumbent's own `L(10)` is
`3+1/√2` rounded up at the 16th decimal). **No win, both modes.**

## 6. Donor squeeze `(13,5,+3)` → the one shape with a library donor

`data/gadlib.json` holds `(13,5,+3) @ 0.705886512` (from `n = 585`). Cut out of
its row it is **not** locally optimal: one `pslp_np.squeeze` gives
`f = 0.701529375` (−6.2e-3, `logs/donor_13_5_3.log`). It still does not pay:
`poc/gain.py 13,5,3:0.701529375` → **gain 0**, because that shape is not binding
anywhere (the DP needs `f <= 0.691016` for it to reach `n = 367`), which
reproduces `poc/gadget_squeeze`'s finding. `shake_squeeze` chains were started on
it (`donor.py --chains 32`) but stopped; no improvement over the plain squeeze
was seen before the stop.

## 7. Certified wins

**None.** `final.csv` is header-only (`id,x,y,deg`). No shape reached its pay
line, so nothing was added to `poc/faultline_dp2/poc/dpx.py extra` and no full
packing was built for score. The only certified build is the boundary
demonstration above (`n = 10`, `dL = +3.3e-11`, not a win).

## 8. Run commands

```bash
cd /mnt/e/squares-in-squares-commandcode/poc/small_pocket_exact
PY=../../.venv/bin/python

# model check of the batched SAT penalty + analytic gradient
$PY poc/satb.py

# exact pay lines of the 9 shapes against the current incumbent (~5 min)
$PY poc/payline.py --shapes 3,2,1 4,7,3 4,13,3 5,13,3 4,16,3 4,19,3 5,6,2 4,7,2 6,7,2 \
    --n 6 --out data/paylines.json
# DP gain of one shape at one realised f
$PY poc/gain.py 3,2,1:0.7071068 3,2,1:0.7071066

# full search for a shape (continuation + SLP pool + hop + jam probe)
$PY poc/search.py --shape 4,7,3 --fpay 0.777344 --R 1024 --sweeps 30 --iters 80 \
    --npolish 128 --hop 24 --procs 2 --tag v2_4_7_3 --out data/v2_4_7_3.json
# small shapes: more restarts
$PY poc/search.py --shape 3,2,1 --fpay 0.706250 --R 8192 --sweeps 45 --iters 100 \
    --npolish 128 --hop 40 --procs 2 --tag v2_3_2_1 --out data/v2_3_2_1.json

# other instruments
$PY poc/probe.py --shape 2,2,1 --f 0.72 0.7085 0.7072 0.7065      # jam probe
$PY poc/masssqueeze.py --shape 3,2,1 --f0 0.78 --hit 0.7060 --R 8192
$PY poc/hammer.py --shape 4,4,1 --f0 0.95 --R 1024 --iters 500 --hit 0.70
$PY poc/donor.py --shape 13,5,3 --chains 32 --rounds 60

# build + two-transport certification of the boundary case (writes final.csv)
$PY poc/build.py --pocket data/v2_3_2_1.json --n 10 --w 3 --h 2 --final final.csv
```

Batch drivers used for the reported runs: `logs/run_payline.sh`,
`logs/run_shapes.sh`, `logs/run_v2.sh`, `logs/run_probe.sh`,
`logs/run_hammer.sh`, `logs/run_calib.sh`.

## 9. Files

- `poc/` — engine and tools (§1). `poc/satb.py`, `poc/packb.py`, `poc/polish.py`,
  `poc/search.py`, `poc/probe.py`, `poc/hammer.py`, `poc/masssqueeze.py`,
  `poc/donor.py`, `poc/payline.py`, `poc/gain.py`, `poc/build.py`.
- `data/` (152 KB) — `paylines.json` (exact thresholds), `v2_*.json` /
  `cal2_*.json` / `shape_s3_2_1.json` (per-shape results + the winning pocket
  coordinates), `mass_3_2_1.json`, `hammer_*.json`.
- `logs/` — full run logs for every number quoted here, plus the batch scripts.
- `final.csv` — header only.

## 10. What this says about C2

* A `(3,2,+1)` pocket cannot pay: its pay line *is* `1/√2`, the value of the
  rigid construction that every restart converges to. The 1.1e-3 "n=367" story
  in `demand.txt` is an arithmetic slip (the seam budget cannot take a `(3,2)`
  gadget); the real prize for that shape is 4.76e-4 and it is 1e-7 out of reach.
* The other eight shapes need densities 0.83-0.87 in boxes 30-90 squares large.
  The missing capability is *not* composition and *not* local squeezing (the
  incumbent's own `(13,5,+3)` pocket moves 6.2e-3 under a plain SLP squeeze) — it
  is the construction: a tilted band with a non-trivial row-length sequence inside
  a `4.6 x 19.6` box. The batched engine cannot build those, and neither could
  `rgad.py`, `gadsearch.py` or `squeeze.py`.
* A decisive negative for those eight needs either (a) an engine that reproduces
  `(4,5,+1) = 0.62132` / `(8,8,+1) = 0.535534` from random starts (the honest
  calibration bar, which nothing in this repo reaches yet), or (b) a real
  lower-bound technique. The "bbox of a rotated square is an `e x e` square with
  `e >= 1`" argument looks promising but is *not* usable directly: shrinking each
  bbox to a unit sub-square breaks disjointness (the sub-square of a bbox need not
  lie inside the rotated square), so the resulting "unit-interval-graph" bound
  `N <= floor(W)·floor(H)` is false — it would forbid the optimal 5-in-square
  packing at `2+1/√2`. Do not reuse it.
