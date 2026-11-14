# minusE_shear — (K+1)² grid minus an L, then shear/interlock (H5)

**Verdict: negative. Nothing certifies, and the family is structurally dominated
by the incumbent at all three targets.** The best side this family certifiably
reaches is `K+1` for every target — `0.293 / 0.366 / 0.386` **above** the
incumbent and `0.309 / 0.394 / 0.400` **above** the frontier. `final.csv` is
header-only; no `dL` exists to report in either transport.

## Idea and why it is arithmetically natural

For `n = K² + D` (with `K = isqrt(n)`):

```
E = (K+1)² - n = 2K + 1 - D
```

and the border L-band of a `(K+1)x(K+1)` grid

```
top row   (x, K)   x = 0..K
right col (K, y)   y = 0..K-1
```

has exactly `2K+1` cells — so **exactly `D` of them survive** the deletion and
the inner `K x K` block is untouched. Every "delete an L" pattern is therefore a
choice of which `D` band cells to keep, and the packing is `K² + D = n` squares
in a `(K+1)`-box. H5 then asks whether shearing the two resulting blocks
(top-right block shifted `(-d,-d)`, or the blocks rotated) lets the staircase
seam interlock and the side fall to `K + c` with `c ≈ 0.52–0.61`.

## What was built (`poc/minusE.py`)

Per target the constructor emits, and every assembly is checked with an exact
SAT overlap test (nearest-pair, vectorised) so self-overlapping states never
reach a solver:

1. **Deletion patterns** — 25 per target, exactly `D` survivors each:
   `topL`, `topR`, `botR`, `topcolR` (whole path on the top row / on the right
   column / both ends), **every contiguous corner split `a:b`** around the
   top-right corner, **every contiguous outer split `a:b`** (top-row left end +
   column bottom end), `spread` / `spread_rev` staircase phases along the band,
   and 6 seeded random subsets.
2. **Block moves** per pattern:
   * `tiltcell` — tilt each surviving band cell in place at ±45°, ±40°, ±50°,
     30°, 60°;
   * `tiltband` — rotate the whole surviving band block (centres **and**
     angles) to those six angles and anchor it on the top-right corner of the
     inner block over a 4-anchor x 9-jitter placement grid;
   * **two-block shear** — six cuts (`band`, `toprow`, `rightcol`, `halfTR`
     `i+j >= G-1`, `stripUp` `i+j >= G`, `diagHigh`), each translated over
     12–32 offsets that include the H5 move `(-d,-d)` and the perpendicular
     seam slide `(d,-d)` on a 0.1 grid, plus 45°/‑45°/90° rotations of the cut
     block.
3. **Fixed-angle box LP** (`poc/lp_core.py`, copied from
   `poc/alln_fixed_lp/poc/alln_fixed_lp.py`: no trust region, one separating
   axis per nearby pair, pair set + axis rebuilt every round, 8–12 rounds),
   then the **free-angle Lipschitz SLP** (copied from
   `poc/lip_slp/poc/lip_slp.py`, 8 rounds) on the best candidates, with the
   official certifier run in **both** literal-decimal and pandas-transport mode.

## Measured results

| n | K | D | E | assemblies (main; tilted run) | min raw side over *all* assemblies | best fixed-angle LP side | best **certified** side (SLP) | incumbent | frontier |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 492 | 22 | 8 | 37 | 730; 2700 | 23.000000000 (25.866 tilted) | 23.000000024 | **23.000000022** | 22.707106781 | 22.690540476 |
| 908 | 30 | 8 | 53 | 869; 2695 | 31.000000000 (33.866 tilted) | 31.000000032 | **31.000000030** | 30.634070752 | 30.605884509 |
| 328 | 18 | 4 | 33 | 728; 1791 | 19.000000000 (20.366 tilted) | 19.000000020 | **19.000000018** | 18.614368938 | 18.599535465 |

Exact certified gaps (`poc/pell_corrugation/data/border-scan/merged-literal-official.json`
as incumbent):

| n | best certified in family | incumbent | dL (positive = worse) | above frontier |
|---:|---:|---:|---:|---:|
| 492 | 23.000000022 | 22.70710678118665359 | **+0.29289324081334841** | +0.309459546 |
| 908 | 31.000000030 | 30.63407075157907730 | **+0.36592927842092570** | +0.394115521 |
| 328 | 19.000000018 | 18.61436893849098605 | **+0.38563107950901395** | +0.400464553 |

Readings:

* `LP side = K + 1 + 2.0e-8…3.2e-8` is the **saturation signature** of the
  unbounded fixed-angle LP (the accumulated `1e-9` wall/pair margins, exactly as
  documented in `poc/alln_fixed_lp/README.md`): there is **no** translational
  slack anywhere in the family. The free-angle SLP adds nothing either
  (`23.000000022` vs `23.000000024`). The deleted L is never closed.
* **One sub-frontier LP number appeared and is a certifier-rejected artefact.**
  For `n=492` / pattern `corner3_5` / `tiltband+60°` the LP reports
  `22.36602542678444`, which would be a spectacular win — but 492 unit squares
  cannot fit in a 22.366 box without overlapping, and the official metric
  rejects the LP state outright:
  `n=492: infeasible: Squares 0 and 1 overlap (float check)`. The unbounded LP
  walked grid squares through each other (this is the documented
  `certify-fail` mode of `alln_fixed_lp`); `eval_candidate` now also drops any
  LP side with `side² < n` before it can waste an SLP slot. **No sub-incumbent
  LP number in this POC survives certification.**
* Tilted assemblies are strictly worse: rotating the surviving band block makes
  it longer than the corner it has to sit in, so the smallest raw side over all
  tilted assemblies is `25.87 / 33.87 / 20.37`. A hand-anchored single 45° band
  layer at `n=492` reaches an LP floor of `23.414214 = 22 + √2` (and `23.408832`
  at 40°/50°), i.e. `0.70` worse than the incumbent.

### No deletion pattern interlocks (`poc/seam_probe.py`)

`docs/seam_probe.json`: for 3 deletion patterns (`topL`, `corner4_4`, `outer`)
x 4 cut definitions x 3 targets = **36 probes**, the deepest bbox-reducing
`(-d,-d)` slide that keeps the packing overlap-free is **d = 0.00** at 0.05
resolution in every single case (with `halfTR`/`stripUp` blocks of 138–443
squares). Hits are not "hard to find": after the L is deleted the two halves are
still **edge-adjacent across the anti-diagonal staircase**, so any inward slide
by `d` makes O(K) pairs overlap by `d`. Sliding along the seam direction
`(d,-d)` is collision-free but leaves the bounding box unchanged, and sliding
outward only grows it.

## Why the family cannot reach the frontier

1. **A solid `K x K` block cannot be shrunk from outside.** Every minus-E
   assembly contains the complete upright `K x K` lattice, which fills
   `[0,K]²`. A square disjoint from it must lie entirely outside, so it
   protrudes `|cos t| + |sin t| >= 1` beyond one edge (equality only for an
   upright square). Hence side `>= K+1`, attained exactly by the minus-E shape
   itself. That bound is what the LP/SLP measure.
2. **The incumbent's own mechanism is a different, tighter family.** Our
   `n=492` row is `22.707106781 = 22 + 1/√2`: 483 upright squares at two
   half-lattice offsets `(0, 1/√2)` and `(1/√2, 0)` plus 9 squares at 45° in the
   seam — i.e. the `KxK` grid **split in half**, separated by one unit along the
   anti-diagonal normal, seam filled with 45° squares. In any single-layer seam
   zipper the seam needs perpendicular width `w >= 1` (a square's extent in any
   direction is at least its side), and the axis components of that separation
   are `(w sin φ, w cos φ)`, whose maximum is minimised at `φ = 45°`, giving
   `w/√2 >= 1/√2`. So the zipper floor is exactly `K + 1/√2`:
   `22.707106781` at `n=492` (**exactly the incumbent** — a strict win needs
   `<= 22.690540`), and `30.707 / 18.707` at `n=908 / 328`, which is *worse*
   than the incumbent (`30.634 / 18.614`). The minus-E/shear recipe is a
   strictly weaker version of a construction we have already saturated.
3. **Tucking the extra squares into a notch cannot pay.** Carving a staircase
   notch into the inner block lets a 45° square protrude `0.7071 - a` instead of
   `0.7071`, but every tuck costs at least one grid square and `n` is fixed, so
   the removed square must be paid back elsewhere; the measured tilted floor
   (`>= K+1.87` raw, `K+√2` best LP) shows the loss is not recovered.

This is consistent with earlier loops: Loop 32 measured that deleting the
right+top border of `n=969` leaves 908 squares at `30.643566` whose LP limit is
`30.640340` — still `0.0063` above the `n=908` incumbent; Loop 33 showed the
small-`m` `D=8` lift is attained at the largest available `m`, so no reusable
small gadget exists; Loop 22 showed the arm rectangles are not rectangle-bounded
(flat walls forbid the contacts that make those packings dense).

## Files

| path | role |
|---|---|
| `poc/minusE.py` | constructor (deletion patterns + block moves), parallel LP stage, SLP stage, both-mode certification |
| `poc/lp_core.py` | fixed-angle box LP + Lipschitz free-angle SLP (copied from `alln_fixed_lp` / `lip_slp`) |
| `poc/seam_probe.py` | deepest feasible `(-d,-d)` slide per cut/pattern + the `K+1/√2` bound table |
| `docs/scan.json`, `docs/scan_main.log` | main scan — all-upright (and gated) family |
| `docs/tilted/scan.json`, `docs/scan_tilted.log` | tilted-band family (`--no-gate-tilted`) |
| `docs/seam_probe.json` | seam-slide measurements (36 probes) |
| `final.csv` | header-only (nothing certified below the incumbent) |

## How to run

```sh
cd /mnt/e/squares-in-squares-commandcode
# main scan: LP + SLP + both-mode certification, 3 targets
.venv/bin/python -u poc/minusE_shear/poc/minusE.py \
    --n-list 492,908,328 --jobs 32 --max-cands 200 --top 6 --rounds 10
# tilted-band family (candidates far above the incumbent are normally gated out)
.venv/bin/python -u poc/minusE_shear/poc/minusE.py \
    --n-list 492,908,328 --jobs 32 --no-gate-tilted --outdir poc/minusE_shear/docs/tilted
# seam-slide probe + bound table
.venv/bin/python -u poc/minusE_shear/poc/seam_probe.py --n-list 492,908,328 --steps 20
```

`docs/scan.json` records, per `n`: number of assemblies, the minimum raw base
side over all of them, the best LP side, the best certified side, and the top-10
candidates. A candidate is only written as `docs/win_<n>.json` after it
certifies strictly below the incumbent in **literal** and **pandas transport**
mode; nothing was written.

## Assumptions

* `K = isqrt(n)`, `D = n - K²`, `G = K+1`; targets `n = 492, 908, 328`, with
  incumbents/frontier from `poc/koboshchan_coordinator/data/gap_vs_frontier.csv`.
* "Best certified side" is `Decimal(core.certify(...))` on rows produced by the
  pipeline; the LP side is only an optimization value and is never trusted
  (see the `22.366` artefact above).
* Certification: `poc/baseline/poc/src/squares/core.certify` plus
  `core.kaggle_roundtrip` for the pandas transport. No win → no `dL`.
