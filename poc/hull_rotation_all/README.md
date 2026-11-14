# C3 — rigid rotation of the whole packing, audited on all 1000 rows

**Result: negative, and stronger than expected. There is no row in the
portfolio where a rigid rotation of the whole packing lowers the official side
`L = max(width, height)`. The best reduction over all 1000 rows is
`1.8e-14` (float-noise level); the *raw* float64 screen's largest apparent gain,
`2.9e-10` on n=798, is an artefact of the screen and certifies as a
**regression** (`dL = +3.1e-10`) in both transport modes. `final.csv` is
header-only.**

Fresh main reference (recomputed here, not copied):
`poc/pell_corrugation/data/border-scan/merged.csv`,
literal total `22.153247131307513`, pandas-transport total `22.153247131307516`,
1000/1000 accepted (`data/fresh-main-literal.json`, `data/fresh-main-pandas.json`).

## Method

`poc/audit.py`, for each of the 1000 incumbent groups:

1. Build the 4n square corners, take their convex hull `H` (4–24 vertices,
   median 5 — these hulls are very simple).
2. `width(t) = max_i p_i·u − min_i p_i·u`, `height(t) = width(t + π/2)`.  On a
   `t`-interval where the four active (argmax/argmin) hull vertices are fixed,
   `width = A cos t + B sin t`; `width > 0` on such an interval forces
   `|t − φ| < π/2`, so each branch is **concave**.  The interval boundaries are
   exactly the hull edge normal directions taken mod π/2 (which covers the
   height events too).
3. On each interval `max(width, height)` is one concave branch between
   consecutive zeros of `width − height`, so its minimum is attained in
   `{interval endpoints} ∪ {width == height crossings}`.  Solve the crossing
   `(ax − hy) cos t + (bx + hx) sin t = 0` per interval (active-vertex
   coefficients) and evaluate the candidate set exactly.  This is an exhaustive
   candidate set, not a sampling.
4. Cross-check every row against a dense 1441-point sweep of `[0, π/2]`; the
   analytic minimum must be ≤ the dense minimum.  No row violates it
   (`inconsistent_rows: []`).  Nineteen rows — every large-`n` row named in the
   brief (791, 908, 969, 803, 861, 492, 328, 365, 367, 902, 963) plus the six
   float-screen rows and 936/1000 — were additionally brute-forced at 20001
   angles with a **second, independent** implementation (`poc/brute_force.py`:
   rotate the centres, add `t` to the angles, form the extents from the rotated
   angles, no hull, no event enumeration).  Result: `brute_min == L` at
   `t = 0.000000°` for all nineteen, matching the audit to 12 decimals
   (`data/brute_force.json`).
5. Any row whose float64 gain exceeds `1e-13` is re-evaluated in 50-digit
   arithmetic (`hp_sides`).  This step matters: near a kink `x cos t − y sin t`
   with a tiny `cos t` loses ~1e-10 absolute accuracy, which is exactly the size
   of the "gains" that survive the float screen.

`poc/corner_tightness.py` adds the structural diagnostics (corner gaps, wall
spans, kink certificate). `poc/make_table.py` merges both into
`data/audit_table.csv`.

## Audit result (all 1000 rows)

| quantity | value |
|---|---|
| rows with `L_official − min_t max(width,height) > 1e-9` | **0** |
| … `> 1e-12` | **0** |
| … `> 1e-13` (float, before 50-digit re-check) | 6 |
| … after the 50-digit re-check | **0** (all six become ≤ 0) |
| largest reduction, any row (50-digit verified) | `1.78e-14` (n=750) |
| largest raw float64 gain before verification | `2.86e-10` (n=798) |
| total score gain if every row were taken | `4.5e-15` |
| rows with a square axis-aligned bounding box (`w == h == L`) | 951 |
| rows with a non-square bbox (`w ≠ h`, L = max) | 49 = nine exact `L×(L−1)` integer rectangles (n = 2, 6, 12, 20, 30, 42, 56, 72, 90) + 40 rows with `|w − h| ≤ 9.6e-10` (the incumbents' own polish slack) |
| rows where the hull is *strictly non-square* in the rotation sense (worst `min_t` below `L`) | **0** |

So the operative answer to "for how many rows could rotation matter" is **zero**,
even though 49 rows have a non-square bounding box: rotating a near-square
rectangle increases both extents, so `max(width, height)` still cannot fall.

### Why: the wall-hinge (kink) mechanism

The hulls are **not** corner-tight — the largest distance from a bounding-box
corner to the nearest square corner is a full `1.0` and the median is `0.122`
(779 rows > 1e-6), so the "corner-tight ⇒ rotation-proof" argument does not
apply. The actual mechanism is that the extreme points are **ties on the walls**.
Write the wall extents from the lower-left box corner as right wall
`[r1, r2]`, left `[l1, l2]`, bottom `[c1, c2]`, top `[d1, d2]` (spans of the
corner points lying on that wall). The chords through those points give, for
`t > 0`,

```
width(t)  >= Lx cos t + (r2 - l1) sin t      width(-t)  >= Lx cos t + (l2 - r1) sin t
height(t) >= Ly cos t + (c2 - d1) sin t      height(-t) >= Ly cos t + (d2 - c1) sin t
```

If all four brackets are positive, `max(width, height)` has a strict **kink**
minimum `L` at `t = 0`: any rotation increases it. **974/1000 rows** satisfy all
four inequalities (median of the smallest bracket: `13.74`, i.e. a huge margin;
35 rows have a wall touched at a single point and 26 fail the local test — all
of those are still covered by the exhaustive enumeration of step 3, which finds
no gain either).

## Certification (both modes, against fresh main)

`poc/certify_rotation.py` emits the rotated packing — centres and angles rotated
rigidly in 60-digit arithmetic, bbox translated to the origin, optional uniform
dilation `1+δ` about the box centre to survive decimal transport — as 40-decimal
literals, and certifies **literal** and **pandas `read_csv`/`to_csv` transport**
against the fresh-main `L` for that `n` in the *same* mode.

Rows certified: the six rows with any float signal (798, 499, 585, 542, 744,
746) and the eleven large-gap rows named in the brief (791, 908, 969, 803, 861,
492, 328, 365, 367, 902, 963), each at its audit angle and its best candidate
angle, at `δ ∈ {0, 1e-12, 1e-11}` — **51 certification trials, 0 wins**.

| n | θ (deg) | δ=0 `dL` literal / pandas | verdict |
|---|---|---|---|
| 798 | 89.999999999 | `+3.117e-10` / `+3.117e-10` | worse |
| 499 | 1.7e-9 | `+2.355e-10` / `+2.355e-10` | worse |
| 585 | 89.999999996 | `+1.344e-10` / `+1.344e-10` | worse |
| 542 | ~0 | `+3.144e-12` / `+3.142e-12` | worse |
| 744 | 90.0 | `+1.418e-11` / `+1.418e-11` | worse |
| 746 | ~0 | `+1.651e-13` / `+1.637e-13` | worse |
| 791, 908, 969, 803, 861, 492, 328, 365, 367, 902, 963 | 0 | `0` literal (`±1e-41`), `≤ 5e-16` pandas | neutral (null control) |

* **No certified win.** `min dL_literal = -2.1e-41` (noise of the 40-decimal
  re-serialisation) and `min dL_pandas = +4.1e-16` over all 51 trials — nothing
  is anywhere near the `dL ≤ -1e-9` bar. Every trial with `θ ≠ 0` is strictly
  *worse*; the rows whose audit angle is 0 reproduce the incumbent exactly
  (`dL_literal = 0` to 1e-41, `dL_pandas ≤ 5e-16`), a null control that the
  emit path is exact.
* Dilation costs `2.8e-11` at `δ=1e-12` and `2.8e-10` at `δ=1e-11` (i.e. `δ·L`),
  which by itself already exceeds the largest float-screen "gain".
* Side observation: at `δ=0` five rows (328, 365, 367, 902, 963) are
  **infeasible after pandas transport** — re-serialising the incumbent centres
  as 40-decimal literals lands on different float64 doubles and their contacts
  become certified overlaps. They pass again at `δ=1e-12`. Those rows carry no
  slack at all, which is consistent with the audit: a rotation would have to
  spend its entire (zero) margin on transport safety.

## Step 3 — rotate, then fixed-angle LP re-shrink

`poc/rotate_then_lp.py` (imports `poc/alln_fixed_lp/poc/alln_fixed_lp.py`
read-only) rotates a row, then re-shrinks with the fixed-angle box LP.

| n | θ | LP side | official certification of the LP solution |
|---|---|---|---|
| 492 | 0° | 22.7071068046 | ok, `L = 22.7071068026` (LP margin `+2.3e-8`) |
| 492 | 5° / 22.5° / 45° | 23.41 / 23.83 / 24.69 | **infeasible** (squares overlap) |
| 908 | 0° | 30.6340707783 | ok, `L = 30.6340707763` (`+2.9e-8`) |
| 908 | 5° / 22.5° / 45° | 31.62 / 26.07 / **21.21** | **infeasible** (squares overlap) |
| 791 | 0° | 28.6340707761 | ok, `L = 28.6340707749` (`+2.8e-8`) |
| 791 | 5° / 22.5° / 45° | 29.46 / 30.33 / 23.63 | **infeasible** (squares overlap) |

The dramatic rotated LP sides (21.2 vs 30.63) are relaxation artefacts: the LP
constrains only pairs that were within `PAIR_RADIUS = 1.6` *before* the solve,
so after a rotation-driven reshuffle it returns overlapping configurations that
the official metric rejects — every rotated LP solution fails with
`infeasible: Squares i and j overlap`. Nothing certifies; rotation does not open
a new LP basin, and at `θ=0` the LP sits `+2.3e-8` above the incumbent (its
`MARGIN = 1e-9` per constraint).

## Why the conclusion is robust

`min_t max(width(t), height(t))` is invariant under rigid motion of the packing,
so this audit covers *every* rotation of every incumbent row exactly once — it
is not a stochastic search. Combined with the fact that the official metric is
exactly `max(width, height)` of the placed squares, a rigid rotation can only
ever be a loss for this portfolio. The remaining mechanical degrees of freedom
(non-rigid reshaping, new constructions) are outside C3.

## Files

| path | contents |
|---|---|
| `poc/official_L.py` | fresh-main per-n official `L` (literal or `--pandas`) |
| `poc/audit.py` | the rotation audit (`data/audit.json`) |
| `poc/corner_tightness.py` | corner gaps, wall spans, kink certificate (`data/corner_tightness.json`) |
| `poc/make_table.py` | merges the two into `data/audit_table.csv` |
| `poc/brute_force.py` | independent 20001-angle cross-check of 19 rows |
| `poc/certify_rotation.py` | emit rotated groups + both-mode official certification |
| `poc/rotate_then_lp.py` | rotate-then-LP experiment |
| `data/audit.json`, `data/corner_tightness.json`, `data/audit_table.csv` | audit data (≈1.3 MB total) |
| `data/brute_force.json` | independent cross-check output |
| `data/fresh-main-{literal,pandas}.json` | fresh main reference `L` per n, both modes |
| `data/certification.json`, `data/rotate_then_lp*.json` | certification / LP evidence |
| `docs/*.log` | raw run logs |
| `final.csv` | **header only** — nothing to certify |

## Run commands

```sh
cd /mnt/e/squares-in-squares-commandcode

# 1. fresh-main reference (both modes) — 2 x ~3 min on 32 cores
.venv/bin/python poc/hull_rotation_all/poc/official_L.py \
  poc/pell_corrugation/data/border-scan/merged.csv \
  poc/hull_rotation_all/data/fresh-main-literal.json
.venv/bin/python poc/hull_rotation_all/poc/official_L.py \
  poc/pell_corrugation/data/border-scan/merged.csv \
  poc/hull_rotation_all/data/fresh-main-pandas.json --pandas

# 2. audit all 1000 rows (≈20 s incl. the 50-digit re-checks)
.venv/bin/python poc/hull_rotation_all/poc/audit.py \
  poc/pell_corrugation/data/border-scan/merged.csv \
  poc/hull_rotation_all/data/audit.json \
  --official poc/hull_rotation_all/data/fresh-main-literal.json

# 3. structural diagnostics + compact table
.venv/bin/python poc/hull_rotation_all/poc/corner_tightness.py \
  poc/pell_corrugation/data/border-scan/merged.csv \
  poc/hull_rotation_all/data/audit.json \
  poc/hull_rotation_all/data/corner_tightness.json
.venv/bin/python poc/hull_rotation_all/poc/make_table.py \
  poc/hull_rotation_all/data/audit.json \
  poc/hull_rotation_all/data/corner_tightness.json \
  poc/hull_rotation_all/data/audit_table.csv

# 3b. independent brute-force cross-check of the named rows (≈15 s)
.venv/bin/python poc/hull_rotation_all/poc/brute_force.py \
  poc/pell_corrugation/data/border-scan/merged.csv \
  poc/hull_rotation_all/data/audit.json \
  poc/hull_rotation_all/data/brute_force.json

# 4. emit + certify rotations in both modes (≈5 min)
.venv/bin/python poc/hull_rotation_all/poc/certify_rotation.py \
  --incumbent poc/pell_corrugation/data/border-scan/merged.csv \
  --audit poc/hull_rotation_all/data/audit.json \
  --literal poc/hull_rotation_all/data/fresh-main-literal.json \
  --pandas poc/hull_rotation_all/data/fresh-main-pandas.json \
  --ns 798,499,585,542,744,746,791,908,969,803,861,492,328,365,367,902,963 \
  --top-candidates 1 --deltas 0,1e-12,1e-11 \
  --output poc/hull_rotation_all/data/certification.json

# 5. rotate-then-LP
.venv/bin/python poc/hull_rotation_all/poc/rotate_then_lp.py \
  --incumbent poc/pell_corrugation/data/border-scan/merged.csv \
  --audit poc/hull_rotation_all/data/audit.json \
  --output poc/hull_rotation_all/data/rotate_then_lp_certified.json \
  --ns 492,908 --rounds 2 --certify
```

## Caveats

* Screening is float64 on the 16–17-digit decimal inputs; the float64 `L` agrees
  with the 50-digit official `L` to `3.9e-14` (checked on all 1000 rows), and
  every row that survived the screen by more than `1e-13` was re-evaluated in
  50-digit arithmetic before being reported.
* The audit is exhaustive over rotations for *one fixed set of squares*; it says
  nothing about non-rigid changes, and nothing about packings other than the
  incumbent.
* `poc/midrange_hull_rotation` reached the same conclusion for n=100–200 and for
  archived (not incumbent) layouts; this audit extends it to the full incumbent
  portfolio and, importantly, shows that the six remaining float-level "wins"
  are numerical artefacts rather than the start of a signal.
* `data/corner_tightness.json` flags exactly those five artefacts as
  `dense_violations` — [499, 542, 585, 744, 798], the rows where the 2881-point
  independent sweep disagrees with the float64 candidate evaluation — which is
  the signal to distrust the float64 number and go to 50-digit + certification.
  After the 50-digit re-check and the official both-mode certification, none of
  them is a gain.
