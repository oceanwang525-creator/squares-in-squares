# dsmall_band — grid + one free-angle tilted band for small `D = n - K²` (negative)

**Verdict: negative.** No rigid "upright grid + one tilted band with a free
optimum angle" configuration reaches the required square count at any of the
eight small-`D` frontier sides. Nothing was certified, `final.csv` is
header-only, `merged.csv` untouched.

Targets (frontier sides from `poc/koboshchan_coordinator/data/gap_vs_frontier.csv`):

| n | K | D | our `L` | frontier `L` | frontier `c = L-K` | squares needed at the frontier side |
|---|---|---|---|---|---|---|
| 842 | 29 | 1 | 29.520815280171462 | 29.5021527079033 | 0.5021527 | 842 |
| 901 | 30 | 1 | 30.520815280171450 | 30.5021527079077 | 0.5021527 | 901 |
| 962 | 31 | 1 | 31.520815280171476 | 31.5021527079108 | 0.5021527 | 962 |
| 843 | 29 | 2 | 29.535533905932930 | 29.5182442831318 | 0.5182443 | 843 |
| 902 | 30 | 2 | 30.535533905932950 | 30.5140787741393 | 0.5140788 | 902 |
| 963 | 31 | 2 | 31.535533905932986 | 31.5140787741295 | 0.5140788 | 963 |
| 328 | 18 | 4 | 18.614368938490983 | 18.5995354648065 | 0.5995355 | 328 |
| 365 | 19 | 4 | 19.612856387671947 | 19.5988242989228 | 0.5988243 | 365 |

Note how small the gap actually is: at `n=901` the frontier needs **one more
square than the plain 30×30 grid** in a box 0.5022 larger than 30.

## Idea (H3)

Our D=1/D=2/D=4 rows are all built from a 45° tilted block wedged into a grid
(`n=901` has 289 squares at 45°, 612 upright). The frontier's `c` is 0.0186
smaller at D=1 and 0.017–0.021 smaller at D=2/D=4. If a *free* band angle (not
45°) plus a free band width, longitudinal phase and offset lets one more square
interlock, the D=1 group alone would pay ~0.0019 of score. So: does any
"upright grid + one free-angle band" hold `n` squares at the frontier side?

## Method (exact, not a count model)

All geometry is exact in float64 with a full separating-axis test; `L` is the
true bounding-box side of the constructed set. Three rigid sub-families were
scanned (each is a superset of the previous construction recipe):

* **v1 `poc/construct.py`** – a rotated `m × w` lattice block (pitch 1 in both
  rotated directions, so it is internally overlap-free by construction) at angle
  `theta`, plus *every* integer unit cell of the `ncol × ncol` grid
  (`ncol = floor(L)`) that does not overlap the block, plus ring caps (extra
  squares at `theta` on the lattice ring around the block) and single 45° corner
  caps. Free: `theta ∈ [4°, 86°]`, `m`, `w ∈ 1..17`, block bbox origin
  `(bx, by)` over the whole box.
* **v2 `poc/construct2.py`** – same block, but the upright cells are two
  corner-anchored staircase slabs (`i+j ≤ t` bottom-left, `(ncol-1-i)+(ncol-1-j)
  ≤ tp` top-right, channel gap `k = 0,1,2`), i.e. the block sits in a diagonal
  channel between the slabs.
* **v3 `poc/construct3.py`** – same block, but the upright grid is quartered:
  each quadrant is anchored to its own box corner, so the right/bottom quadrants
  are shifted by `d = L - ncol` (this is the only way a *rigid* upright fill can
  use the 0.5022 margin at all). Split column `ic` and row `jc` are free.

Scans (`poc/scan.py` for v1, `poc/scan2.py` for v2, `poc/scan_all.py` for all
three over every target): `theta` every 4–6° over `[4°, 86°]`, `m, w` over a
5×6…7×8 (`m ≤ 28`, `w ≤ 17`) grid, block bbox origin on a 1.0 (0.5 for
`K = 18/19`) lattice over the whole box — 0.3–1.0 M exact configurations per
family per target — plus ring/45° caps evaluated on the best configurations.
Raw results: `docs/scan_*.json`, `docs/summary.json`, `docs/scan_all.log`;
`docs/candidate_<n>.json` would hold an extracted candidate (none produced).

## Measured results

Best square count found at each frontier side (need = `n`), `docs/summary.json`:

| n | K | D | needed | v1 best | v2 best | v3 best | best overall | short by |
|---|---|---|---|---|---|---|---|---|
| 901 | 30 | 1 | 901 | 900 | 870 | 900 | **900** | 1 |
| 842 | 29 | 1 | 842 | 841 | 812 | 841 | **841** | 1 |
| 962 | 31 | 1 | 962 | 961 | 930 | 961 | **961** | 1 |
| 902 | 30 | 2 | 902 | 900 | 870 | 900 | **900** | 2 |
| 963 | 31 | 2 | 963 | 961 | 930 | 961 | **961** | 2 |
| 843 | 29 | 2 | 843 | 841 | 812 | 841 | **841** | 2 |
| 328 | 18 | 4 | 328 | 324 | 309 | 324 | **324** | 4 |
| 365 | 19 | 4 | 365 | 361 | 344 | 361 | **361** | 4 |

Every column's optimum is exactly `K²` — the plain `K × K` upright grid, with the
tilted band pushed into the margin where it no longer pays for itself. **No
configuration in any of the three families reaches `K² + D` squares at the
frontier side, i.e. the band never buys even one square.** The shortfall equals
`D` exactly at all eight targets (1, 1, 1, 2, 2, 2, 4, 4).

Calibration at *our own* sides (the family must at least reproduce the incumbent
there):

| target | side | need | v1 best | v2 best |
|---|---|---|---|---|
| n=901, our `L = 30.52081528017145` | 30.5208 | 901 | 900 | 870 |
| n=901, frontier `L = 30.5021527079077` | 30.5022 | 901 | 900 | 870 |

so the family is 1 square short even where the tracked row holds 901 — the
rigid grid+band recipe is *strictly weaker* than the tracked construction, and
the frontier is a further 0.0186 tighter.

### Why this family cannot close the gap

1. **A single-lattice upright grid is unbeatable, by construction.** With an
   upright grid anchored to one corner, a tilted square that touches a cell
   costs that whole cell, while it can only buy itself back if it lies wholly in
   the `L - ncol` margin — impossible for a unit square, whose bounding box is
   ≥ 1 in both axes while the margin is 0.5022. Hence
   `count = ncol² - blocked + m·w ≤ ncol²`, and the scans hit exactly `ncol²`
   at all eight targets. Free angle, band width, phase and offset change
   nothing: this is a counting obstruction, not a tuning issue.
2. **The margin is only usable by a shifted lattice, and the seam eats it.**
   v2 (block in a diagonal channel of width `d = L - ncol` between two
   corner-anchored staircase slabs) loses cells at the slab seam and tops out at
   870 / 812 / 930 / 309 / 344 — always *below* the plain grid. v3 (grid
   quartered, right/bottom quadrants anchored to the far box edges, so the shifted
   quadrants reach the far 0.5022 margin) merely ties v1 at `ncol²`: its seam
   costs every cell the channel gains. Neither family ever exceeds `K²`.
3. **The tracked D=1 rows are not a lattice.** Decomposing incumbent `n=901`
   (612 upright + 289 at 45°): the upright centres fall on *two* lattices whose
   phases differ by `d = 0.5208` in both axes (353 squares at one x phase and
   161 at the other; 314 / 174 in y), with the remainder polished off both — i.e.
   a free-form patchwork around the block, not a grid. Feeding the incumbent's
   **own** tilted block into any quadrant/shifted-lattice upright fill yields
   only **850** squares (847 for a single lattice) versus the 901 it actually
   holds, so the rigid fill is worth ≈51 squares less than the polished one at
   the same `L`. The frontier's extra square is therefore an interlocking/polish
   effect inside a free-form fill, i.e. exactly the LP/SA/SLP machinery that
   loops 1–40 already prove is a local optimum on this portfolio — not a band
   angle.

This is consistent with loops 26/30/37/38 (periodic strip cells, 2:1 rational
band + void filling, free-angle population search all negative) and with loop 29
(insertion into the `n=901` grid at 0.5022 of slack found nothing): the missing
squares are not a parameter-tuning issue inside rigid band families.

## How to run

```sh
cd poc/dsmall_band/poc
../../../.venv/bin/python scan_all.py                  # all 8 targets, writes docs/summary.json
../../../.venv/bin/python scan.py --n 901 --L 30.5021527079077 --tstep 4 --step 1.0 --jobs 30
../../../.venv/bin/python scan2.py --n 901 --L 30.5021527079077 --tag t
# certify one configuration (literal + pandas transport, vs the incumbent row)
../../../.venv/bin/python verify.py --n 901 --L 30.5021527079077 \
    --theta 45 --m 17 --w 17 --bx 6.5 --by 6.5
```

`verify.py` refuses to certify unless the constructed count equals `n`, then
reports `certify()` (literal Decimal) and `certify(kaggle_roundtrip(...))`
(pandas transport) against the incumbent row's certified `L`; it prints
`STRICT WIN` only when both `dL ≤ -1e-9`.

Caveat for any future candidate from a *lattice* constructor: emitting float64
`repr` coordinates of exactly touching squares can fail the official exact
metric (demonstrated by round-tripping the incumbent's own `n=901` coordinates
through float64: `infeasible: Squares overlap (certified)`, even though the
`Decimal` original certifies in both modes). A real candidate must be emitted
with exact/decimal coordinates or a contact-safe centre dilation
(`squares.core.prepare_kaggle`) — `L` then grows by ~`6e-7` at these sizes, which
is still far below the 0.014–0.021 wins this hypothesis was aiming for.

## Files

- `poc/geom.py` – SAT for unit squares, bbox/side helpers.
- `poc/construct.py` – v1 family (+ ring/45° caps).
- `poc/construct2.py` – v2 family (two corner staircase slabs + block).
- `poc/construct3.py` – v3 family (quartered grid, corner-anchored quadrants).
- `poc/scan.py`, `poc/scan2.py`, `poc/scan_all.py` – scans.
- `poc/verify.py` – extraction + both-mode certification vs the incumbent.
- `docs/scan_*.json`, `docs/summary.json`, `docs/scan_all.log` – raw results.
- `final.csv` – header-only (no certified strict win).

Note (correctness): the first scan pass marked blocked grid cells with a
transposed flat index (`i + j*ncol` vs `i*ncol + j`). On a square `ncol × ncol`
lattice the transposition is a bijection of the domain, so the *number* of
blocked cells — and hence every count reported here — is invariant; it only
mis-selected which cells were emitted in the packing array. The index is fixed
and the blocked-cell mask now matches a brute-force SAT mask exactly on 12
random configurations; `docs/recheck_901.log` re-runs the `n=901` target with
the fixed code and reproduces exactly `v1 = 900, v2 = 870, v3 = 900`.
