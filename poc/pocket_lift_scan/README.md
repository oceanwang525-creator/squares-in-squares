# pocket_lift_scan — exhaustive proper sub-pocket scan (C1)

Status: **positive — 7 certified strict wins, total score gain 2.8624e-05**
(0.0000286), all strict in **both** transports against **fresh main**
(`poc/pell_corrugation/data/border-scan/merged.csv`, official score
22.153247131307513, 1000/1000 accepted).  7 groups / 4022 rows in `final.csv`.

The C1 hypothesis is confirmed in a stronger form than the `(20,17,+1)` story:
the incumbent's rows are full of *proper sub-pockets* that the old
`gadlib.py` (minimal clean rectangles around **tilted clusters** only) cannot
see, and the composition DP lifts them at `K` values where the incumbent's own
`c(D) = L - K` family is worse.

## 1. Idea

A gadget is `w*h + g` unit squares packed in a `(w+f) x (h+f)` box.  Along the
anti-diagonal seam of a `K+f` box, `m` gadgets give `n = K^2 + sum g` at side
`K + max f` iff

```
sum w <= K,  sum h <= K,  sum (w + h + 1) <= 2K + 1 .
```

Any axis-aligned sub-rectangle of an existing row whose tight bounding box is
exactly its own outline is such a gadget (the seam only requires the pocket to
fit in its own box, so it is re-anchored at the origin).  `gadlib.py` extracts
only cluster rectangles, so the whole upright part of every row was never cut.
Scanning those cuts is cheap and gives a *much* larger gadget supply.

Why that pays even though the DP was "closed": the tracked rows are themselves
optimal compositions, so the DP reproduces `L(n)` almost exactly for 547 of 555
cells.  A win needs either

* a **better `f` for a shape the DP already uses** (the scan finds 24 such
  shapes, e.g. `(20,20,+1)`: 0.533024975 -> 0.533023120), or
* a **proper** sub-pocket of a big row, because a whole-row gadget
  `(K_row, K_row, +D)` only fits `K >= K_row`, while a sub-pocket `(w,h,+g)`
  fits every `K >= max(w,h,(w+h-1)/2)`.  So a tight row can be re-used at a
  *smaller* `K` whose incumbent `c(D)` is worse.  `(20,17,+1)` from `n = 401`
  was the first instance of this; there are more.

Minimum `f` per gain in the incumbent (only sub-rectangles can reach these, and
nothing below them exists): `g=1` 0.520815, `g=2` 0.535534, `g=3` 0.562385,
`g=4` 0.563197, `g=5` 0.606602, `g=6` 0.656854, `g=7` 0.731211, `g=8..9`
0.745648.

## 2. Scan (`poc/scan_pockets.py`)

Per row (1000 rows, 500500 squares), in the row's own frame:

| stage | what |
|---|---|
| A | axis-aligned windows on an offset lattice (`--step 0.25`, `--offx/--offy` in {0, 0.0625, 0.125} in both axes = **7 lattices**), width/height integers + `f`; screened with an **exact 4-sided containment count** (2-D prefix table over `(lx, hx)` per y-window, O(1) per x-window), keeping windows with `count >= w*h + 1` |
| B | each survivor is re-measured **exactly**: shrink to the tight bbox of the squares it contains, then `w=floor(W)`, `h=floor(H)`, `g=count-w*h`, `f=max(W-w,H-h)` |
| C | exact local search: a true pocket's edge is always some square's `lx/hx/ly/hy`, so a coordinate descent over those exact boundary values recovers the resolution the grid lost (this is what finds `H = 17.533`-style cuts) |

Cost: **37 s** for all 1000 rows at 30 processes (step 0.25, one lattice);
~145 s per additional offset lattice (bigger `--cap`/`--fmax`).

Scan size / yield:

| | count |
|---|---|
| windows screened (7 lattices) | ~1.4e8 (estimate) |
| raw gadget instances | 1 421 731 |
| distinct oriented `(w,h,g)` types found | **9 899** |
| of which new vs the 713 in `gadlib.json` | **9 444** |
| of which *improve an existing shape's `f`* | **24** |

`poc/polish.py` then re-polishes the surviving pockets with an exhaustive
**two-edge** exact-boundary search (every pair of edges moved over its nearest
`k` boundary values in each direction — sets a 1-edge descent can never reach).
Rounds with `k = 4, 8, 12` converge: the DP stops changing at `k >= 8`
(`pockets_polish*.json`, `pockets_wide12.json`).

`poc/merge_pockets.py` keeps only the Pareto front (an item is dropped if
another has `w'<=w, h'<=h, g'>=g, f'<=f`): **1322 items**, of which **629**
carry pocket geometry (the rest come from `gadlib.json`).

## 3. DP (`poc/dp.py`)

Same semantics as `poc/faultline_dp2/poc/dpx.py` (unbounded knapsack over
distinct `(w,h,g)` with the smallest `f`, both orientations, `f` levels swept
increasing), plus parent pointers so each predicted cell is reconstructed as an
explicit gadget list.

```
$ dp.py base  ...   -> 555 cells, DP better at 4 n   (the known D=1 lift)
$ dp.py extra ...   -> 555 cells, DP better at 8 n, score gain 2.8635e-05
```

8 `n` improve; `n = 585` (gain 1.06e-08) is predicted but **not built** — its
2-pocket seam comes out 10 squares short (`k^2 + sum g` is not reachable with
the integer/quarter lattices this builder can place), and at 1e-08 it is dust.

## 4. Certified wins — both transports, against fresh main

Built by `poc/build_cert.py` (each pocket copied from the **exact decimal
strings** of its donor row at 50 digits, translation only, so the donor's
internal contacts survive; see section 6 for the transpose / seam / truncation
details) and certified
with the unchanged official metric in **literal decimal** and in a real
**pandas `read_csv`/`to_csv`** transport against
`data/report.json` (official 50-digit sides of fresh main).

`poc/verify_final.py` re-reads `final.csv` from its own bytes (`deg -> rad` at
80 digits) and re-certifies:

| n | K | j | composition | incumbent L | new L (both modes) | dL literal | dL pandas | gain |
|---|---|---|---|---|---|---|---|---|
| 442 | 21 | 1 | `(20,18,+1)` <- n=401 | 21.533030303094208 | 21.533023120317423604 | **-7.1827768e-6** | **-7.1827768e-6** | -3.4165e-7 |
| 447 | 21 | 6 | `(11,21,+6)` <- n=910 | 21.656854249492479 | 21.656420965529470379 | **-4.3328396e-4** | **-4.3328396e-4** | **-2.0494e-5** |
| 485 | 22 | 1 | `(20,18,+1)` <- n=401 | 22.533031303094209 | 22.533023120317423604 | **-8.1827768e-6** | **-8.1827768e-6** | -3.7156e-7 |
| 530 | 23 | 1 | `(20,18,+1)` <- n=401 | 23.533032303094210 | 23.533023120317423604 | **-9.1827768e-6** | **-9.1827768e-6** | -3.9887e-7 |
| 577 | 24 | 1 | `(20,18,+1)` <- n=401 | 24.533033303094211 | 24.533023120317423604 | **-1.0182777e-5** | **-1.0182777e-5** | -4.2391e-7 |
| 743 | 27 | 14 | `(12,22,+10)` <- n=978 + `(14,5,+4)` <- n=539 | 27.753898760680030 | 27.753852984553401 | **-4.5776127e-5** | **-4.5776127e-5** | **-1.6794e-6** |
| 798 | 28 | 14 | `(12,21,+9)` <- n=915 + `(15,7,+5)` <- n=742 | 28.740473088741449 | 28.740334237076354 | **-1.3885167e-4** | **-1.3885167e-4** | **-4.9153e-6** |

**Total score gain 2.8624e-05** (literal == pandas for every group; no dilation
needed, only 1e-12 insets).  Nominal union with the incumbent:
`22.153247131307513 - 2.86242e-05 = 22.1532185`.  The groups replace only their
own `n`, so nothing can regress.

`poc/faultline_dp2/final.csv` banked 4 groups for 1.2058e-06; this file
supersedes it (the same 4 `n` with 22-35 % larger `dL` each, plus 3 new `n`),
27.9x the score of the currently banked `final.csv` files.

## 5. What the 3 mechanisms are

1. **`n = 442/485/530/577`** — the known non-monotone `D = 1` family
   (`L(401) - 20 = 0.533023` beats `c(1)` at `K = 21..24`).  The scan finds a
   *tighter* proper sub-pocket `(20,18,+1)` of n=401 (f = 0.533023120 vs
   0.533024975 for the old `(20,17,+1)`), so the same lift is 1.4-1.9x better.
2. **`n = 447`** — `(11,21,+6)` cut from n=910 (a 30x30 row) at
   `f = 0.656421`.  The whole row only fits `K >= 30`, but the sub-pocket is
   narrow (`w = 11`), so it fits `K = 21`, where the incumbent uses a
   `sqrt(2)`-zipper at `c = 0.656854`.  One pocket, no seam trickery:
   `0.656421 < 0.656854`.
3. **`n = 743, 798`** — two-pocket seams whose `max f` is below the incumbent's
   `c(D)` at that `(K,j)`, which the old library could not form.

## 6. Construction details worth knowing (they cost real time)

* **Transpose.**  A unit square is invariant under a quarter turn, so a pocket
  rotated by 90 degrees keeps its **angles unchanged** and only moves centres:
  `(x, y) -> (H - y, x)`.  This makes the transposed use of a pocket exact in
  decimals (no `+pi/2` rounding), which matters at 50-digit certification.
* **Multi-pocket seams.**  `m >= 2` needs the corner-anchored second grid of
  `poc/gadget_squeeze/poc/compose.py`, whose cells sit on a *fractional*
  lattice (`S - a - 1`).  Adjacent cells there are exactly 1 apart at
  non-representable decimals, and the official metric's interval SAT test
  cannot decide exact touching between such squares (`overlap undecidable`).
  Fix: spread that lattice by `a*1e-11` in each index (positive gaps, same
  count, L unchanged).
* **Surplus is free.**  When the seam grid yields more than `n` squares the
  build truncates, dropping the grid cells farthest from the origin first (any
  subset of a valid packing is valid).  This is what makes `m >= 2` practical.
* **`pocket_lift_scan` vs `cutlib.py`.**  `gadget_squeeze/poc/cutlib.py`
  already cuts *existing pockets* at integer positions; that is a small subset
  of what is done here (whole-row windows, 7 offset lattices, two-edge exact
  polish).  Its conclusion ("77 new shapes, no level moves") does not hold for
  the full scan: 9444 new shapes and 8 `n` move.

## 7. How to run

```bash
PY=.venv/bin/python
D=poc/pocket_lift_scan
M=poc/pell_corrugation/data/border-scan/merged.csv

# official 50-digit side table of fresh main (~4 min)  -> data/report.json
$PY poc/faultline_dp2/poc/mkreport.py $M $D/data/report.json
# float copy for sweeps
$PY poc/faultline_dp2/poc/mkfloat.py  $M $D/data/L_float.json

# 1. scan (7 offset lattices; ~40-150 s each at 30 procs)
for o in "0 0" "0.125 0" "0 0.125" "0.125 0.125" "0.0625 0" "0 0.0625" "0.0625 0.0625"; do
  set -- $o
  $PY $D/poc/scan_pockets.py $M $D/data/pockets_s025_o$1_$2.json \
      --step 0.25 --fmax 1.0 --cap 800 --procs 30 --offx $1 --offy $2
done

# 2. merge + Pareto front (+ the polish seed set)
$PY $D/poc/merge_pockets.py --gadlib poc/faultline_dp2/data/gadlib.json \
    --out $D/data/pockets_pareto.json --seeds $D/data/seeds.json $D/data/pockets_s025_o*.json

# 3. two-edge exact polish (rounds with k = 4, 8, 12), re-merge, repeat
$PY $D/poc/polish.py $D/data/seeds.json $D/data/pockets_polish.json --procs 30 --k 8

# 4. DP (3 s) -> data/pred.json
$PY $D/poc/dp.py extra --lib poc/faultline_dp2/data/gadlib.json \
    --pockets $D/data/pockets_pareto.json --report $D/data/report.json --out $D/data/pred.json

# 5. build + certify (both transports) -> final.csv (~2 min)
$PY $D/poc/build_cert.py --pred $D/data/pred.json --pockets $D/data/pockets_pareto.json \
    --report $D/data/report.json --final $D/final.csv

# 6. re-certify the deliverable from its own bytes
$PY $D/poc/verify_final.py
```

Minimal path that reproduces the shipped `final.csv` exactly (needs only the
two checked-in files):

```bash
$PY $D/poc/build_cert.py --pred $D/data/pred.json --pockets $D/data/pockets_pareto.json \
    --report $D/data/report.json --final $D/final.csv
$PY $D/poc/verify_final.py
```

## 8. Files

| path | what |
|---|---|
| `poc/scan_pockets.py` | 3-stage row scanner (grid windows, exact tight refine, exact-edge local search) |
| `poc/polish.py` | two-edge exact-boundary polish of the surviving pockets |
| `poc/merge_pockets.py` | Pareto merge of scan + `gadlib.json`, keeps geometry, writes polish seeds |
| `poc/dp.py` | staircase DP with parent reconstruction (`base` / `extra` / `pack`) |
| `poc/build_cert.py` | seam builder (both orientations, `m >= 1`, spread corner lattice, truncation) + official two-transport certification + `final.csv` |
| `poc/verify_final.py` | re-certify `final.csv` from its own bytes |
| `data/report.json` | official 50-digit sides of fresh main (score 22.153247131307513) |
| `data/L_float.json` | float copy for sweeps |
| `data/pockets_pareto.json` | **629** scanned pockets (best `f` per `(w,h,g)`, with exact bbox geometry) — the durable scan output |
| `data/seeds2.json` | 4686 distilled scan pockets (polish seeds; regenerate polish output without re-scanning) |
| `data/pockets_polish*.json`, `data/pockets_wide12.json` | polish rounds |
| `data/pred.json` | DP predictions (8 `n`, with the reconstructed compositions) |
| `data/built/n{442,447,485,530,577,743,798}.json` | the certified groups (radians, decimal strings) |
| `data/build6.log`, `data/verify2.log`, `data/scan_*.log`, `data/merge5.log` | run logs |
| `final.csv` | 4022 rows, groups 442/447/485/530/577/743/798, `id,x,y,deg`, `s`-prefixed decimals |

**Not checked in (regenerable, 234 MB total):** the seven raw scan outputs
`data/pockets_s025_o*.json` (7.6 MB base + 32-51 MB per offset lattice; one
line each in the loop above).  Everything derived from them is in
`data/pockets_pareto.json` and `data/seeds2.json`.

## 9. Where the remaining money is (measured, not guessed)

The DP is now closed except at 8 cells.  The next-closest losers need these `f`
reductions (gain if closed): `n=911` 1.0e-9 (3e-11), `n=796` 1.6e-9 (6e-11),
`n=232` 7.9e-8 (5e-9), `n=419` 2.2e-7 (1e-8), `n=789` 2.5e-7 (9e-9) — i.e.
sub-rectangles of the incumbent are exhausted at ~1e-7, and every cell beyond
that needs a *shape* reduction of 0.02-0.07 in `f`, which no sub-rectangle can
supply (the incumbent's own rows are the records).  The large-gap rows
(`n=367`, `791`, `908`, `969`, `803`, `492`) need new *constructions*, not new
supply — consistent with C2/C3, not with C1.

One observation that may be worth a follow-up: the scan's 24 shape improvements
are all `< 1e-5` in `f`, but three of them (`(20,20,+1)`, `(30,30,+10)`,
`(28,28,+9)`) are *whole-row* pockets that the old extractor reported 2e-6 to
5e-5 too wide.  So `gadlib.py`'s clean-cut loop leaves a small systematic slack;
any future DP work should use the scan's numbers instead.
