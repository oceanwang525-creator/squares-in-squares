# rational_zipper — does a rational-slope `p/q` seam zipper beat the incumbent? (E3)

**Verdict: negative. Nothing certifies; `final.csv` is header-only (no `dL` exists
to report in either transport).** The hypothesis is not just unproductive, it is
arithmetically impossible for the family as stated (a uniform single-layer seam
zipper):

1. For a **single-layer seam zipper** — two rigid unit-square blocks separated
   across a seam of slope `q:p` and filled with squares whose edges are parallel
   to the seam — the perpendicular separation must be `>= 1` (a unit square's
   width in any direction is `>= 1`), and the axis cost is
   `min over separation vectors of max(u,v) = h/(p+q)`, `h = sqrt(p^2+q^2)`.
   That is `>= 1/sqrt(2)` with equality **only for `p == q`** (45°). *Every*
   other rational slope is strictly worse; among Pythagorean (rational-side)
   slopes the best is `(20,21,29)` at `29/41 = 0.707317073` — i.e. `2.1e-4`
   **worse** than the `1/sqrt(2)` the incumbent already has at the rows that sit
   on the 45° zipper. Exact table in §2.
2. The premise's numbers are true but mean something different: the `22.705882`
   n=492 build is a **two-pocket composition** whose coefficient is the *larger
   of two donor pockets'* `f`, and that `f = 0.7058824 = 12/17 + 1.2e-9` is
   inherited from the donor row `n=585`, which is itself a `12:5` **pocket
   family** member (§1, §3). The rational constant is real — it is `p/(p+q)` for
   slope `q:p` — but it lives in the *pocket composition* rule, not in a seam
   band, and that rule is already the incumbent's own family.
3. Measured, not just argued: the shipped composition DP over all 1699 Pareto
   pockets reproduces the fresh-main incumbent **exactly** (deficit `~1e-13`) at
   20 of the top-gap rows and is *worse* at the other 4 (492, 974, 975, 852), so
   no composition of known tilted blocks improves any of them (`docs/reach.md`);
   and the independent `poc/free_band_ga` / loops 30+37+38
   scans of the digital-staircase + tilted-band family top out 8-11 squares
   **below** the needed count at the n=908/911 side.

Fresh-main baseline regenerated for this POC (50-digit side table, snapshot kept):
`data/report.json`, `mkreport.py poc/pell_corrugation/data/border-scan/merged.csv`,
**score `22.15258085488673`**, 1000/1000 accepted, missing 0, rejected 0,
merged.csv sha256 `c9e5398775e31b536cbfa1519ab9c6066c95f98edf1d7bc9bec3a54a85cafd9e`.

---

## 1. The n=492 template, exactly

Source: `poc/pocket_lift_scan2/data/built/n492.json` (492 rows) and the DP
prediction that produced it, `poc/pocket_lift_scan2/data/pred2.json`.

```
n = 492,  K = isqrt(492) = 22,  D = 492 - 484 = 8
side = 22 + f_max = 22.705882354152596      (declared)
composition (pred2.json, both pockets used TRANSPOSED):

  pocket A = (w,h,g) = (5,13,+3)   <- n=585   f = 0.7058823541515906
  pocket B = (w,h,g) = (16,9,+5)   <- n=686   f = 0.6967165817586718
  grid     = 275 integer cells (the two L-shaped corners of the 22x22 grid)
  check    : Sigma w = 21 <= 22,  Sigma h = 22 <= 22,
             Sigma(w+h+1) = 45 <= 2K+1 = 45   (saturated)
             Sigma g = 8 = D;   217 pocket squares + 275 grid = 492
```

The two pockets are **axis-aligned windows of donor rows**, copied at 50 digits
by translation only. Their declared windows (from
`poc/pocket_lift_scan2/data/pockets_pareto2.json`) are

| donor | window (x0,y0)-(x1,y1) | bbox | counted squares | `(w,h,g)` | `f` |
|---|---|---|---|---|---|
| 585 | (0, 19.0000)-(13.6851, 24.7059) | 13.685 × 5.706 | 68 | (13,5,+3) | 0.705882354152 |
| 686 | (0, 10.0000)-(9.6967, 26.6967) | 9.697 × 16.697 | 149 | (9,16,+5) | 0.696716581759 |

Verification status (honest): `poc/decode492.py` re-cuts both windows and searches
for the rigid embedding into `built/n492.json`. It finds the **angle content**
exactly (donor 686's `68.6884°` / `67.4135°` blocks appear as 29 + 9 squares;
donor 585's `60.268°` as 24 + 15 at `60.273-60.275°`) but **not** a verbatim
translate (best 61/149 and 0/68 at `1e-7` tolerance; 1/149 at `1e-3`), so the
builder re-fit / re-phased the pockets rather than copying them rigidly and the
per-square placement is not reproducible from the checked-in data. The
composition bookkeeping itself is exact and self-consistent
(`Sigma w, Sigma h, Sigma(w+h+1), Sigma g`, `217 + 275 = 492`), and that is what
§1-§3 rely on.

So the "seam" is not a band: it is two dense rectangular slabs butted against a
275-cell integer lattice, with the **worst** of the two slab excesses setting the
side. The geometry inside the slabs is whatever the donors are:

* `n=585` (itself `24.705882373 = 24 + 12/17 + 2e-8`): 96 squares at `22.6199° =
  atan(5/12)`, 50 at `60.268°`, 227/91 axis-aligned — i.e. a **12:5 slope
  family** row, and `12/17 = p/(p+q)` for `q:p = 5:12`;
* `n=686` (`26.696717`): 29 squares at `68.6884°`, 298/142 axis-aligned;
* the built row's own angle histogram (`poc/analyze.py`-style, deg mod 90):
  319 axis-aligned, 39 at `60.273-60.275°`, 29 at `68.688°`, 9 at `67.414°`,
  remainder in small clusters (no `45°` squares at all).

**Two corrections to hypothesis E3.**

* The win is a *pocket composition*, not a rational-slope seam. There is no
  `p/q` seam in it; the rational constant is a donor row's own coefficient.
* It is **superseded**: fresh main's `n=492` is `22.701529381`, which is
  `-4.35e-3` *better* than the `22.705882354` build (verified: the composition
  DP's best side at n=492 is `22.705882354` vs report `L=22.701529381`, so
  upstream has a packing outside this family). The E3 premise row is therefore
  no longer a live win at all.

## 2. Exact cost of a rational-slope seam zipper

Two rigid blocks separated by a vector `(u,v)`, void filled with one layer of
unit squares at angle `theta = atan(q/p)`. Required perpendicular separation
`rho = (u p + v q)/h >= 1`; axis cost `c = max(u,v)`; therefore

```
c_zip(p,q) = h/(p+q)      (minimised at u = v = h/(p+q))
           >= 1/sqrt(2) = 0.707106781187,   equality iff p == q
2(p^2+q^2) - (p+q)^2 = (p-q)^2 >= 0     <-- the exact inequality
```

`poc/slope_table.py` (exact `Fraction`, q <= 40):

```
 slope q:p        c_zip                     note
 1:1              0.707106781187            45 deg, irrational; the only minimiser
 39:40            0.707163429036            best non-45 rational slope, irrational c
 38:39            0.707166409873
 ...              ...                       (all p != q are worse)
```
Pythagorean slopes (the only ones with a **rational** side `K + p/q` style):

```
 (20,21,29)  slope 20:21   29/41  = 0.707317073171   <-- best rational-side zipper
 (48,55,73)  slope 48:55   73/103 = 0.708737864078
 (3,4,5)     slope  3:4     5/7    = 0.714285714286
 (28,45,53)                53/73   = 0.726027397260
 (8,15,17)                 17/23   = 0.739130434783
```

So a rational-slope zipper is **strictly dominated** by the 45° zipper at the
same `K`, by at best `2.1e-4` of side and typically far more. This is the direct
answer to "if a rational slope `p/q` zipper beats their coefficient, it is a
large win": it cannot, because `1/sqrt(2)` is already the infimum over slopes and
the portfolio's `1/sqrt(2)` rows are exactly the minimiser.

Caveat, stated honestly: the bound is for a **uniform single-layer seam**. A
seam whose void is locally wider (a stepped/notched seam) can beat the floor
locally — that is what the portfolio's small-`n` rows do (`n=26` at
`5+0.62132`, `n=401` at `20+0.53302`, `n=585` at `24+12/17`). Those are
*corner-tuck* constructions, not zippers, they are already in the portfolio, and
the composition DP over every window of every row is closed on them (§4).

## 3. The rational constant is `p/(p+q)`, and it is a pocket constant

`c = p/(p+q)` for a slope `q:p` matches the tracked rows and the frontier:

| row | `c = L - K` | `p/(p+q)` | the slope is visible in the row |
|---|---|---|---|
| n=585 (`24+…`) | 0.705882373 | **12/17** (q:p = 5:12) | 96 squares at `atan(5/12) = 22.6199°` |
| n=853 (`29+…`) | 0.705882420 | 12/17 | same family |
| n=974 (`31+…`) | 0.705886746 | 12/17 (+4e-6) | 67 at `22.6199°`, 105 at `66.38°` |
| n=913 (`30+…`) | 0.707106781 | ≈ 1/√2 | 23 at `22.613°`, 23 at `67.387°` |
| frontier D=1 rows | 0.5022 | **1/2** (q:p = 1:1) | (frontier, not in our portfolio) |
| frontier D=4 rows | 0.599535 | **3/5** (q:p = 2:3) | idem |
| frontier D=7/8 rows | 0.605885 / 0.605571 | 3/5 + finite-size | idem |

Two consequences:

* The **prize for matching the frontier** is closing the gap from the incumbent's
  constant to the next smaller `p/(p+q)` family (down to `1/2`). Per target that
  is `c_inc - 0.5`, i.e. `0.10 - 0.35` of side — 1e-3..1e-2 of score each. This
  is real money and it is exactly the coordinator's 0.053 remaining gap.
* It is a **pocket/free-form** target, not a seam target: `c = p/(p+q)` is a
  property of a whole dense slab (the way our donors' slabs have `f = 12/17`),
  and our slabs' constants (0.634071 = the `(13,16,+4)` pocket family, 0.705882 =
  `(13,5,+3)`/`(18,10,+6)`) are all above the frontier's.

I did **not** manage to construct a `p/(p+q)` slab: reproducing the geometry
requires either the donor rows themselves (only `12/17` exists there) or a
free-form search at the frontier's density (0.96-0.97), which is the known
unsolved piece. No certified claim is made.

## 4. Per-target table (fresh main, `docs/targets.md`, `data/targets.json`)

`floor = K + 1/sqrt(2)` is the best a single-layer *any-slope* zipper can do;
`D_zip(K)` is the measured capacity of one seam layer, taken from the portfolio's
own rows that sit **exactly** at `K + 1/sqrt(2)` (21 to `1e-13`, plus n=913 at `+2e-11`)
(`n = 5,10,27,38,52,67,104,125,149,174,201,231,262,296,331,369,408,449,493,586,635,913`
→ `(K,D) = (2,1),(3,1),(5,2),(6,2),(7,3),(8,3),(10,4),(11,4),(12,5),(13,5),(14,5),(15,6),(16,6),(17,7),(18,7),(19,8),(20,8),(21,8),(22,9),(24,10),(25,10),(30,13)`,
i.e. `D_zip(K) ~ 0.433 K`; `n=493` is literally "484 upright + 9 at 45°").

| n | K | D | L_inc | c_inc | c_front | c_inc−1/√2 | K+1/√2 | D_zip | squares at the floor | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 791 | 28 | 7 | 28.634070747 | 0.634071 | 0.605885 | −0.0730 | 28.707107 | 12 | 796 | provably worse |
| 908 | 30 | 8 | 30.634070747 | 0.634071 | 0.605885 | −0.0730 | 30.707107 | 12 | 912 | provably worse |
| 969 | 31 | 8 | 31.633982501 | 0.633983 | 0.605571 | −0.0731 | 31.707107 | 13 | 974 | provably worse |
| 803 | 28 | 19 | 28.817412499 | 0.817412 | 0.793943 | +0.1103 | 28.707107 | 12 | 796 | capacity short by 7 |
| 328 | 18 | 4 | 18.614368938 | 0.614369 | 0.599535 | −0.0927 | 18.707107 | 7 | 331 | provably worse |
| **492** | 22 | 8 | 22.701529381 | 0.701529 | 0.690540 | −0.0056 | 22.707107 | 9 | 493 | provably worse |
| 861 | 29 | 20 | 29.817033760 | 0.817034 | 0.795415 | +0.1099 | 29.707107 | 12 | 853 | capacity short by 8 |
| 365 | 19 | 4 | 19.612856388 | 0.612856 | 0.598824 | −0.0943 | 19.707107 | 8 | 369 | provably worse |
| 367 | 19 | 6 | 19.692388155 | 0.692388 | 0.670857 | −0.0147 | 19.707107 | 8 | 369 | provably worse |
| 975 | 31 | 14 | 31.718773696 | 0.718774 | 0.701629 | +0.0117 | 31.707107 | 13 | 974 | capacity short by 1 |
| 747 | 27 | 18 | 27.810663459 | 0.810663 | 0.793760 | +0.1036 | 27.707107 | 11 | 740 | capacity short by 7 |
| 804 | 28 | 20 | 28.827155402 | 0.827155 | 0.810061 | +0.1200 | 28.707107 | 12 | 796 | capacity short by 8 |
| 974 | 31 | 13 | 31.705886746 | 0.705887 | 0.688380 | −0.0012 | 31.707107 | 13 | 974 | provably worse |
| 848 | 29 | 7 | 29.623065212 | 0.623065 | 0.605884 | −0.0840 | 29.707107 | 12 | 853 | provably worse |
| 456 | 21 | 15 | 21.846667193 | 0.846667 | 0.834218 | +0.1396 | 21.707107 | 9 | 450 | capacity short by 6 |
| 852 | 29 | 11 | 29.695584615 | 0.695585 | 0.682145 | −0.0115 | 29.707107 | 12 | 853 | provably worse |
| 366 | 19 | 5 | 19.656854249 | 0.656854 | 0.645802 | −0.0503 | 19.707107 | 8 | 369 | provably worse |
| 228 | 15 | 3 | 15.604601476 | 0.604601 | 0.596043 | −0.1025 | 15.707107 | 6 | 231 | provably worse |
| 795 | 28 | 11 | 28.703403923 | 0.703404 | 0.687289 | −0.0037 | 28.707107 | 12 | 796 | provably worse |
| 982 | 31 | 21 | 31.810539458 | 0.810539 | 0.793106 | +0.1034 | 31.707107 | 13 | 974 | capacity short by 8 |
| 262 | 16 | 6 | 16.707106781 | 0.707107 | 0.698115 | 0.0000 | 16.707107 | 6 | 262 | **exact tie** (the 45° zipper itself) |
| 790 | 28 | 6 | 28.608712578 | 0.608713 | 0.593240 | −0.0984 | 28.707107 | 12 | 796 | provably worse |
| 263 | 16 | 7 | 16.733166008 | 0.733166 | 0.724373 | +0.0261 | 16.707107 | 6 | 262 | capacity short by 1 |
| 965 | 31 | 4 | 31.562050659 | 0.562051 | 0.544500 | −0.1451 | 31.707107 | 13 | 974 | provably worse |

Global, over all 1000 rows: **207 have `c_inc < 1/sqrt(2)`** — no rational-slope
seam zipper can even tie them; **21 sit exactly at `K + 1/sqrt(2)`** (22 counting n=913 at
`+2e-11`; the family above), so a zipper can at best *tie* those; the other **772** need
`D > D_zip(K) ~ 0.433 K` extra squares, which one seam layer cannot supply. A
second layer costs another `1/sqrt(2) = 0.707` of side (perpendicular separation
1 per layer), which is `6x..60x` more than the 0.012..0.12 of budget the
"capacity short" rows would need. Concretely for the two closest misses:
`n=975` (K=31, D=14) would need 1 extra square at `31.7071`, and a second layer
would pay `+0.7071` for `~13` squares — the incumbent (`31.7188`) is 100x closer
than that; same for `n=263` (K=16, one square short at `16.7071`).

## 5. Composition reachability (measured, `docs/reach.md`)

`poc/reach.py` runs the shipped staircase composition DP (`dp.dp_levels`) over
`gadlib.json` ∪ `pockets_pareto2.json` (591 + 1539 items → **1699** after the
Pareto merge) and reports, per top-gap target, the best side `K + f` reachable.

```
   n    K   D      L_inc          K+f_best       deficit      win
 791   28   7  28.634070747    28.634070746     -8.6e-10    False
 908   30   8  30.634070747    30.634070746     -8.3e-10    False
 969   31   8  31.633982501    31.633982501     -9.0e-12    False
 803   28  19  28.817412499    28.817412499     -7.1e-15    False
 328   18   4  18.614368938    18.614368938     +1.4e-14    False
 492   22   8  22.701529381    22.705882354     +4.4e-03    False   <- the E3 premise build
 974   31  13  31.705886746    31.707106781     +1.2e-03    False   <- 45-deg zipper is 1.2e-3 worse
 975   31  14  31.718773696    31.721540002     +2.8e-03    False
 852   29  11  29.695584615    29.699128038     +3.5e-03    False
 ...    (24 targets; every deficit is 0 +/- 1e-13 or positive)
```

Two readings: (a) at all but a handful of the top-gap rows the DP's best side
equals the incumbent to `1e-13`, i.e. those rows *are* compositions of the
pocket library — the family is saturated, not under-explored; (b) where the DP is
worse than the incumbent (492, 974, 975, 852) the incumbent already uses
geometry outside the composition family, and a zipper would be a step **back**.

## 6. What I tried, and what is left

Tried (all negative): exact rational-slope cost enumeration (§2); exact
per-target capacity accounting against the portfolio's own zipper rows (§4);
composition DP over the full pocket library (§5); decoding and cross-checking the
n=492 build (§1).

Also re-ran an exact-integer counter for the digital-staircase + tilted-band
family (`poc/ratzip.py`, all co-prime `(p,q)` with `q <= 40`, band layers
`w = 0..6`, gaps, 2 phases, full sweep over the staircase offset `A`, box side =
`L_inc - 1e-9`; `data/ratzip.json`). It reaches **exactly the plain grid** at 5
of 6 targets and `+2` at n=803, against `D = 4..19` required:

| n | L = L_inc − 1e-9 | best count found | plain grid `K²` | n needed | short |
|---:|---:|---:|---:|---:|---:|
| 492 | 22.701529 | 484 | 484 | 492 | 8 |
| 791 | 28.634071 | 784 | 784 | 791 | 7 |
| 908 | 30.634071 | 900 | 900 | 908 | 8 |
| 969 | 31.633983 | 961 | 961 | 969 | 8 |
| 328 | 18.614369 | 324 | 324 | 328 | 4 |
| 803 | 28.817412 | 786 | 784 | 803 | 17 |

Caveat: that counter models the two sub-grids on one lattice without the extra
*tangential shear* the true 45° zipper uses, so its numbers are a lower bound on
the family, not a proof — which is exactly why it is not used as the argument.
The independent `poc/free_band_ga` (free-angle GA over the same family, at the
n=911 frontier side) and loops 30/37/38 (2:1 band, void filling) also top out at
895-900 < 908/911, so the family is measured, not merely modelled, to sit below
the incumbent near these sides. The load-bearing arguments are §2 (exact) and
§4/§5 (exact counting against fresh main).

Left for someone with a proper free-form engine (this is the real lever, and it
is not a zipper):

* **build a `p/(p+q)` slab.** Target constants, best first: `1/2` (45°, worth
  0.10-0.20 of side on the `D<=4` rows), `3/5` (2:3 slope, worth ~0.03 on the
  `D=7,8` rows: `n=791/908` would go `28.634 → 28.605` and `30.634 → 30.606`),
  `5/8`, `7/12`. Feasibility test: a slab holding `w*h+g` squares in a
  `(w+f)x(h+f)` box at `f = p/(p+q)` is a density-0.96 packing — the donors
  already show `12/17` at density 0.93, so `3/5` needs a top-team-quality patch.
* the frontier's own route: waste `~0.93 K` per row, i.e. one wasted square per
  row — free-form, not compositional.

## 7. Files

| path | what |
|---|---|
| `poc/slope_table.py` | exact `c_zip = h/(p+q)` table (Fraction), `p/(p+q)` family constants, per-target `slopes.json` |
| `poc/targets.py` | per-target verdict table (§4) → `docs/targets.md`, `data/targets.json` |
| `poc/reach.py` | composition-DP reachability of the shipped pocket library (§5) → `data/reach.json` |
| `poc/mkreachmd.py` | renders `docs/reach.md` from `data/reach.json` |
| `poc/decode492.py` | window decode + embed check of the n=492 build (§1) |
| `poc/ratzip.py` | exact integer counting for the digital-staircase + tilted-band family (no shear; kept for the audit) |
| `poc/analyze.py` | exact rotated bbox of any `merged.csv` group (`analyze.py 492 974 …`) |
| `data/report.json` | **fresh-main snapshot**, 50-digit sides, score `22.15258085488673` |
| `data/slopes.json`, `data/targets.json`, `data/reach.json`, `data/ratzip.json` | the tables above |
| `docs/targets.md` | the §4 table |
| `docs/reach.md` | the §5 table (rendered) |
| `data/mkreport.log` | baseline regeneration log |
| `final.csv` | **header only** (`id,x,y,deg`) — nothing certified |

Total data footprint < 200 KB.

## 8. Run commands

```bash
PY=.venv/bin/python
D=poc/rational_zipper

# baseline snapshot (267-280 s)
$PY poc/faultline_dp2/poc/mkreport.py poc/pell_corrugation/data/border-scan/merged.csv $D/data/report.json

# exact rational-slope cost table + per-target family analysis
$PY $D/poc/slope_table.py

# per-target verdict table
$PY $D/poc/targets.py --md $D/docs/targets.md

# composition reachability of the shipped pocket library (~3 s)
$PY $D/poc/reach.py

# decode the n=492 build
$PY $D/poc/decode492.py

# exact rotated bbox of any group (used for every coefficient above)
$PY $D/poc/analyze.py 492 974 791 367 328
```

## 9. Honest read

E3 as literally stated ("a rational slope `p/q` zipper with `p/q < 1/sqrt(2)`")
cannot exist for a single-layer seam: the cost is `h/(p+q) >= 1/sqrt(2)`, with
the 45° (irrational `1/sqrt(2)`) zipper as the unique minimiser, and the best
rational-*side* slope `(20,21,29) → 29/41` loses by `2.1e-4`. The real rational
structure in the winning rows is `c = p/(p+q)` of a **dense slab**, worth
1e-3..1e-2 per affected row if a `3/5` or `1/2` slab could be built — that is a
free-form packing problem at density 0.96-0.97, not a seam problem, and this POC
produced no such slab, hence no certified `dL` in either transport.
