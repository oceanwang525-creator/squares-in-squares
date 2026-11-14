# faultline_dp2 — staircase DP over an expanded small-gadget library (H4)

Status: **negative for the hypothesis as stated, but 4 certified strict wins**
(1.21e-06 of score, dust scale) came out of the DP closure check itself.
No *new* gadget with a lower `f` was found, and the available search is
demonstrably too weak to decide the question.

## 1. Idea (H4)

The leader ("Cats and boxes", alexchilton) builds large-`n` packings by placing
gadgets along an anti-diagonal staircase. `poc/rectangle_gadgets` showed the
tracked rows are already gadget compositions, and that their DP reports closure:
no combination of the incumbent's own sub-gadgets beats any tracked row. Hence
the hypothesis: the missing wins must come from **new, better gadgets**; search a
small randomised library of low-`f` small gadgets (`w,h <= 10`, `g in 1..4`), add
them to the DP, and read off every `n` that improves.

Composition rule (from `poc/rectangle_gadgets/README.md`, m gadgets in a `K+f` box):

```
sum w <= K,   sum h <= K,   sum (w + h + 1) <= 2K + 1      =>  n = K^2 + sum g
```

## 2. Environment and reproduction of the closure result

`numba` is **not installed** in `.venv` (only numpy 2.5 / scipy 1.18 / pandas 3.1),
so `poc/rectangle_gadgets/poc/gaddp.py` cannot run here. Everything was
re-implemented with numpy, with identical semantics (unbounded knapsack over
distinct `(w,h,g)` items with the smallest `f`, both orientations, `f` levels
swept in increasing order, each `n` keeps the smallest `f` that reaches it):

| file | role |
|---|---|
| `poc/mkreport.py` | official 50-digit side table of the incumbent (`poc/pell_corrugation/data/border-scan/merged.csv`) → `data/report.json`; **score 22.15325720501231, 1000/1000 accepted** |
| `poc/mkfloat.py` | same table in float64 (`data/L_float.json`), 13 s, for sweeps |
| `poc/gadlib.py` | verbatim copy of the POC extractor (sub-gadget library, 892-style) |
| `poc/gaddp.py` | straightforward numpy port of the POC DP (reference, ~3–5 min) |
| `poc/gaddp_numba_reference.py` | untouched copy of `poc/rectangle_gadgets/poc/gaddp.py` (needs numba, cannot run here) |
| `poc/dpx.py` | fast DP: Pareto pruning + one item pass per `K` (3 s), plus `base`/`thr`/`extra` modes |
| `poc/gadsearch.py` | free-angle penalty search for a gadget (`a*b+g` squares in `(a+f)x(b+f)`) |
| `poc/squeeze.py` | donor-squeeze probe: delete squares from a library pocket, shrink `w` |
| `poc/build_pocket.py` | builds and certifies the full packing from a chosen gadget |

Library rebuilt from the **current** incumbent (`data/gadlib.json`):
**889 gadgets, 334 proper sub-gadgets, 101 distinct proper `(w,h,g)` types**
(the POC's `docs/gadget_types.txt` lists 95 — the tracked file has changed since).

### Closure check

```
$ .venv/bin/python poc/faultline_dp2/poc/dpx.py base \
    --lib poc/faultline_dp2/data/gadlib.json \
    --report poc/faultline_dp2/data/report.json
items: 713 raw -> 587 after pareto (+0 extra)
predicted cells 555; DP strictly better at 5 n; score gain 1.227e-06
  n=577 K=24 j=1: ours 24.533033303 -> 24.533024975 (gain 3.467e-07)
  n=530 K=23 j=1: ours 23.533032303 -> 23.533024975 (gain 3.183e-07)
  n=485 K=22 j=1: ours 22.533031303 -> 22.533024975 (gain 2.873e-07)
  n=442 K=21 j=1: ours 21.533030303 -> 21.533024975 (gain 2.534e-07)
  n=585 K=24 j=9: ours 24.705887939 -> 24.705887436 (gain 2.078e-08)
```

So the environment is validated and the closure claim reproduces **to dust
level**: the DP is better at 5 `n`, but only by 5–8e-06 in `L` (total 1.23e-06 in
score), i.e. composition of the existing library is exhausted for all practical
purposes — exactly the POC's conclusion.

### Where those 5 come from (and why the POC saw 0)

The `D = 1` family of the incumbent is **non-monotone in `K`**:

```
L(401) - 20 = 0.533024975   (K=20)
L(442) - 21 = 0.533030303   (K=21)   <- 5e-06 worse
L(485) - 22 = 0.533031303
L(530) - 23 = 0.533032303
L(577) - 24 = 0.533033303
L(626) - 25 = 0.520815280   (K>=25 is the good flat family)
```

The library contains the **proper sub-pocket `(20,17,+1)` of n=401** (341 squares
in `20.533024975 x 17.533024975`, `f = 0.533024975`) besides the whole-row gadget
`(20,20,+1)`. For any `K in 21..24` the single-pocket composition
`sum w = 20 <= K`, `sum h = 17 <= K`, `sum(w+h+1) = 38 <= 2K+1` is legal, so
`n = K^2 + 1` is reachable at side `K + 0.533024975`. That is the whole trick: a
*proper* sub-gadget lifts where the whole row cannot (no seam problem, because it
is placed flush in the top-right corner and the `K x K` grid is cut back to the
L-shape `{i < K-w} u {j < K-h}` = `K^2 - w*h` squares).

This also corrects `poc/loop_log.md` loop 33, which asserted that
`delta(D) = min_m [L(m^2+D) - m]` is always attained at the largest available `m`
("our c decreases monotonically with m"). For `D = 1` it is attained at `m = 20`,
not `m = 31`, and the corner lift therefore wins at `K = 21..24`.

## 3. Certified wins (both transports)

Built by `poc/build_pocket.py` — the `(20,17,+1)` pocket (copied from the exact
decimal strings of the n=401 row at 50 digits, so its internal contacts are
unchanged, with a 1e-12 inset off the grid) plus the L-shape grid; then certified
with the unchanged official metric in **literal decimal** and in **actual pandas
read_csv/to_csv transport** (`squares.core.certify`, `squares.core.kaggle_roundtrip`).

| n | incumbent L | new L (literal) | new L (pandas) | dL literal | dL pandas | gain |
|---:|---|---|---|---:|---:|---:|
| 442 | 21.53303030309420807270940782919147366610233310356856 | 21.53302497513530126125136532229486935419616894914978 | 21.53302497513530008146932293629671836274942030139601 | **-5.328e-06** | **-5.328e-06** | -2.5343e-07 |
| 485 | 22.53303130309421007270940782919147366610233310356856 | 22.53302497513530126125136532229486935419616894914978 | 22.53302497513530008146932293629671836274942030139601 | **-6.328e-06** | **-6.328e-06** | -2.8734e-07 |
| 530 | 23.53303230309421007270940782919147366610233310356856 | 23.53302497513530126125136532229486935419616894914978 | 23.53302497513530008146932293629671836274942030139601 | **-7.328e-06** | **-7.328e-06** | -3.1831e-07 |
| 577 | 24.53303330309421007270940782919147366610233310356856 | 24.53302497513530126125136532229486935419616894914978 | 24.53302497513530008146932293629671836274942030139601 | **-8.328e-06** | **-8.328e-06** | -3.4670e-07 |

Total certified score gain **1.20578e-06** (all four are strict in both modes, no
dilation needed). Groups: 2034 rows in `final.csv` (`442, 485, 530, 577`).

`n = 585` (gain 2.08e-08) is **not** built: it needs the `j = 9` composition the
DP used (several gadgets), which `build_pocket.py` does not assemble; at 2e-08 it
is not worth it.

## 4. Step 2 — searching for *new* lower-`f` gadgets (negative / inconclusive)

### 4a. What a new gadget is worth now

`poc/rectangle_gadgets/docs/value_map.md` advertises 0.01–0.03 score for shapes
such as `(4,7,3)` and `(4,13,3)`. Those numbers are stale (they were computed
against an older, worse tracked table and with the over-strict composition rule).
Recomputed exactly against the **current** incumbent with the correct rule
(`dpx.py thr`, bisection on the largest `f` that improves anything, base DP
recomputed at every step):

| shape | pays only if `f <=` | best achievable gain | `n` improved at that `f` | density needed at the threshold |
|---|---:|---:|---|---:|
| (4,7,3) | 0.777734 | 6.68e-05 | 150, 203 | 0.838 |
| (4,11,3) | 0.739844 | 8.28e-05 | 587 | 0.845 |
| (4,13,3) | 0.692188 | 2.15e-05 | 330, 367 | 0.856 |
| (4,16,3) | 0.655469 | 6.55e-05 | 447 | 0.864 |
| (4,19,3) | 0.632031 | 8.09e-05 | 582 | 0.871 |
| (5,13,3) | 0.690625 | 9.20e-05 | 367 | 0.846 |
| (9,10,2) | 0.603906 | 1.90e-04 | 102 | 0.903 |

(`data/thr_*.json`, `data/thr.log`, `data/thr2.log`.) The library's own best
comparable pockets sit just above these lines, e.g. `(5,13,+3)` at 0.705887 and
`(5,7,+3)` at 0.764187 — the incumbent already owns most of the value, which is
why each shape would now win at only 1–2 `n`.

So even a *perfect* new gadget of these types is worth ~1e-05…1e-04 of score, two
to three orders of magnitude below the value map's claim. The upside of H4 under
the current tracked table is bounded at ~1e-04, not 0.01–0.03.

### 4b. Search attempts

* `poc/gadsearch.py` — free-angle penalty relaxation (vectorised numpy port of
  `poc/rectangle_gadgets/poc/rgad.py`'s model: squared corner penetration +
  squared wall violation, best of the four separating axes; scipy L-BFGS-B) with
  random, `a x b`-grid-plus-strip and tilted-lattice starts, shrinking `f` towards
  a target with warm starts.
  **Calibration fails**: for `(4,4,1)` (17 squares, known record `f = 0.67553`)
  the search reaches only `f = 1.0356` in 45 s x 6 chains, and `(3,4,1)` only
  `1.0125`. The searcher is 0.35 in `f` away from the known value on a 17-square
  problem, so a failure on 31–90 square shapes carries no information. (The
  original POC saw the same weakness: 0.769 vs 0.6755 for n=17 with numba.)
* `poc/squeeze.py` — stronger warm start: take the library pocket closest to the
  target, delete the squares that stick out, shrink the box. Best result
  `(4,7,3)`: **f = 0.843561** (150 s, 5 donors/deletions modes) versus the
  0.777734 pay line — a factor 1.08 too wide, i.e. no win, but this is a single
  heuristic, not a bound (`data/squeeze_4_7_3.json`).
* **Closest near-miss, and it is rigid.** `(9,10,+2)` needs `f <= 0.603906`
  (worth 1.9e-04 at n=102) and the incumbent's own `(10,9,+2)` pocket from n=102
  is at `f = 0.605829` — only **0.32 %** too wide. Warm-starting from that pocket
  and compressing with the penalty model accepts **no** step at all (200 s, f
  stays 0.605829; no output file, since nothing improved): the pocket is at a local optimum,
  exactly like every whole-row packing under fixed-angle LP / Lipschitz SLP in
  `poc/loop_log.md`. (Careful: a gadget's `(w,h)` in `gadlib.json` is the pocket
  orientation, so the n=102 pocket is stored as `(10,9)`; passing `(9,10)` makes
  `f` look 1.0 too large.)

**Verdict**: no new gadget beats the library with the search available here, and
the search is demonstrably too weak for a meaningful negative. The honest status
is *inconclusive*, with the value of a win now known to be <= 1e-04.

### 4c. Bug found in the reused searcher (worth fixing upstream)

`poc/rectangle_gadgets/poc/rgad.py::eg` has a wrong `dtheta` sign in the `a >= 2`
branch of the pair gradient: the `ro(phi=tj-ti)` term is added with the wrong sign
to **both** squares (the correct law is `d pen/d ti = -dro - bs*dproj`,
`d pen/d tj = +dro - bs*dproj`). Checked against finite differences: the port in
`poc/gadsearch.py` with the fix has a relative gradient error of 2e-08, the
original has O(1) errors on the affected pairs. Energy values are unaffected, so
the POC's reported `f` values stand; only its descent direction was degraded.
(Same family of bug as loop 23's `free_angle_slp` sign error.)

## 5. How to run

```
PY=.venv/bin/python; D=poc/faultline_dp2
# official side table of the incumbent (8 min, cached in data/report.json)
$PY $D/poc/mkreport.py poc/pell_corrugation/data/border-scan/merged.csv $D/data/report.json
# sub-gadget library from the incumbent (14 s)
$PY $D/poc/gadlib.py poc/pell_corrugation/data/border-scan/merged.csv $D/data/gadlib.json
# closure check (3 s)
$PY $D/poc/dpx.py base --lib $D/data/gadlib.json --report $D/data/report.json
# exact pay line for a hypothetical shape (2 min)
$PY $D/poc/dpx.py thr --lib $D/data/gadlib.json --report $D/data/report.json \
   --shape 4,7,3 --lo 0.6 --hi 0.95 --n 8
# DP with extra gadgets, listing which n they improve
$PY $D/poc/dpx.py extra --lib $D/data/gadlib.json --report $D/data/report.json --extra extra.json
# build + certify the 4 winning groups and rewrite final.csv
$PY $D/poc/build_pocket.py --n 442 485 530 577 --final $D/final.csv
```

`extra.json` is a list of `{"w":a,"h":b,"g":g,"f":f,"src":"label"}`.

## 6. Files

- `data/report.json` — official 50-digit sides, incumbent (score 22.15325720501231)
- `data/gadlib.json` — 889-gadget library from the incumbent (21 MB)
- `data/base_pred.json` — DP cell table; `data/thr_*.json` — pay lines
- `data/built/n{442,485,530,577}.json` — the certified groups (radians, decimal strings)
- `data/squeeze_4_7_3.json` — best donor-squeeze attempt
- `final.csv` — 2034 rows, groups 442/485/530/577, `id,x,y,deg`, `s`-prefixed decimals

## 7. Next steps that could actually pay

1. **Exhaustive proper-pocket scan.** `gadlib.py` only cuts clean rectangles
   around tilted clusters; the `(20,17,+1)` pocket that produced all four wins is
   a sub-rectangle of a *whole* row. Scan every tracked row for the best
   `(w,h,g)` pocket over all window placements (needs the 2-D window/anchor
   machinery, ~1000 donors x a few hundred shapes). Given that one such pocket
   was worth 1.2e-06 and the incumbent still has non-monotone `c(D)` families,
   this is the cheapest remaining source of certified strict wins.
2. **Fix the search before re-testing H4.** The penalty search needs a working
   engine (numba or a batched Adam over ~1000 simultaneous restarts) to get near
   published small-`n` records, otherwise no negative result here is trustworthy.
3. Do **not** invest further in new `(4,*,3)`-type gadgets for score: their entire
   remaining value against the current table is <= 1e-04.
