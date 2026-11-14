# smallD_lift — corner-gadget "lift" of `K^2 + D` packings

Status: **H1's premise is refuted; the lift itself is valid and yields 5 strict
certified wins (dust, `dL ≈ −5e-6 … −8e-6` plus one `−1e-7`).**
Nothing here is a `0.01–0.03` gain, and the mechanism that was hoped to produce
one does not exist in any public data.

Everything below was produced with the unchanged official metric
(`poc/baseline/poc/src/squares.core.certify`) and compared against the tracked
incumbent `poc/pell_corrugation/data/border-scan/merged.csv`.

---

## 1. The idea, stated exactly

For a target `n = K² + D`, put a **donor** packing of `p` unit squares with an
exact bounding box `W × H` in the bottom-left corner of the box and fill the
rest with upright unit squares anchored at that corner:

```
right arm : j columns at x = W, W+1, …      each holding K squares (y = 0..K−1)
top arm   : i rows    at y = H, H+1, …      each holding C = floor(W) squares
```

Nothing can overlap: the donor lives in `[0,W]×[0,H]`, the right arm in
`[W,W+j]×[0,K]`, the top arm in `[0,C]×[H,H+i]`, and `C ≤ W`.  Hence

```
L       = max(W + j, H + i, K)                     (box side)
count   = j·K + i·C + p      ≥ n   (squares are deleted to hit n exactly)
```

so a lift of the target exists whenever `j·K + i·C + p ≥ n`.  This is exact
integer counting — no interface constraint at all, because the donor touches the
arms only along the two lines `x = W` and `y = H`.

The classical "corner gadget" case of H1 is `j = i = K − m`, `C = m`,
`p = m² + D`, which gives

```
L(K² + D)  ≤  K + δ ,     δ = max(W − m, H − m) = L(m²+D) − m ,
```

**for every `K ≥ m`**.  So `δ(D) = min_{m ≤ K} [L(m²+D) − m]` is a universal
upper bound for the whole column, and any donor with `δ < L(K²+D) − K` is an
immediate certified win at that `n` (and often at several `n` at once).

Two hard consequences, which are exactly what the POC tested:

1. a donor must satisfy `m ≤ K`, so a gadget can only improve rows **at least
   as large as itself**;
2. the win condition is purely about the *excess* `c = L − K`, so H1 only pays
   if a small-`m` gadget has `c ≤ 0.6059`.

## 2. Method

### 2.1 `poc/search.py` — certified multi-start packing optimiser

* Exact penalty, no discretisation: `E(q,L) = Σ_{i<j} max(depth_ij + m,0)² +
  3·Σ walls²`, with `depth_ij` the exact separating-axis penetration of two
  unit squares (`poc/overlap.py`).  `E = 0` ⇔ valid packing of the box.
  The depth is evaluated as `½ + h(θi−θj) − max(|D·u|)`
  over the four axes, on all `O(n²)` pairs, with a `|D|² < 3` prefilter
  (`poc/overlap.py:penalty_fast`, cross-checked against the KD-tree path and
  against brute-force corner projection; gradients are analytic and verified
  against finite differences).
* Seeds: rotated unit lattices (16 angles, random phase), upright grids, greedy
  hard-square insertion, uniform random, the incumbent coordinates, and
  Gaussian perturbations of the best state found (true basin hopping).
* Repair = Adam (decreasing lr) + L-BFGS-B; **compression** = controlled
  `L → L − step` continuation (accept → grow `step`, fail → halve, deeper
  repair once `step` is small); **refinement** = bisection between the best
  feasible and the smallest infeasible side.
* Every feasible state is certified with `squares.core.certify`; the reported
  number is the *certified* side.  Because a jammed optimum has exact
  tangencies that the exact decimal metric rejects (see §5), certification
  falls back to scaling all centres about the bbox corner by `1+eps`
  (`eps ≤ 1e-7`), which costs `eps·L` of side and is guaranteed to separate
  every touching pair.
* `poc/quicktest.py` runs a `(D, m)` grid, one process per cell.

### 2.2 `poc/scan_lift.py`, `poc/closure.py` — the lift and its closure

`scan_lift.py` enumerates, with the count formula above, **every** `(target n,
donor, j, i)` over all 1000 tracked rows as donors (`n ≤ 1000`, `K ≤ 31`,
`j ≤ K`), ranks by `L − L_tracked(n)`, and then builds and certifies the winners
in *both* transports (literal decimal and `kaggle_roundtrip` pandas
`read_csv→to_csv`).  `closure.py` iterates the operation (a lift is itself a
legal donor) until nothing changes.

Implementation notes that matter:

* the donor keeps its **original decimal strings** (untranslated), so its exact
  internal contact geometry survives the metric bit for bit — re-emitting a
  translated copy breaks tangencies at the `1e-60` level and the certification
  fails;
* the donor is separated from the arms by `1e-12` and the arm lattice pitch is
  `1+1e-12`.  Exact tangency between *different* constructions is rejected by
  the interval certifier; `1e-12` of side is free at our resolution.

## 3. Result 1 — the small-`m` gadget of H1 does not exist

`quicktest.py --ms 8,9,10 --Ds 6,7,8 --seconds 120 --jobs 9`
(`data/quicktest3.json`, log `data/quicktest3.log`; 31–53 independent starts per
cell, each start = seed → repair → compress → bisect → certify):

| D | m | n = m²+D | best **certified** side | c = L−m | target m+0.6059 | shortfall | feasible at target? |
|--:|--:|--:|--:|--:|--:|--:|:--|
| 6 | 8 | 70  | 8.880960… | 0.880960 | 8.6059 | 0.275 | no |
| 6 | 9 | 87  | 9.838815… | 0.838815 | 9.6059 | 0.233 | no |
| 6 | 10 | 106 | 10.822908… | 0.822908 | 10.6059 | 0.217 | no |
| 7 | 8 | 71  | 8.944072… | 0.944072 | 8.6059 | 0.338 | no |
| 7 | 9 | 88  | 9.882451… | 0.882451 | 9.6059 | 0.277 | no |
| 7 | 10 | 107 | 10.846667… | 0.846667 | 10.6059 | 0.241 | no |
| 8 | 8 | 72  | 9.000000  | 1.000000 | 8.6059 | 0.394 | no |
| 8 | 9 | 89  | 9.949747… | 0.949747 | 9.6059 | 0.344 | no |
| 8 | 10 | 108 | 10.904821… | 0.904821 | 10.6059 | 0.299 | no |

The best certified side found **equals the tracked (and all-team best) value at
every one of the nine `n`** — the optimiser cannot even match those records from
scratch, let alone beat them by 3–4 %.  Runs started *at* the target side
(`data/at_target_n{70,71,72}.log`) never reach feasibility: every lattice/grid/
greedy/random seed at side `m+0.6059` is unrepairable.

The decisive evidence is not our optimiser, it is the data: `c(m)` for
`D = 6,7,8` from the tracked table (which **coincides with the all-team frontier
at every one of these `n`**):

```
D=6 : 0.881 0.823 0.778 0.728 0.707 0.692 0.657 0.637 0.634 0.609 0.609 0.591   (m = 8..30)
D=7 : 0.944 0.847 0.823 0.778 0.733 0.707 0.692 0.684 0.657 0.637 0.634 0.618
D=8 : 1.000 0.905 0.831 0.803 0.774 0.740 0.707 0.707 0.678 0.643 0.637 0.634
```

* `D=7`: `c` never reaches `0.6059` anywhere in `m ≤ 31` (min `0.608713`), and
  the best available with `m ≤ 28` is `0.634071` — i.e. `n = 791` itself.
* `D=8`: min `c = 0.633983` (whole column). `D=6`: `c ≤ 0.6059` only at
  `m = 30,31`, i.e. only the rows we already want to improve (a lift at `m = K`
  is the identity).
* `m ≤ 12` would need `c ≈ 0.61–1.0`: **0.22–0.39 of side** below the best
  packing anyone (including the private all-team envelopes) has at those `n`.

So H1 is not "hard to find": the required gadget is refuted by the best-known
table for every `m ≤ 12`, and uneconomical even at the `m` where the excess gets
small.  `c = 0.6059` is a property of the frontier's *interlocked* large-`K`
packings, not of a transplantable small corner gadget (also the frontier's `c`
is not actually `K`-independent: `D=7` gives `0.6313, 0.6267, 0.6059, 0.6059,
0.6023, 0.5999` for `K = 26..31`).

## 4. Result 2 — the lift is valid, and it does win (dust) on the rows where the
tracked table is internally inconsistent

`python poc/scan_lift.py --certify 11` then `python poc/closure.py`:

| n | K | D | donor p | j,i | L (lift) | L (incumbent) | **dL literal** | **dL pandas** | score gain |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 442 | 21 | 1 | 401 | 1,1 | 21.53302497513530126… | 21.53303030309420807… | **−5.327958907e−6** | −5.327958908e−6 | −2.53425e−7 |
| 485 | 22 | 1 | 401 | 2,2 | 22.53302497513630126… | 22.53303130309421007… | **−6.327957909e−6** | −6.327957910e−6 | −2.87338e−7 |
| 530 | 23 | 1 | 401 | 3,3 | 23.53302497513730126… | 23.53303230309421007… | **−7.327956909e−6** | −7.327956908e−6 | −3.18306e−7 |
| 577 | 24 | 1 | 401 | 4,4 | 24.53302497513830126… | 24.53303330309421007… | **−8.327955909e−6** | −8.327955910e−6 | −3.46697e−7 |
| 853 | 29 | 12 | 796 | 1,1 | 29.70588238054423134… | 29.70588248048808350… | **−9.994385217e−8** | −9.994385071e−8 | −3.42201e−9 |

* total portfolio gain: **−1.209188e−6** (all five are `dL ≤ −1e−9` in **both**
  transports, so they count as strict wins, but they are dust);
* `final.csv` is written from these five groups and then **re-read from disk**,
  degrees converted back to radians and re-certified: all five reproduce
  (`poc/make_final.py` prints `ALL WINNING GROUPS REPRODUCE`).

Cause: the tracked D=1 column is not self-consistent — `c` decreases with `n`
(`0.533024975` at `n = 401` vs `0.533033303` at `n = 442…577`), so the 401 row
lifts to all four larger `n`.  The 853 win is the same effect at D=12
(`n = 796` vs `n = 853`).

### The lift closes immediately, and it cannot touch the H1 target rows

`closure.py` (iterate the lift on the tracked table): round 1 finds these 5
wins, **round 2 finds no candidate at all**.  And for the rows H1 wanted
(`K = 28..31`, `D = 6,7,8`) the best lift available from the *entire* tracked
table is the row itself:

```
target n  K  D   tracked L      best lift L    gain          donor,j,i
     790 28  6   28.608712578   28.608712578   +0.000e+00    (790, 0, 0)
     791 28  7   28.634070747   28.634070747   +0.000e+00    (791, 0, 0)
     792 28  8   28.637029402   28.637029402   +0.000e+00    (792, 0, 0)
     847 29  6   29.606601718   29.606601718   +0.000e+00    (847, 0, 0)
     848 29  7   29.623065212   29.623065212   +0.000e+00    (848, 0, 0)
     849 29  8   29.636883108   29.636883108   +0.000e+00    (849, 0, 0)
     906 30  6   30.591378145   30.591378145   +0.000e+00    (906, 0, 0)
     907 30  7   30.618366065   30.618366065   +0.000e+00    (907, 0, 0)
     908 30  8   30.634070747   30.634070747   +0.000e+00    (908, 0, 0)
     967 31  6   31.591378145   31.591378145   +0.000e+00    (967, 0, 0)
     968 31  7   31.608712578   31.608712578   +0.000e+00    (968, 0, 0)
     969 31  8   31.633982501   31.633982501   +0.000e+00    (969, 0, 0)
```

(D=1 rows, for contrast, show `−4e−14`, i.e. numerical noise; the 5 real wins
above are all elsewhere.)

## 5. Conclusions

* **H1 is dead as a `0.01–0.03` lever.** The small-`m` gadget with
  `c ≈ 0.6059` is refuted by the best-known table for `m ≤ 12` (needed
  improvement 0.22–0.39 of side) and is not economical at the `m` where the
  excess becomes small, because a donor must satisfy `m ≤ K`.
* **The lift is nevertheless a correct and useful closure operation** and it
  found 5 strict certified wins worth `1.209e−6` of score, concentrated where
  the tracked table is internally inconsistent (`n = 442, 485, 530, 577, 853`).
  `final.csv` contains exactly those complete groups.
* **`poc/rectangle_gadgets`' "closure" claim is slightly too strong**: its DP
  uses integer pocket shapes and the strict `Σ(w+1) ≤ K+1` rule, so it misses
  the whole-row gadgets this lift uses; those give the 5 wins above.  Re-running
  `gaddp.py` with the exact `(W,H)` excess of every tracked row (this POC's
  `scan_lift.py`) is the right way to re-check closure after a merge — it takes
  ~10 s.
* **Practical warning for whoever works on the big rows next:** exact tangency
  between squares from *different* constructions is rejected by the interval
  certifier (re-emitted decimals move by ~1e-60).  Assemble multi-piece
  packings with a `1e-12` clearance and a `1+1e-12` lattice pitch, or the
  certificate will fail for a reason that has nothing to do with your geometry.

## 6. How to run

```bash
cd /mnt/e/squares-in-squares-commandcode
PY=.venv/bin/python

# 1. small-m gadget test (a (D,m) grid, one process per cell)
$PY poc/smallD_lift/poc/quicktest.py --ms 8,9,10 --Ds 6,7,8 --seconds 120 --jobs 9

# 2. single-packing optimiser (any n, certified output)
$PY poc/smallD_lift/poc/search.py --n 71 --side 8.95 --target 8.6059 \
      --seconds 60 --incumbent poc/smallD_lift/data/donors_small_n.csv \
      --out poc/smallD_lift/data/rows71.csv

# 3. lift scan + exact certification of every candidate (both transports)
$PY poc/smallD_lift/poc/scan_lift.py --certify 11

# 4. iterate the lift to closure (parallel workers)
$PY poc/smallD_lift/poc/closure.py --rounds 6 --jobs 12

# 5. write + re-verify final.csv
$PY poc/smallD_lift/poc/make_final.py
```

## 7. Files

| path | what |
|---|---|
| `final.csv` | the 5 winning groups (`id,x,y,deg`, `s`-prefixed decimals, complete groups) |
| `poc/overlap.py` | vectorised exact SAT penalty + gradients (fast all-pairs path) |
| `poc/search.py` | multi-start certified packing optimiser (basin hopping + shrink + bisect) |
| `poc/quicktest.py` | `(D, m)` grid driver, one process per cell |
| `poc/scan_lift.py` | exact lift construction, exhaustive enumeration, dual-transport certification |
| `poc/closure.py` | iterated (transitive) lift until closure |
| `poc/make_final.py` | writes `final.csv` and re-certifies it from disk |
| `data/quicktest{2,3}.json`, `data/quicktest*.log` | gadget-search results per `(D,m)` |
| `data/at_target_n*.log` | searches started exactly at the target side (all infeasible) |
| `data/lift_candidates.json` | every lift candidate with `dL < −1e−9` |
| `data/closure_wins.{json,pkl}`, `data/certified_wins.json` | accepted lifts with certified sides |
| `data/wins_summary.json` | per-`n` `L`, `dL` in both transports, score gain (source of the §4 table) |
| `data/bbox_tracked.json` | `(W,H)` of all 1000 tracked rows (the donor library) |
| `data/gadget_D*_m*.csv` | the best certified small-`m` packings found (evidence for §3) |
| `data/donors_small_n.csv` | tracked rows for `n ≤ 200` (seed material) |

Assumptions: "incumbent" is the tracked `merged.csv`; score gain is
`dL/√n`; nothing outside `poc/smallD_lift/` was modified and no `git` command
was run.
