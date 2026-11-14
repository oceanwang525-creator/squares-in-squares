# pocket_lift_scan2 — scaled proper-sub-pocket scan (D1)

Status: **positive — 10 certified strict wins, total score gain 8.4507e-05**
(2.95x the C1 `pocket_lift_scan` result of 2.8624e-05). Every win is strict
(`dL <= -1e-9`) in **both** transports — literal decimal strings and a real
pandas `read_csv`/`to_csv` round trip — against **fresh main**
`poc/pell_corrugation/data/border-scan/merged.csv` (regenerated 50-digit side
table via `poc/faultline_dp2/poc/mkreport.py`: score **22.153247131307513**,
1000/1000 accepted, missing 0, rejected 0). 10 groups / 5839 rows in `final.csv`.

Hypothesis D1 ("the sub-pocket supply is far from exhausted; scale the C1
pipeline") is **confirmed**, but only through two of its four levers:

| lever | result |
|---|---|
| finer offset lattices (step 0.125, phase 0.03125/0.09375) | **yes** — found the `(9,16,+5)` pocket that produces the two new wins |
| bigger `cap` (2000) / `fmax` (1.10) | neutral by itself, needed to keep the finer lattices' seeds alive |
| 3+ pockets per seam | the DP already does it (unbounded knapsack); no 3-pocket seam ever wins — but **2-pocket seams now win twice** (`n=492`, `n=740`) after a builder fix |
| iterate with built rows as new donors | **negative** — 7 built rows gave 11 785 raw gadgets / 4 743 distinct types, yet only 4 undominated items and *zero* DP change |
| L-border propagation (`n' = n+2K+1`) | **negative** for these wins (see §6) |

## 1. What was run

Everything is a copy of the C1 pipeline (`scan_pockets.py`, `polish.py`,
`merge_pockets.py`, `dp.py`, `build_cert.py`, `verify_final.py`) with two changes
plus three new tools:

* `build_cert.py` — the seam layout is now searched: **every pocket order × every
  staggering of the `m-1` extra columns** (`layouts_all`), keeping the one that
  leaves the largest integer grid. This is what makes `n=492` buildable
  (it was 13 squares short with C1's single staggering). No regression: C1's
  staggering is one of the candidates.
* `build_cert.py` — `--outdir` now points inside this folder.
* new `border_prop.py` — L-border propagation with `GAP=1e-9`, pitch `1+1e-12`
  (a `1e-12` gap is *not* enough: the official metric answers "overlap
  undecidable" on exactly-touching border squares — measured, see §6).
* new `mkdonors.py` — donor CSV = fresh main + every built group (as `n=2000+i`),
  used for the iteration scan.
* new `mkfinal.py` — assembles `final.csv` from every group in `data/built*`,
  keeping the *smallest certified side* per `n` and re-certifying both transports.

### Scan sizes

| scan | lattices | rows | raw gadget instances | distinct `(w,h,g)` per lattice |
|---|---|---|---|---|
| C1 base (reused as input) | 7 (step 0.25) | 1000 | 1 421 731 | — |
| **fine, this POC** (step 0.125, `--cap 2000 --fmax 1.10`) | `(0.03125,0.03125)`, `(0.03125,0.09375)`, `(0.09375,0.03125)`, `(0.09375,0.09375)` | 1000 | **1 636 889** (369 608 + 418 400 + 423 621 + 425 260) | 7 666 / 9 518 / 9 326 / 10 459 |
| iteration, built rows as donors (step 0.125) | 2 of the above phases | 7 (`n=442,447,485,530,577,743,798`) | 11 785 | 2 871 / 1 872 |
| polish round (two-edge, `k=8`) | — | 6 103 seeds | — | 12 247 |

Merge results (Pareto front, `w'<=w, h'<=h, g'>=g, f'<=f`):

| library | Pareto items | scanned pockets with geometry | DP score gain |
|---|---|---|---|
| gadlib + C1 pareto (baseline, reproduces C1 exactly) | 1 322 | 629 | 2.8635e-05 (8 n) |
| + iteration scan (2 lattices on built rows) | 1 326 | 631 | 2.8635e-05 (8 n) — **no change** |
| + 4 fine lattices (**shipped**) | **1 699** | **908** | **8.4507e-05 (10 n)** |
| + polish `k=8` re-merge | 1 990 | 1 105 | 8.4507e-05 (10 n) — no change |

Only the fine lattices move the DP. The fine phase `0.03125` is the decisive one:
the two lattices with x-phase `0.03125` are the only ones that see the pocket
`(9,16,+5)` of `n=686` at `f = 0.696716582` (the y-phase alone is not enough: the
`0.09375`-x lattices cut the same shape at `f = 0.725718`, and C1's library only
had `(16,9,+5)` from `n=741` at `f = 0.711813327`).

Cost: 1 305–1 380 s per fine lattice at 8 processes (4 lattices in parallel on
32 cores), i.e. ≈ 23 min wall for the whole fine scan; scan of the 7 built rows
≈ 85 s each; polish ≈ 40 s; DP 3 s.

## 2. Certified wins — both transports, fresh main

Built by `poc/build_cert.py` (pockets copied from the **exact decimal strings** of
their donor row at 50 dps, translation only) and re-certified from `final.csv`'s
own bytes by `poc/verify_final.py` (deg → rad at 80 dps):

| n | K | j | composition (pocket, donor, exact `f`) | incumbent L | new L (literal = pandas) | dL (both modes) | gain |
|---|---|---|---|---|---|---|---|
| 442 | 21 | 1 | `(20,18,+1)` <- n=401 `f=0.533023120` | 21.533030303094208 | 21.533023120317424 | **-7.1827768e-6** | -3.4165e-7 |
| 447 | 21 | 6 | `(11,21,+6)` <- n=910 `f=0.656420966` | 21.656854249492479 | 21.656420965529470 | **-4.3328396e-4** | -2.0494e-5 |
| 485 | 22 | 1 | `(20,18,+1)` <- n=401 | 22.533031303094209 | 22.533023120317424 | **-8.1827768e-6** | -3.7156e-7 |
| **492** | **22** | **8** | **`(13,5,+3)`<-n=585 `f=0.705882354` + `(9,16,+5)`<-n=686 `f=0.696716582`** | 22.707106781186653 | **22.705882354152596** | **-1.2244270e-3** | **-5.5201e-5** |
| 530 | 23 | 1 | `(20,18,+1)` <- n=401 | 23.533032303094210 | 23.533023120317424 | **-9.1827768e-6** | -3.9887e-7 |
| 577 | 24 | 1 | `(20,18,+1)` <- n=401 | 24.533033303094211 | 24.533023120317424 | **-1.0182777e-5** | -4.2391e-7 |
| 585 | 24 | 9 | `(18,10,+6)`<-n=853 `f=0.705882420` + `(13,5,+3)`<-n=585 `f=0.705882354` | 24.705882675920645 | 24.705882419967224 | **-2.5595342e-7** | -1.0582e-8 |
| **740** | **27** | **11** | **`(18,10,+6)`<-n=853 `f=0.705882420` + `(9,16,+5)`<-n=686 `f=0.696716582`** | 27.705900674794851 | **27.705882419967224** | **-1.8254828e-5** | -6.7106e-7 |
| 743 | 27 | 14 | `(12,22,+10)`<-n=978 + `(14,5,+4)`<-n=539 | 27.753898760680030 | 27.753852984553401 | **-4.5776127e-5** | -1.6794e-6 |
| 798 | 28 | 14 | `(12,21,+9)`<-n=915 + `(15,7,+5)`<-n=742 | 28.740473088741449 | 28.740334237076354 | **-1.3885167e-4** | -4.9153e-6 |

**Total score gain 8.4507e-05** (7 of these `n` are C1 wins unchanged;
`n=492`, `n=740` and `n=585` are new — `585` is dust at 1.06e-8 in score).
Literal and pandas sides are identical for every group (only 1e-12 insets are
used, no dilation). Nominal union with the incumbent:
`22.153247131307513 - 8.4507364e-05 = 22.15316262`.

`n=585` was unbuildable with C1's fixed staggering (10 squares short); the
order/stagger search builds it exactly (`pocket 254 + grid 331 = 585`) and it
certifies at `dL = -2.56e-7`. It is only 1.06e-08 of score, but it is strict, so
it ships.

## 3. Where the money came from (mechanism)

The two new wins are **the same new gadget**: `(9,16,+5)` cut from `n=686`
(a 26x26 row) at `f = 0.696716582`, found only by the phase-`0.03125` fine
lattices. It is a *proper sub-pocket* (the whole row would need `K >= 26`), it is
narrow in the transposed orientation, and it is dense: 149 squares in
`9.697 x 16.697`.

* **n = 492, K = 22, D = 8** — the incumbent sits at `22 + 1/sqrt(2) = 22.7071`
  (a √2-zipper, the `D=8` family). Two pockets on one seam —
  `(13,5,+3)` (from n=585, already in C1's library) and `(9,16,+5)` (new) — give
  `sum w = 18 <= 22`, `sum h = 21 <= 22`, `sum(w+h+1) = 42 <= 45`, `sum g = 8`,
  and `max f = 0.705882354 < 0.707106781`. Side `22.705882354`, `dL = -1.22e-3`.
  That single cell is **65 % of the whole POC gain**. It also needed the builder
  fix: with C1's fixed staggering the grid only reaches 479 of 492 squares; the
  order/stagger search gets 275 grid squares (`pocket 217 + grid 275 = 492`).
* **n = 740, K = 27, D = 11** — the incumbent is `27.705900675`; the seam
  `(18,10,+6)` (n=853, `f = 0.705882420`) + `(9,16,+5)` gives
  `sum w = 27 <= 27`, `sum h = 16 <= 27`, `sum(w+h+1) = 46 <= 55`, `sum g = 11`,
  `max f = 0.705882420`, `dL = -1.83e-5`.
* **n = 442/485/530/577** — unchanged C1 mechanism: the `D=1` family is
  non-monotone in `K`, `(20,18,+1)` of n=401 at `f = 0.533023120` beats `c(1)`
  at `K = 21..24`.
* **n = 447** — unchanged C1 mechanism: a *narrow* pocket `(11,21,+6)` of n=910
  fits `K = 21` where the incumbent uses the √2-zipper.
* **n = 743/798** — unchanged C1 mechanism: two-pocket seams whose `max f` beats
  the incumbent's `c(D)` at that `(K,j)`.

So the fine lattices did **not** improve `f` of any shape that mattered; they
supplied one *new shape* (`(9,16,+5)`) at f = 0.6967 that a coarser lattice
(phase 0.0625/0.125 at step 0.25) never cut, and the DP turned it into two
2-pocket seams exactly at the two `D` values (8 and 11) where the incumbent's
family coefficient is 0.7071 / 0.7059.

## 4. Iteration with built rows as donors — negative (measured)

The 7 built rows (442/447/485/530/577/743/798) were appended to the donor set
(`mkdonors.py`, as `n=2000..2006`) and scanned at step 0.125 on two novel phases:
7 097 + 4 688 raw gadgets, 2 871 + 1 872 distinct `(w,h,g)` types, best examples
`(12,21,+9)` <- n=798 at `f = 0.740334` (the same value the DP already had) and
`(9,16,+5)` <- n=447 at `f = 0.824507` (much worse than the original n=686 cut).

After the Pareto merge this added **4 undominated items out of 1 326** and the DP
output did not change by a single bit. Reason (structural, not bad luck): a built
row is *pocket + integer grid*, and any axis-aligned window of it is either
(a) inside the pocket (already a gadget of the donor), (b) inside the grid (a
perfect `w x h` rectangle, `g = 0`, useless), or (c) straddling the seam, where
the window's tight bbox is dominated by (a) or has `f` ≥ the pocket's own `f`.
The seam is therefore *not* a new source of gadgets. Round 2 was not run.

Larger `g` per unit area is impossible locally: `g >= 1` needs
`(w+f)(h+f) > w*h`, so a window with an exactly integer bbox can never hold a
surplus square — hence no `f ≈ 0` gadget exists in any row, built or incumbent.

## 5. 3+ pockets per seam

Verified, not just assumed:

* `dp.py` is an unbounded knapsack over `(w,h,g)` with `sum w <= K`,
  `sum h <= K`, `sum(w+h+1) <= 2K+1`, so `m >= 3` is reachable;
* `build_cert.layouts_all` / `grid_cells` handle `m >= 3`
  (unit-tested: `m=3, K=24` -> `X=[0,9,17], Y=[11,4,0]`; `m=5, K=16` ->
  `X=[0,4,7,10,13], Y=[13,9,6,3,0]`).

But **no 3-pocket seam wins**: to pay, three pockets must beat the incumbent's
`c(D)` at `K = 20..31` while their `w+h+1` sum stays `<= 2K+1`; the small pockets
that satisfy the geometric budget have `f >= 0.52` (g=1) / `0.5355` (g=2) /
`0.5624` (g=3), and the incumbent's `c(D)` for `D <= 5` is already ≤ 0.6 at every
large `K`. Both new wins are `m=2`; the DP never chooses `m >= 3`.

## 6. L-border propagation — negative here (and why)

`border_prop.py` was applied to all groups built at the time (9) (target `n' = n + 2K + 1`):

| base n | n' | dL of the extended group |
|---|---|---|
| 442 | 485 | -8.1817768e-6 (1e-9 worse than the direct build — same geometry) |
| 485 | 530 | -9.1817768e-6 (same) |
| 530 | 577 | -1.0181777e-5 (same) |
| 447 | 490 | **+0.01939** (incumbent 22.6370 vs 22.6564) |
| 492 | 537 | **+0.01349** |
| 577 | 626 | **+0.01221** |
| 740 | 795 | **+0.00248** |
| 743 | 798 | **+0.01338** |
| 798 | 855 | **+0.00635** |

So the "free win" propagation does **not** multiply these wins. The `n' = n+2K+1`
free-win rule holds for the incumbents' own rows because their `c = L - K` is
*non-increasing* along that chain; a composition win sits at `c ≈ 0.53-0.71`,
while the incumbent's row at `n'` already has a lower `c` (e.g. `c = 0.637` at
490, `0.5208` at 626) — so `L+1` loses. Propagation only pays when the base row
is at least as good as the incumbent's coefficient at `n'`; nothing here is.
(The three "wins" at 485/530/577 are the direct builds again, 1e-9 worse because
of the mandatory border gap. `mkfinal.py` keeps the smaller side, i.e. the direct
builds.)

Precision note (cost real time): a `1e-12` border gap and a `1e-15` pitch make
the official metric answer *"Overlap undecidable at max precision"*; `GAP = 1e-9`
with pitch `1 + 1e-12` (the coordinator's proven values) certifies.

## 7. How to run

```bash
PY=.venv/bin/python
D=poc/pocket_lift_scan2
M=poc/pell_corrugation/data/border-scan/merged.csv

# 0. fresh-main 50-digit side table -> data/report.json  (267 s; score 22.153247131307513)
$PY poc/faultline_dp2/poc/mkreport.py $M $D/data/report.json

# 1. fine scan: step 0.125, cap 2000, fmax 1.10, four novel phases
#    (C1 already covered phases {0, 0.0625, 0.125} mod 0.25, so the *new*
#     information is at phases 0.03125 / 0.09375 -- those four lattices are shipped)
for o in "0.03125 0.03125" "0.03125 0.09375" "0.09375 0.03125" "0.09375 0.09375"; do
  set -- $o
  $PY $D/poc/scan_pockets.py $M $D/data/pockets_f0125_o$1_$2.json \
      --step 0.125 --fmax 1.10 --cap 2000 --procs 8 --offx $1 --offy $2 &
done; wait                      # ~23 min wall on 32 cores, 1.64M raw gadgets

# 2. merge (C1's shipped Pareto set is a valid input: same native geometry format)
$PY $D/poc/merge_pockets.py --gadlib poc/faultline_dp2/data/gadlib.json \
    --out $D/data/pockets_pareto2.json --seeds $D/data/seeds_all.json \
    poc/pocket_lift_scan/data/pockets_pareto.json $D/data/pockets_f0125_o*.json

# 3. DP -> 10 cells, score gain 8.4507e-05
$PY $D/poc/dp.py extra --lib poc/faultline_dp2/data/gadlib.json \
    --pockets $D/data/pockets_pareto2.json --report $D/data/report.json \
    --out $D/data/pred2.json

# 4. build + certify both transports -> data/built/n*.json  (2 min)
$PY $D/poc/build_cert.py --pred $D/data/pred2.json --pockets $D/data/pockets_pareto2.json \
    --report $D/data/report.json --outdir $D/data/built

# 5. (optional) L-border propagation, negatives are informative
$PY $D/poc/border_prop.py --report $D/data/report.json --built $D/data/built \
    --outdir $D/data/built_prop

# 6. assemble + re-certify final.csv, then re-certify it from its own bytes
$PY $D/poc/mkfinal.py
$PY $D/poc/verify_final.py

# 7. (optional, ran, no change) two-edge polish of the merged seed set, then re-merge+DP
$PY $D/poc/mkdonors.py                       # -> data/donors.csv (34 MB, needs built groups)
$PY $D/poc/polish.py $D/data/seeds_all.json $D/data/pockets_polish2.json \
    --csv $D/data/donors.csv --procs 26 --k 8
$PY $D/poc/merge_pockets.py --gadlib poc/faultline_dp2/data/gadlib.json \
    --out $D/data/pockets_pareto3.json poc/pocket_lift_scan/data/pockets_pareto.json \
    $D/data/pockets_iter_o*.json $D/data/pockets_polish2.json
$PY $D/poc/dp.py extra --lib poc/faultline_dp2/data/gadlib.json \
    --pockets $D/data/pockets_pareto3.json --report $D/data/report.json --out $D/data/pred3.json

# 8. (optional, ran, no change) iteration: built rows as donors
$PY $D/poc/mkdonors.py
for o in "0.03125 0.03125" "0.09375 0.09375"; do set -- $o
  $PY $D/poc/scan_pockets.py $D/data/donors.csv $D/data/pockets_iter_o$1_$2.json \
      --step 0.125 --fmax 1.10 --cap 2000 --procs 2 --rows 2000,2001,2002,2003,2004,2005,2006 \
      --offx $1 --offy $2
done
```

Minimal reproduction of the shipped `final.csv` (needs only the checked-in files):

```bash
$PY poc/pocket_lift_scan2/poc/build_cert.py --pred poc/pocket_lift_scan2/data/pred2.json \
    --pockets poc/pocket_lift_scan2/data/pockets_pareto2.json \
    --report poc/pocket_lift_scan2/data/report.json \
    --outdir poc/pocket_lift_scan2/data/built
$PY poc/pocket_lift_scan2/poc/mkfinal.py
$PY poc/pocket_lift_scan2/poc/verify_final.py
```

## 8. Files

| path | what |
|---|---|
| `poc/scan_pockets.py`, `polish.py`, `merge_pockets.py`, `dp.py` | copies of the C1 pipeline, unchanged |
| `poc/build_cert.py` | C1's builder **+ seam-layout search** (`layouts_all`, order × staggering), `--outdir` local |
| `poc/border_prop.py` | **new** — L-border propagation to `n+2K+1`, `GAP=1e-9`/pitch `1+1e-12`, both-transport check |
| `poc/mkdonors.py` | **new** — donor CSV = fresh main + built groups as `n=2000+i` |
| `poc/mkfinal.py` | **new** — assemble `final.csv` from `data/built*`, smallest side per `n`, both transports |
| `poc/verify_final.py` | C1's re-certifier, paths local |
| `data/report.json` | fresh-main 50-digit sides (score 22.153247131307513) |
| `data/pockets_pareto2.json` | **908** scanned pockets with exact geometry (the durable fine-scan output, 1.7 MB-scale) |
| `data/pockets_pareto3.json` | the same + the `k=8` polish round (**1 105** pockets, DP no change) |
| `data/pockets_polish2.json` | polish output, 12 247 types |
| `data/pockets_iter_*.json` | the built-row (iteration) scans, 11 785 gadgets |
| `data/seeds_all.json` / `seeds_iter.json` / `seeds2.json` | polish seeds |
| `data/pred2.json` | DP prediction that produced `final.csv` (10 cells) |
| `data/pred_base.json`, `pred_iter.json`, `pred3.json` | DP runs: C1 baseline / +iteration / +polish |
| `data/built/n{442,447,485,492,530,577,585,740,743,798}.json` | the 10 certified groups (radians, decimal strings) |
| `data/built_prop/` | the 3 border-propagated duplicates (kept for the audit in §6) |
| `data/*.log` | mkreport, 4 scan logs, iteration logs, build, polish, mkfinal, verify |
| `final.csv` | 5839 rows, groups 442/447/485/492/530/577/585/740/743/798, `id,x,y,deg`, `s`-prefixed decimals |

**Not checked in (regenerable, 278 MB):** the four fine raw scan outputs
`data/pockets_f0125_o*.json` (63-72 MB each) and `data/donors.csv` (34 MB). All
of them are one command each in §7, and everything derived from them is in
`data/pockets_pareto2.json` / `data/seeds_all.json`, so nothing is lost.

## 9. Honest read on the remaining money

* The DP is closed except at 10 cells and **all 10 are built and certified**;
  the smallest of them (`n=585`, 1.06e-08) needed the layout search too.
* Fine lattices buy shape *diversity*, not `f` refinements: 0 of the 23
  "improves an existing shape" items moved the DP. What pays is a *new*
  `(w,h,g)` at an `f` just below the incumbent's family coefficient at that
  `(K, D)` — `(9,16,+5)` at 0.6967 was worth 1.22e-3 because `D = 8` sits on the
  worst coefficient in the whole table (the √2-zipper 0.7071).
* Corollary worth a follow-up: **the highest-value targets are the `(K, D)` cells
  whose incumbent coefficient is a hard zipper** (`D=1` 0.5208, `D=8` 0.7071,
  `D=11` 0.7059, `D=2` …). Any proper sub-pocket with `max f` below that number
  and `sum g >= D` is a jackpot of order 1e-3. Scanning *only* the donors that
  can supply `D`-sized dense narrow pockets, at phase resolution 0.03125 or
  finer (step 0.0625), is the cheapest next step (`poc/pocket_reach` would say
  which `(K,D)` are reachable at all).
* Iteration (built rows as donors) is dead by construction (§4) — do not spend
  more cycles there. L-border propagation is dead for composition wins (§6)
  except when the base row's `c` beats the incumbent's `c` at `n'`.
