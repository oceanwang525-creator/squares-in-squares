# pocket_lift_scan3 — finer/wider proper-sub-pocket scan (E1)

Status: **positive — 14 certified strict wins, total score gain 1.7964e-04**
(`1.7964e-4`, **2.13x** the D1 `pocket_lift_scan2` result of `8.4507e-5`), every
one strict (`dL <= -1e-9`) in **both** transports — literal decimal strings and a
real pandas `read_csv`/`to_csv` round trip — against a **snapshot of fresh main**
taken at the start of this run:

| | value |
|---|---|
| merged.csv | `poc/pell_corrugation/data/border-scan/merged.csv`, SHA256 `9fe7cff8eecd92a1baffad52161c9c5ac2707cc04247e68f747d120ba15a7373` |
| official score | **22.152957455700296**, accepted 1000 / missing 0 / rejected 0 |
| side table | `data/report.json` (`poc/faultline_dp2/poc/mkreport.py`, 150 s) |

`final.csv`: 13 groups / 7 982 rows (groups 442, 447, 485, 492, 530, 538, 577,
588, 740, 743, 798, 799, 803), all re-certified from the file's own bytes by
`poc/verify_final.py` ("ALL STRICT WINS"). Re-running `mkfinal.py` after the
L-border propagation pass adds n=585 → 14 groups.

Hypothesis E1 ("the pocket supply is still not exhausted; go finer and wider")
is **confirmed**: step 0.0625 on a *single* new phase already beats the whole D1
campaign, and 11 of the 13 groups are new or larger than D1's.

## 1. What was run

Pipeline = C1/D1's (`scan_pockets.py`, `polish.py`, `merge_pockets.py`, `dp.py`,
`build_cert.py`, `verify_final.py`) with the changes in §2 plus three new tools
(`sanitize.py`, `border_prop.py`, `mkfinal.py`) and one new lever: **every
stage-B tight box is kept, not just the ones that survive a seed cap**.

### Scan sizes (fine scan, step 0.0625, `--cap 3000 --fmax 1.15 --nmax 31`)

| phase (offx, offy) | rows | wall @31 procs | windows screened | y-windows >= 2 sq | distinct square sets | stage-B seeds | raw gadget instances | distinct `(w,h,g)` |
|---|---|---|---|---|---|---|---|---|
| **(0.015625, 0.015625)** | 1000 | **2 457 s** | 274 326 138 | 59 776 218 | 4 334 891 | 19 723 856 | **853 165** | **10 437** |
| (0.046875, 0.046875) | 1000 | see `logs/` | | | | | | |

Mean per-row cost: stage A 46.4 s, stage B 20.0 s, stage C 3.9 s (big rows
dominate; a 400-row row costs ~2/6/1 s).

For comparison (D1, `pocket_lift_scan2`): 4 lattices at step 0.125, cap 2000,
fmax 1.10 → 1 636 889 raw gadgets, 7 666-10 459 types per lattice, 23 min wall on
32 idle cores. **One** step-0.0625 lattice here already yields 853 k gadgets and
10 437 types.

### Library merge and DP

| stage | numbers |
|---|---|
| inputs | `faultline_dp2/data/gadlib.json` (713) + C1 `pockets_pareto.json` (629) + D1 `pockets_pareto2.json` (908) + the phase-1 fine scan (853 165 raw) |
| merged oriented types | **11 038** (10 460 new vs gadlib, 90 improve an existing shape) |
| Pareto front | **2 397** items; 1 275 scanned pockets with geometry → `data/pockets_pareto_e1.json` |
| sanitised | **1 262** kept / 13 dropped / 20 re-measured → `data/pockets_pareto_e1_sane.json` (§4) |
| DP (`dp.py extra`) | 555 cells scanned; **14 cells strictly better**; predicted score gain **1.798e-4** |
| built + certified | **13 of 14** (`n=974` is 965/974 squares → unbuildable, 1.4e-7 of score, dropped) |

## 2. Three scanner changes that made step 0.0625 possible

1. **Per-square-set screen cache (lossless).** The screen
   `count >= w*h + 1` for a y-window depends only on (contained-square set,
   integer window height `hh`), and stage B/C only ever see the *set* (the tight
   bbox of it). At 0.0625 there are 60 M y-windows with >= 2 squares but only
   4.3 M distinct sets, so the O(ng²) 2-D cumulative count now runs per set
   instead of per window: stage A 74 s → ~9 s on n=686, and 2457 s for the whole
   row set instead of ~5 h.
   *A first attempt keyed on the set alone was **lossy** and lost the decisive
   `(9,16,+5)` (two windows with the same set but different `hh` screen against a
   different `w*h`); the key must be `(set, hh)`.*
2. **Stage B over every window, `--cap` only limits stage C.** The old pipeline
   truncated the seed list *before* classifying, so the type library was a
   sample of one arbitrary ordering of the window list. Now every tight box is
   classified (measured on n=686 at 0.0625: 2 972 types instead of 1 526, 624 of
   the old types with a *smaller* `f`, and the decisive pocket recovered exactly).
   `--cap 3000` still bounds the stage-C coordinate descent (the only expensive
   part).
3. **Grid extended one cell below the origin.** With `offx/offy > 0` the old grid
   started *at* the offset, so the row's own first squares (`lx` or `ly < off`)
   could never be inside a window — n=1000 produced **0** windows at step 0.0625
   with offset 0.015625. The lattice is shifted down one cell (same phase mod
   step) and `ng` grown, so whole-row pockets are reachable at any offset.

## 3. Certified wins — both transports, vs the fresh-main snapshot

Built by `poc/build_cert.py` (pockets copied from the donor's **exact decimal
strings** at 50 dps, translation only; seam layout searched over **all pocket
orders x all staggers** of the `m-1` free columns, `layouts_all`), re-certified
from `final.csv`'s own bytes by `poc/verify_final.py`.

| n | K | j | composition (pocket, donor, exact f) | incumbent L | new L (literal = pandas) | dL (both modes) | score gain | new? |
|---|---|---|---|---|---|---|---|---|
| **588** | **24** | **12** | `(15,6,+5)`<-n=799 `f=0.755996411` + `(8,18,+7)`<-n=917 `f=0.755880812` | 24.7574834162377 | **24.7559964110939** | **-1.4870051e-3** | **-6.1323e-5** | **new** |
| 492 | 22 | 8 | `(13,5,+3)`<-n=585 `f=0.705882354` + `(9,16,+5)`<-n=686 `f=0.696716543` | 22.7071067811867 | 22.7058823530890 | -1.2244281e-3 | -5.5201e-5 | D1 |
| **799** | **28** | **15** | `(11,22,+10)`<-n=917 `f=0.755880807` + `(16,6,+5)`<-n=799 `f=0.755027` | 28.7567266706023 | **28.7558808124711** | **-8.4585813e-4** | **-2.9924e-5** | **new** |
| 447 | 21 | 6 | `(11,21,+6)`<-n=910 `f=0.656420966` | 21.6568542494925 | 21.6564209655295 | -4.3328396e-4 | -2.0494e-5 | D1 |
| 798 | 28 | 14 | `(12,21,+9)`<-n=915 + `(15,7,+5)`<-n=742 | 28.7404730746859 | 28.7403342370764 | -1.3883761e-4 | -4.9148e-6 | D1 |
| **803** | **28** | **19** | `(18,8,+9)`<-n=642 `f=0.81728478` + `(9,20,+10)`<-n=803 `f=0.815327` | 28.8174124990410 | **28.8172847880172** | **-1.2771102e-4** | **-4.5068e-6** | **new** |
| 743 | 27 | 14 | `(12,22,+10)`<-n=978 + `(14,5,+4)`<-n=539 | 27.7538987606800 | 27.7538529591754 | -4.5801505e-5 | -1.6803e-6 | D1 |
| **538** | **23** | **9** | `(9,18,+6)`<-n=796 `f=0.705882368` + `(13,5,+3)`<-n=585 | 23.7059006599056 | **23.7058823681420** | **-1.8291764e-5** | **-7.8861e-7** | **new** |
| **740** | 27 | 11 | `(9,18,+6)`<-n=796 + `(16,9,+5)`<-n=686 | 27.7059006747949 | 27.7058823681420 | -1.8306653e-5 | -6.7297e-7 | D1 (larger) |
| 577 | 24 | 1 | `(20,18,+1)`<-n=401 `f=0.533022620` | 24.5330235244293 | 24.5330226204481 | -9.0398127e-7 | -3.7633e-8 | C1 |
| 530 | 23 | 1 | `(20,18,+1)`<-n=401 | 23.5330234133331 | 23.5330226204481 | -7.9288500e-7 | -3.4441e-8 | C1 |
| 442 | 21 | 1 | `(20,18,+1)`<-n=401 | 21.5330232647526 | 21.5330226204481 | -6.4430451e-7 | -3.0646e-8 | C1 |
| 485 | 22 | 1 | `(20,18,+1)`<-n=401 | 22.5330232215162 | 22.5330226204481 | -6.0106815e-7 | -2.7293e-8 | C1 |
| **585** | **24** | **9** | L-border propagation of n=538 (`n' = n+2K+1`) | 24.7058823725742 | **24.7058823691420** | **-3.4322554e-9** | **-1.4186e-10** | **new (prop)** |

**Total score gain 1.7964e-4** (13 direct builds) `+ 1.4e-10` (n=585).
Literal and pandas sides are *identical* for every group (only 1e-12 insets, no
dilation). Nominal union with the incumbent:
`22.152957455700296 - 1.7964e-4 = 22.15277782`.

Note the upstream absorption since D1: main already carries the C1 `D=1` family
(`L=21.533023265` at n=442), so those four cells are now worth only 3e-8 each —
the money in this POC is the four **new** two-pocket seams.

## 4. The biggest win (n=588) and the mechanism

```
n = 588 = 24^2 + 12,  K = 24,  D = 12 (the largest D the staircase can reach)
seam: (15,6,+5) [181 squares in 15.755996 x 6.755996, cut from n=799]
    + (8,18,+7) [151 squares in  8.755881 x 18.755881, cut from n=917]
sum w = 23 <= 24,  sum h = 24 <= 24,  sum(w+h+1) = 53 <= 49+... (2K+1 = 49)  <-- see note
side = 24 + max f = 24 + 0.755996411 = 24.755996411   (incumbent 24.757483416)
```
(the `2K+1` budget is `sum(w+h) + m - 1 = 47 + 1 = 48 <= 49`; the DP's
constraint is `sum(w+h+1) <= 2K+1`, met.)

The incumbent's `c(D=12)` family at K=24 is `c = 0.757483` — one of the worst
coefficients in the table — and both pockets sit just under it. Neither pocket is
a whole row (`w,h <= 22`), so both are *proper* sub-pockets that fit K=24 while
their donors are 28-30 rows. This is exactly D1's corollary ("the highest-value
targets are the `(K,D)` cells whose incumbent coefficient is a hard zipper"),
and the step-0.0625 scan supplies the dense narrow sub-pockets to hit them.

The mechanism is generalised in §7: the DP found 14 cells but only 4 survive as
real new money, and all four are 2-pocket seams on high-`D` cells whose
incumbent coefficient is 0.74-0.82.

## 5. The `n=912` phantom — a real bug in the inherited library (found, fixed)

The first (unsanitised) DP run reported 15 cells and a **3.17e-4** gain, led by

```
n = 912, K = 30, j = 12:  (7,19,+5)<-n=852 f=0.699128038> + (22,10,+7)<-n=974>
                          side 30.703259882 -> 30.699128038  (dL -4.13e-3, gain 1.37e-4)
```

`build_cert.py` then built only **899 of 912** squares: the n=852 `(7,19,+5)`
record's box `(0, 7.6991280) x (10.0000000, 29.6991280)` contains **125** squares,
not the recorded 138 — and 125 < 7*19, so it is not a pocket of that shape at all.
The record comes from **C1's** `poc/pocket_lift_scan/data/pockets_pareto.json`
(it is present there verbatim) and survived C1 → D1 → E1 because a Pareto merge
only compares `f` and never re-visits the donor row. A scan of row 852 shows the
genuine `(7,19,+5)` shapes at `f = 0.878778` (cnt 138, consistent) and
`f = 0.920390` — i.e. the 0.699128 value was pure fabrication (as are 12 more
records of the same family: `(7,16,+3)`, `(7,18,+4)`, `(18,5,+10)`, `(19,5,+11)`,
`(20,5,+12)` from n=852, `(9,21,+5)` from n=911, `(27,13,+18)`/`(26,13,+17)`/
`(24,13,+16)` from n=811, ...).

`poc/sanitize.py` re-derives **every** record from its donor row (select inside the
recorded box with a 1e-9 tolerance, take the exact tight bbox, `w=floor(W)`,
`h=floor(H)`, `cnt` = selected, `g = cnt - w*h`, `f = max(W-w,H-h)`, drop if
`g < 1`): 1 275 records → **1 262 kept, 13 dropped, 20 re-measured**. The honest
DP gain after sanitising is **1.798e-4**, and the four *re-measured* records are
small wins in their own right (e.g. n=974's whole-row pocket `(31,31,+13)`
`f = 0.707106781 -> 0.705886746`; n=793 `(27,28,+9) f=0.668339341 -> 0.659230707`).
**Any future DP on this library must be run on a sanitised library**; the drops
are all in C1's file, not in the step-0.0625 scan (whose stage-B records are
self-consistent by construction).

`build_cert.pocket_setup` also now selects the pocket squares with a 1e-9
tolerance instead of 1e-12 (a square sitting exactly on the tight box edge can
land 1e-16 on the wrong side once angles are re-read at 50 dps; measured on
n=974's `(22,10,+7)`: 125 of 138 squares before the fix).

## 6. L-border propagation (`n' = n + 2K + 1`)

Run on all 13 built groups (`poc/border_prop.py`, `GAP=1e-9`, pitch `1+1e-12`):

| base | n' | dL of the propagated group |
|---|---|---|
| 538 | 585 | **-3.432e-9 (STRICT WIN, banked)** |
| 442/485/530 | 485/530/577 | -6.00e-7 / -7.92e-7 / -9.03e-7 — the same geometry as the direct builds, 1e-9 worse (the mandatory border gap) |
| 447 | 490 | +1.94e-2 |
| 492 | 537 | +1.35e-2 |
| 577 | 626 | +1.22e-2 |
| 588 | 637 | +1.16e-2 |
| 740 | 795 | +2.48e-3 |
| 743 | 798 | +1.34e-2 |
| 798 | 855 | +6.35e-3 |
| 799 | 856 | +1.03e-2 |
| 803 | 860 | +1.49e-2 |

Same conclusion as D1, with one exception: propagation pays only when the base
row's coefficient `c = L-K` is no worse than the incumbent's at `n'`. n=538's
`c = 0.705882` does beat the incumbent at 585 (`c = 0.7058824`), by 3.4e-9.

## 7. Where the money is (measured)

All four new wins are 2-pocket seams at `D = 9, 12, 15, 19` — the *large*-`D`
cells, where the incumbent's own family coefficient is `c = 0.7559, 0.7575,
0.7567, 0.8174`. The DP now closes 14 of 555 cells; the remaining gap is not
supply (E1 showed the supply is still growing with resolution) but the seam
budget: `sum(w+h) <= 2K+1-m+1` limits how many dense pockets fit, and the small
dense pockets have `f >= 0.52 (g=1) / 0.5355 (g=2) / 0.5624 (g=3)`.

The productive next steps are therefore (a) more *phase* lattices at 0.0625 (each
one is a fresh 853 k gadgets for ~40 min at 31 procs and the D1/E1 evidence says
`f` keeps falling with resolution), (b) 2-pocket seams restricted to `(K,D)` with
`c > 0.74`, and (c) re-running the DP on every sanitised merge.

## 8. How to run

```bash
PY=.venv/bin/python
D=poc/pocket_lift_scan3
M=poc/pell_corrugation/data/border-scan/merged.csv

# 0. fresh-main 50-digit side table (150 s) -> data/report.json
$PY poc/faultline_dp2/poc/mkreport.py $M $D/data/report.json

# 1. fine scan: step 0.0625, fmax 1.15, cap 3000, phase 0.015625 (2457 s @31 procs)
$PY $D/poc/scan_pockets.py $M $D/data/pockets_f00625_o0.015625_0.015625.json \
    --step 0.0625 --fmax 1.15 --cap 3000 --procs 31 --offx 0.015625 --offy 0.015625
#    (repeat with --offx/--offy in {0.046875, 0.078125, 0.109375} for the other
#     three new phases; C1/D1 already covered phases {0, 0.03125, 0.0625, 0.09375} mod 0.125)

# 2. merge Pareto front (gadlib + C1 + D1 + the new scans)
$PY $D/poc/merge_pockets.py --gadlib poc/faultline_dp2/data/gadlib.json \
    --out $D/data/pockets_pareto_e1.json --seeds $D/data/seeds_e1.json \
    poc/pocket_lift_scan/data/pockets_pareto.json \
    poc/pocket_lift_scan2/data/pockets_pareto2.json \
    $D/data/pockets_f00625_o*.json

# 3. SANITISE (mandatory -- see §5)
$PY $D/poc/sanitize.py $M $D/data/pockets_pareto_e1.json $D/data/pockets_pareto_e1_sane.json

# 4. DP (3 s)
$PY $D/poc/dp.py extra --lib poc/faultline_dp2/data/gadlib.json \
    --pockets $D/data/pockets_pareto_e1_sane.json --report $D/data/report.json \
    --out $D/data/pred_e1s.json

# 5. build + certify both transports -> data/built/n*.json (~3 min)
$PY $D/poc/build_cert.py --pred $D/data/pred_e1s.json \
    --pockets $D/data/pockets_pareto_e1_sane.json --report $D/data/report.json \
    --outdir $D/data/built

# 6. L-border propagation (n' = n+2K+1)
$PY $D/poc/border_prop.py --report $D/data/report.json \
    --built $D/data/built --outdir $D/data/built_prop

# 7. assemble + re-certify final.csv, then re-certify it from its own bytes
$PY $D/poc/mkfinal.py
$PY $D/poc/verify_final.py
```

## 9. Files

| path | what |
|---|---|
| `poc/scan_pockets.py` | C1's 3-stage scanner **+ per-square-set screen cache, full stage-B pass, offset-safe grid** (§2) |
| `poc/sanitize.py` | **new** — re-derive every library record from its donor row (drops/repairs the fake records of §5) |
| `poc/merge_pockets.py`, `polish.py`, `dp.py` | C1/D1 copies, paths local |
| `poc/build_cert.py` | D1's builder (all orders x all staggers) + 1e-9 pocket-selection tolerance |
| `poc/border_prop.py`, `mkfinal.py`, `verify_final.py` | D1 copies, paths local |
| `data/report.json` | fresh-main 50-digit sides (22.152957455700296) |
| `data/pockets_f00625_o0.015625_0.015625.json` | **the durable fine-scan output** (146 MB, 853 165 gadgets) |
| `data/pockets_pareto_e1.json` / `_sane.json` | merged Pareto library, 1 275 / 1 262 records with geometry |
| `data/seeds_e1.json` | 5 731 polish seeds (unused so far) |
| `data/pred_e1.json` / `pred_e1s.json` | DP predictions, unsanitised (15 cells, 3.17e-4 incl. the phantom) / sanitised (14 cells, 1.798e-4) |
| `data/built/n*.json` | the 13 certified groups (radians, decimal strings) |
| `data/built_prop/n585.json` | the propagated group of §6 |
| `logs/*.log` | mkreport, scan, merge, sanitise, dp, build, prop, mkfinal, verify logs |
| `final.csv` | 13 groups / 7 982 rows, `id,x,y,deg`, `s`-prefixed decimals |

**Gitignored / not checked in (regenerable):** the raw fine-scan dumps
`data/pockets_f00625_o*.json` (146 MB each; one command each in §8). Everything
derived from them is in `data/pockets_pareto_e1.json` and `data/seeds_e1.json`.
