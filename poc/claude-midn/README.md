# claude-midn: n = 200..599

Scope: only puzzles `200 <= n <= 599`. Nothing outside that range is touched, and
`poc/pell_corrugation/data/border-scan/merged.csv` is read-only here.

## Result

**18 strict per-puzzle wins, all dust-scale. Net official gain `1.765e-08`.**

| file | literal | pandas | accepted |
|---|---|---|---|
| incumbent `merged.csv` (upstream `cd17a65`) | `22.173715450068453` | `22.173715450068453` | 1000 / 1000 both |
| incumbent + `data/changed-rows.csv` | `22.173715432420227` | `22.173715432420227` | 1000 / 1000 both |

All 18 changed n are inside 200..599, each strictly lower in both transports;
the other 982 n are identical to the incumbent, so there are zero regressions.
The team target of about 22.0999 is not approached: every structural experiment
below is negative. Certification output: [docs/VERIFICATION.md](docs/VERIFICATION.md).

The first certification (commit `ad02213`) was against the earlier incumbent
(`22.17371545025265`) and had 19 wins. The team union then landed in
`merged.csv`; against it `n = 591` is a tie and was dropped.

No Kaggle submission, no push, no PR.

## Idea

The loop log already says the portfolio is a local optimum, but two of the
tools behind that statement cannot see what they claim to test:

1. `poc/alln_fixed_lp` puts a `1e-9` margin on every wall and pair row. A
   saturated packing comes back `1.5e-8 .. 2.7e-8` *above* the incumbent, so
   any slack smaller than that is invisible.
2. `poc/lip_slp` charges `LIP * |dtheta|` on every row. In that model a rotation
   can only tighten a constraint, so it is a fixed-angle LP with extra
   variables; it cannot detect a first-order rotational escape such as a tilted
   block turning rigidly.

So step one was to build tools that do measure those two things, and step two
was to test the transfer moves that the log had only sampled (crop, peel,
re-fill, deletion) exhaustively over the whole range.

## Method

Everything is in `poc/`. Certification always goes through the unchanged
official per-puzzle evaluator in `poc/baseline`, in two transports: the literal
decimal strings, and the strings after a real `pd.read_csv(...).to_csv()` round
trip. A group is accepted only if both are strictly lower.

| script | what it does |
|---|---|
| `score_full.py` | whole-file official score, both transports. Same per-puzzle evaluator and sorted-n sum as `squares.core.score`, only with a larger pool (the bundled metric hard-codes 4 workers). |
| `rows.py` | extract / splice / per-puzzle two-transport gate. Never re-formats a string. |
| `tight_lp.py` | fixed-angle LP in displacement variables scaled to `1e-7`, pair margin `5e-13`, no wall margin, dual simplex. Touching contacts are frozen rather than forced apart. |
| `free_slp.py` | free-angle SLP in a kink-free corner formulation (below), exact fixed-angle LP repair, line search, final tight LP, official gate. |
| `crop_scan.cpp`, `dump_ext.py` | exhaustive axis-aligned crop of every larger packing. |
| `fault_peel.py` | interior fault-slab removal (generalises the border peel). |
| `refill.py` | exact upright re-fill (maximum independent set) around the tilted squares of a packing: own-box capacity, and transplant into other boxes. |
| `delete_slp.py` | delete squares from the `n+k` packing chosen by LP shadow price, then compress with the SLP. |
| `finalize.py` | per-puzzle gate, whole-file certification of union and incumbent, zero-regression check, writes `data/` and `docs/`. |
| `render.py` | dependency-free PNG of one packing (used to read the structures). |

### The free-angle SLP

For each nearby pair the best separating face normal `a` of one square (the
"owner") is chosen, and the four corners of the other square each get a row

```
a(theta_w) . (c_o + R(theta_o) w_k - c_w) >= 1/2
```

Walls are written on the four corners of every square. Each row is smooth in
`(x, y, theta)`, so an aligned face-to-face contact needs no `|dtheta|`
variable: a joint rotation of an aligned block is free, and a relative rotation
is charged exactly at first order. The step is tried at `alpha = 1, 1/2, 1/4,
1/8`; the second-order overlap is removed by an exact fixed-angle LP in the
centres (fallback: minimal uniform dilation), and the step is kept only if the
measured side after repair is lower.

HiGHS needed care at this scale: with displacements in natural units its
solutions violate rows by up to `7e-8`, which is larger than the slack being
looked for. Scaling the variables to the trust radius brings the residual down
to `~1e-15`.

## Baseline at the start of the session (re-computed, not trusted)

| file | literal | pandas | accepted |
|---|---|---|---|
| `border-scan/merged.csv` (incumbent) | `22.17371545025265` | `22.173715450252654` | 1000 / 1000 both |
| local union `Squares in Squares/submission.csv` | `22.173715449559737` | `22.17371544955974` | 1000 / 1000 both |
| PR #5 round 3 `public_refresh_nov12/data/submission.csv` | `22.173820983104367` | `22.173820983104367` | 1000 / 1000 both |

Per-puzzle comparison inside 200..599:

- local union vs incumbent: 10 strict wins, 0 worse.
- PR #5 round 3 vs incumbent: 11 strict wins, 4 worse. It is strictly below the
  local union at 10 of them (`420, 421, 459, 463, 464, 507, 508, 510, 550, 591`)
  and ties it at `546`, because round 3 shrank the same packings a second time.

So the starting point for 200..599 is the incumbent plus those 11 PR #5 groups.
`upstream/main` had nothing new to merge, and PRs #1-#5 are all closed/merged.

## Positive results (all dust)

| n | old L | new L | delta L |
|---:|---|---|---:|
| 420 | `20.960692680591730626777` | `20.960692680488174763961` | `-1.036e-10` |
| 421 | `20.974560502818374996088` | `20.974560502351638548647` | `-4.667e-10` |
| 459 | `21.901370072222965069434` | `21.901370030262454700043` | `-4.196e-08` |
| 463 | `21.968178945339402117991` | `21.968178945230692461067` | `-1.087e-10` |
| 464 | `21.983395789594238762050` | `21.983395789544272921416` | `-4.997e-11` |
| 502 | `22.877637090604511299999` | `22.877637089311898899999` | `-1.293e-09` |
| 504 | `22.911414563909498676200` | `22.911414563808444325479` | `-1.011e-10` |
| 505 | `22.930439920239041704899` | `22.930439919960285577931` | `-2.788e-10` |
| 507 | `22.960284705359482093341` | `22.960284457672172867332` | `-2.477e-07` |
| 508 | `22.978961490130026086423` | `22.978961459496640921883` | `-3.063e-08` |
| 510 | `22.997221435621395046091` | `22.997221435569194771458` | `-5.220e-11` |
| 535 | `23.634070747577996192439` | `23.634070746805632391496` | `-7.724e-10` |
| 546 | `23.848351044066986167335` | `23.848351044060371299200` | `-6.615e-12` |
| 550 | `23.911174766213658738933` | `23.911174766094907768823` | `-1.188e-10` |
| 551 | `23.925171448345933951272` | `23.925171445716330829415` | `-2.630e-09` |
| 595 | `24.861685491745497705738` | `24.861685483765431617721` | `-7.980e-09` |
| 598 | `24.903159935190023649999` | `24.903159868188931349456` | `-6.700e-08` |
| 599 | `24.923742501833048676855` | `24.923742501821128106319` | `-1.192e-11` |

Where it comes from:

- 11 groups started from the PR #5 round 3 rows (since merged upstream);
  most were then lowered further by the two tools below, so the table shows the
  final value.
- `tight_lp.py` finds 14 groups with translational slack between `7e-12`
  and `7.6e-9` that the `1e-9`-margin LP could not see.
- `free_slp.py` finds 11 groups that are not first-order rigid once rotations are
  modelled correctly. `n = 507`, `459` and `598` keep descending for hundreds of
  steps but converge: the predicted drop per step falls from `1.5e-7` to
  `3e-8` within 40 steps while the trust radius stays near `1e-3`.

These are honest strict wins but they do not move the team towards 22.0999.

## Negative results

All measured on the current rows, n = 200..599 unless stated.

| experiment | scope | outcome |
|---|---|---|
| Fixed-angle LP, 12 rounds (`alln_fixed_lp`) | 400 n | 0 wins. 394 `no-gain`, 6 `certify-fail` (unbounded LP flew apart). |
| Corrected LIP-SLP, 8 rounds (`lip_slp`) | 400 n | 0 wins, 398 s. |
| Free-angle SLP, kink-free (`free_slp.py`) | 400 n | 391 are first-order rigid: predicted drop `> -1e-9` at trust `2e-3`. The 9 that move give `< 3e-7` each. |
| Exhaustive crop (`crop_scan`) | every donor `n' > m` up to 1000, every anchor, 4 reflections | 0 wins. 235 targets tie (plateaus and border peels), the other 165 are worse by a median `0.013`. The earlier proxy in `poc/bign_donors` had only tried 10 donors. |
| Interior fault-slab peel (`fault_peel.py`) | 392 donors have an interior fault line; slabs up to 2.6 wide on both axes | 0 wins. 175 ties, median `+0.031`. |
| Own-box capacity re-fill (`refill.py capacity`) | 400 n | No box holds a square more than the portfolio already knows. Every surplus is an existing plateau (grids, the `n, n+1` pairs at `0.5355`, and `367/368`). |
| Transplant re-fill (`refill.py transplant`) | tilted set of `n` re-filled in the box of every `m` in `(n, n+70]` at `L(m) - 2e-4`, 4 corner anchorings | Stopped early for CPU: 7,781 donor/target pairs, donors 200..407. 0 hits; the best is always at least one square short (`data/scans/refill_transplant_partial.csv`). |
| Shadow-priced single deletion (`delete_slp.py`) | 12 targets with the smallest `L(n+1) - L(n)` (0.0027 .. 0.0073), every square priced; plus a whole-range screen (`delete_screen.py`) | 0 wins on the 12. Eight donors have no priced deletion at all (grids and rigid diamonds). `n = 234` compresses to `1.2e-4` above the incumbent after 400 steps, `540` to `1.4e-3`, `451` stays `7e-3` above. Whole-range screen: see `data/scans/delete_screen.jsonl` (status below). |
| Local files | 23 full CSVs in `Squares in Squares/` and `final_atlas.csv`, float prefilter at `1e-12` | 0 candidates below the working rows. |

Two of these are stronger statements than what was logged before:

- **Deletion splits into two cases.** For grid-like and rigid-diamond donors, no
  single deletion gives a first-order drop of the box side at all: the side is
  held by many parallel chains and one hole breaks only a few. For the softer
  arm-type donors deletion plus SLP does compress a long way (from the 202
  packing it reaches `L(201)` to `6e-12`, from 235 it reaches `L(234) + 1.2e-4`),
  but it lands on the incumbent's own basin or just above it, not below.
- **The portfolio is closed under crop, peel and own-box re-fill.** Together with the
  border bound in `poc/constant_clusters`, no packing in the file can be turned
  into a better packing of another `n` in 200..599 by cutting, peeling,
  wrapping, deleting or re-filling uprights.

### Whole-range deletion screen

`delete_screen.py`, all 400 targets, k = 1, every square of the `n+1` packing
priced, the 3 best spatially distinct deletions compressed for 40 SLP steps.
Report: `data/scans/delete_screen.jsonl`.

- **0 wins.**
- 245 donors are rigid: no single deletion gives a first-order drop.
- 155 compress but stop above the incumbent, median `+0.017`. Closest:
  `n = 201` (`+6e-12`, lands exactly on the incumbent's constant), `234`
  (`+2.6e-4`; `+1.2e-4` after 400 steps), `410` (`+5.1e-4`), `540` (`+1.4e-3`),
  `509` (`+2.3e-3`), `421` (`+2.7e-3`).

Only `k = 1` and 40 steps were screened. `k >= 2` and deeper runs on the
closest targets are untested, apart from the 400-step run on `234`.

## What the gap looks like

Gap to the all-team frontier (`poc/frontier_nov12/docs/per_n_gap.csv`) inside
200..599 is `0.0422` of the `0.1228` total; 164 n are behind by more than
`1e-6` in `L`. The largest are `367, 328, 530, 577, 492, 365, 333, 379, 587`.

Reading the pictures:

- `n = K^2 + 1` (401, 442, 485, 530, 577) are two rigid 5x5 diamonds plus a
  corrugated block at `L - K = 5/sqrt2 - 3`. For `K >= 25` the file uses a 17x17
  diamond with four corner staircases at `17/sqrt2 - 11.5 = 0.5208`.
  In that family the count is exact arithmetic: with staircase sizes
  `(a_TL, a_TR, a_BR, a_BL)`, opposite pairs must sum to `2L - 4 - 17 sqrt2`
  and adjacent pairs to at least `L - 4`. At `K = 24`, `L = 24.5208` the best
  split is `(10, 11, 11, 10)`, which gives 577 squares on paper but leaves the
  left tip `0.26` outside the wall. Removing the tip square gives 576, and the
  freed notch (area 1.93, arrow-shaped) cannot hold a unit square. That is why
  the 17-diamond starts at `K = 25`.
- `n` just below `(K+1)^2` (e.g. 379, 587) are an upright block plus two arms of
  tilted rows, the Erdős-Graham layout. Arm width, squares per row and tilt are
  tied by `m cos(theta) + sin(theta) = W`.

The frontier values at `K^2 + 1` (`0.5261, 0.5229, 0.5173, 0.5134, ...`) are not
of the form `m/sqrt2 mod 1/2` for any `m`, nor of `m (cos + sin)/2` for the
small Pythagorean angles, so they are not rigid lattice constructions.

## Unresolved

- A new root for `n = 401, 442, 530` (and by one border `485, 577`). This is
  the cheapest block of gap in the range (`0.0029` in score for five n) and
  needs a non-rigid topology.
- The arm family near `(K+1)^2`: no generator here; only the existing rows were
  tested.
- Transplants were only tried for a single rigid tilted set at the four corner
  anchorings. Two-block placements and fractional phases are untested in this
  range.

## Reproduce

```sh
PY=python   # numpy, scipy, pandas, mpmath
INC=poc/pell_corrugation/data/border-scan/merged.csv

# baseline
$PY poc/claude-midn/poc/score_full.py $INC /tmp/merged 8

# PR #5 round 3 rows, gated against the incumbent
$PY poc/claude-midn/poc/rows.py extract poc/public_refresh_nov12/data/submission.csv \
    420,421,459,463,464,507,508,510,546,550,591 /tmp/pr5rows.csv
$PY poc/claude-midn/poc/rows.py gate $INC /tmp/pr5rows.csv /tmp/stage1.csv
$PY poc/claude-midn/poc/rows.py splice $INC /tmp/stage1.csv /tmp/base1.csv

# margin-tight LP, then free-angle SLP (twice: movers need more iterations)
$PY poc/claude-midn/poc/tight_lp.py /tmp/base1.csv 200 599 8 /tmp/tight.csv /tmp/tight.json
$PY poc/claude-midn/poc/rows.py splice /tmp/base1.csv /tmp/tight.csv /tmp/base2.csv
$PY poc/claude-midn/poc/free_slp.py /tmp/base2.csv 200-599 8 /tmp/fs1.csv /tmp/fs1.json 60
$PY poc/claude-midn/poc/rows.py splice /tmp/base2.csv /tmp/fs1.csv /tmp/base3.csv
$PY poc/claude-midn/poc/free_slp.py /tmp/base3.csv 200-599 8 /tmp/fs2.csv /tmp/fs2.json 250
$PY poc/claude-midn/poc/rows.py splice /tmp/base3.csv /tmp/fs2.csv /tmp/work.csv

# gate + whole-file certification; writes data/ and docs/
$PY poc/claude-midn/poc/finalize.py $INC /tmp/work.csv /tmp/claude-midn-union.csv 8
```

To rebuild the certified 1000-row file from what is committed:

```sh
$PY poc/claude-midn/poc/rows.py splice $INC poc/claude-midn/data/changed-rows.csv /tmp/claude-midn-union.csv
```

The scans:

```sh
clang++ -O3 -std=c++17 -o /tmp/crop_scan poc/claude-midn/poc/crop_scan.cpp
$PY poc/claude-midn/poc/dump_ext.py $INC /tmp/ext.bin
/tmp/crop_scan /tmp/ext.bin 200 599 8 > /tmp/crop.csv
$PY poc/claude-midn/poc/fault_peel.py /tmp/ext.bin 200 599 > /tmp/peel.csv
$PY poc/claude-midn/poc/refill.py capacity /tmp/work.csv 200 599 8 $FILLFAST_DIR
$PY poc/claude-midn/poc/refill.py transplant /tmp/work.csv 200 599 8 $FILLFAST_DIR 70
$PY poc/claude-midn/poc/delete_slp.py /tmp/work.csv 495,510,308,295 8 /tmp/del.csv /tmp/del.json
```

`refill.py` imports `fillfast.py` from a local checkout of the public
`evand/square-packing` packer (`search/packer/`); that file is not vendored
here.

## Files

- `data/changed-rows.csv` - the winning groups only.
- `data/summary.json` - scores, hashes, per-n old/new `L` in both transports.
- `data/scans/` - small reports of the scans above.
- `docs/literal.json`, `docs/pandas.json` - per-n official report of the union.
- `docs/incumbent-literal.json`, `docs/incumbent-pandas.json` - the same for the
  incumbent, from the same run.
- `docs/VERIFICATION.md` - the certification output.
