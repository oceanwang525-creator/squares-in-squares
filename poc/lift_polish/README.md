# lift_polish — polish the C1 lifts, then re-lift with the polished rows as donors (D2)

Status: **positive — 6 certified strict wins, total score gain 9.6567e-05**
(0.0000966), i.e. **3.4x the whole C1 gain (2.8624e-05)**, on top of C1's own 7
wins.
Every claim is strict (`dL <= -1e-9`) in **both** transports (literal decimal
*and* a real `pandas.read_csv`/`to_csv` round trip) **against both**
fresh main (`poc/pell_corrugation/data/border-scan/merged.csv`, score
22.153247131307513) **and** the C1 value for the same `n`.

The hypothesis was "the C1 lifts are rigid compositions that no optimiser has
ever touched, so they still have slack". That is **confirmed and then some**:
the polished lifts are *not* a dead end but a **donor cascade** — each polish
round lowers the seam-gadget's excess `f`, and the DP of `poc/pocket_lift_scan`
immediately propagates that to every larger `K` where the gadget fits. Four
polish rounds interleaved with four DP+build rounds kept paying: the third
polish round still moved `n = 485/530/577` by 2.2e-4..3.1e-4, and the fourth
still moved `n = 442` by 3.0e-6.

## 1. What the three proven optimisers did to the raw C1 lifts

Per `n`, all three were run on the C1 group reconstructed from
`poc/pocket_lift_scan/data/built/n{...}.json` (`poc/mk_c1.py` →
`data/c1_merged.csv`, official sides in `data/c1_report.json`):

| optimiser | run | result on the 7 raw C1 lifts |
|---|---|---|
| `poc/alln_fixed_lp` (fixed-angle centre LP, 12 rounds, no trust region) | `data/lp_fixed/` | **0 wins**; the LP returns the C1 side + 9e-9..1e-8 (the four wall rows each add the 1e-9 margin). n=743's `lp_side` is 1.4e-7 below its C1 side but does not certify (float overlap) |
| `poc/lip_slp` (Lipschitz free-angle SLP, 8 rounds) | `data/lipslip/` | **0 wins** in 43 s |
| `poc/swarm_lpslack` (soft-angle SLP + unbounded fixed-angle LP + uniform shrink) | `data/lpslack/results.json` | **6 wins** — this is the only optimiser that moves the C1 lifts |

Round 1 (`lpslack` on the raw C1 rows, 43-198 s per group):

| n | C1 side | after round 1 | dL |
|---:|---:|---:|---:|
| 442 | 21.533023120317 | 21.533023022362 | -9.7955e-8 |
| 447 | 21.656420965529 | 21.656420962933 | -2.5964e-9 |
| 485 | 22.533023120317 | 22.532830036125 | **-1.9308e-4** |
| 530 | 23.533023120317 | 23.532839850927 | **-1.8327e-4** |
| 577 | 24.533023120317 | 24.532840282822 | **-1.8284e-4** |
| 743 | 27.753852984553 | unchanged | 0 |
| 798 | 28.740334237076 | 28.740330645824 | -3.5913e-6 |

So D2's premise holds for the `D = 1` lifts specifically: the C1 pocket lifted
to `K = 22, 23, 24` was 1.8e-4 of `f` away from being polished, while the
pocket it was cut from (n=401) is itself a long-polished row.

## 2. The cascade: a polished lift is a better donor (this is the real finding)

Round 1's improved `n = 485` row is a `22x22` grid plus one seam gadget. Re-cutting
its sub-rectangles (`poc/pocket_lift_scan/poc/scan_pockets.py`, 3 offset lattices)
gives `(18,21,+1)` at `f = 0.532830036` instead of C1's `(20,18,+1)` at
`f = 0.533023120`, and the C1 staircase DP then predicts that gadget for
*every* `K` it fits:

| round | what happened | n=442 | n=485 | n=530 | n=577 |
|---|---|---:|---:|---:|---:|
| 0 | C1 lift | 21.533023120 | 22.533023120 | 23.533023120 | 24.533023120 |
| 1 | `lpslack` polish | 21.533023022 | 22.532830036 | 23.532839851 | 24.532840283 |
| 2 | DP re-lift of 1 (build) | **21.532830036** | 22.532830036 | **23.532830036** | **24.532830036** |
| 3 | `lpslack` polish | **21.532779290** | 22.532784030 | **23.532731639** | **24.532784769** |
| 4 | DP re-lift of 3 (build) | 21.532779290 | **22.532731526** | **23.532731526** | **24.532731526** |
| 5 | `lpslack` polish | **21.532732133** | **22.532466364** | **23.532417292** | **24.532516434** |
| 6 | DP re-lift of 5 (build) | 21.532732133 | 22.532466364 | 23.532417292 | **24.532417292** |
| 7 | `lpslack` polish (round 5/6 of the loop) | **21.532729164** | **22.532436630** | **23.532367091** | **24.532367092** |
| 8 | DP re-lift of 7 (build, +1.7e-6 at n=530) | 21.532729164 | 22.532436630 | **23.532367091** | 24.532367092 |

(final `final.csv` values are the bold ones; n=798 went 28.740334237 ->
28.740330093 with the same loop, and n=447 only 6.2e-9.)

Each `lpslack` round takes 43 s - 11 min per group (wall clock is set by the
largest `n`), each DP+build round ~2 min, and each `assemble` (which certifies
every candidate in both transports) ~1 min. Nothing in the loop is stochastic.

Medium-scale sanity check of the mechanism: `f` for this gadget family descends
`0.5330231 -> 0.5328300 -> 0.5327315 -> 0.5324173 -> 0.5323671` over the four
rounds above. The pocket's *own* internal contacts are already jammed (C1
measured that); what is not jammed is the way the pocket's edge meets the grid
seam, and the soft-angle SLP moves whole `squares` (not just seam offsets), so
the coupling relaxes a little every time the gadget is re-embedded at a new `K`.

## 3. Per-n result (certified, both transports, both references)

Produced by `poc/mk_table.py` from `final_verify.json`, which
`poc/verify_final.py` writes by reading `final.csv` **from its own bytes**
(`deg -> rad` at 80 dps) and re-certifying:

| n | C1 side | best side (literal / pandas) | dL vs C1 (lit / pan) | dL vs main (lit / pan) | residual to frontier | score gain |
|---:|---:|---:|---:|---:|---:|---:|
| 442 | 21.533023120317 | 21.532729164093 / 21.532729164093 | -2.939562e-04 / -2.939562e-04 | -3.011390e-04 / -3.011390e-04 | 9.806224e-03 | +1.3982e-05 |
| 447 | 21.656420965529 | 21.656420959283 / 21.656420959283 | -6.246889e-09 / -6.246889e-09 | -4.332902e-04 / -4.332902e-04 | 8.055106e-03 | +2.9547e-10 |
| 485 | 22.533023120317 | 22.532436630049 / 22.532436630049 | -5.864903e-04 / -5.864903e-04 | -5.946730e-04 / -5.946730e-04 | 9.534115e-03 | +2.6631e-05 |
| 530 | 23.533023120317 | 23.532367090941 / 23.532367090941 | -6.560294e-04 / -6.560294e-04 | -6.652122e-04 / -6.652122e-04 | 1.506217e-02 | +2.8496e-05 |
| 577 | 24.533023120317 | 24.532367091853 / 24.532367091853 | -6.560285e-04 / -6.560285e-04 | -6.662112e-04 / -6.662112e-04 | 1.506220e-02 | +2.7311e-05 |
| 798 | 28.740334237076 | 28.740330092571 / 28.740330092571 | -4.144505e-06 / -4.144505e-06 | -1.429962e-04 / -1.429962e-04 | 4.101428e-03 | +1.4671e-07 |

**sum of score gains: 9.656715e-05 vs C1, 1.235121e-04 vs fresh main.**
`n = 743` is not in `final.csv`: no optimiser and no cascade round moved it, so
there is nothing to claim. Literal and pandas sides are identical to the last
printed digit for every group, so the win does not depend on the friendlier
transport. `final.csv` is 3279 rows / 6 complete groups and **supersedes
`poc/pocket_lift_scan/final.csv` for these 6 `n`** (same `n`, smaller `L`).

Nominal union with fresh main: `22.153247131307513 - 1.235121e-04 = 22.1531236`
(C1 alone was `22.1532185`).

## 4. The frontier-target probe (step 2 of the brief): negative, and provably so

`poc/shrink_toward.py` scales each C1 group uniformly about its bbox centre to
the per-`n` `frontier` side from
`poc/koboshchan_coordinator/data/gap_vs_frontier.csv` (residual before
polishing: 3.8e-3 .. 1.6e-2), then the optimisers are started from that folded
start.

* **Fixed-angle LP: exactly a no-op.** The LP is `min L` over
  `a.(c_j - c_i) >= r_ij` with one *fixed* separating axis per pair; the axis
  choice is `argmax` of a homogeneous function of the offsets, so scaling the
  input by `f` scales the whole LP by `f` and returns the same configuration.
  Measured: `data/lp_fixed/ladder_*.json` for shrink factors
  `1-1e-6, 1-1e-5, 1-1e-4, 1-1e-3, 1-1e-2, 1-5e-2` — the LP side is **identical**
  (within its 1e-9 margin) at every factor and equal to the C1 side,
  i.e. the residual to the frontier does not close at all.
* **`lpslack` from the folded start: no win at any `n`**
  (`data/lpslack/frontier_results.json`). Its best float side is the folded side
  itself (e.g. 21.523396 for n=442 vs frontier 21.522923), but nothing at that
  side certifies — the SLP cannot repair 0.05 %-scale overlaps, so it returns
  `NO_WIN` for all 7.

So the remaining 3.8e-3..1.6e-2 residual to the all-team frontier is **not**
reachable by any local movement from these rows: it needs a different
construction, as C1 and cycles 1-3 already concluded. Reported as required, even
though it is not a win.

## 5. What was run (exact commands)

```bash
PY=.venv/bin/python ; D=poc/lift_polish

# 0. reconstruct the C1 groups as an incumbent CSV + their official sides
$PY $D/poc/mk_c1.py $D/data/c1_merged.csv
$PY poc/faultline_dp2/poc/mkreport.py $D/data/c1_merged.csv $D/data/c1_report.json

# 1a. fixed-angle centre LP, 12 rounds, no trust region
$PY poc/alln_fixed_lp/poc/alln_fixed_lp.py \
  --incumbent $D/data/c1_merged.csv --report $D/data/c1_report.json \
  --output $D/data/lp_fixed/results.json --wins $D/data/lp_fixed/wins.json \
  --n-list 442,447,485,530,577,743,798 --rounds 12 --jobs 7

# 1b. Lipschitz free-angle SLP
$PY poc/lip_slp/poc/lip_slp.py \
  --incumbent $D/data/c1_merged.csv --report $D/data/c1_report.json \
  --output $D/data/lipslip/results.json --wins $D/data/lipslip/wins.json \
  --n-list 442,447,485,530,577,743,798 --rounds 8 --jobs 7

# 1c. alternating soft-angle SLP + unbounded fixed-angle LP + uniform shrink
$PY poc/swarm_lpslack/poc/lpslack.py --n-list 442,447,485,530,577,743,798 \
  --jobs 7 --outer-iters 8 --fixed-rounds 8 --tr-sched 5e-4,5e-5 \
  --out $D/data/lpslack/results.json --jsonl $D/data/lpslack/results.jsonl
# (lpslack has a hard-wired portfolio loader; it already picks the C1 row for
#  these n because `poc/pocket_lift_scan/final.csv` is in the portfolio.)

# 2. keep the best certified candidate per n (both transports, strict)
$PY $D/poc/assemble.py --base $D/data/c1_merged.csv \
  --wins $D/data/lpslack/results.json ... --out $D/data/polished1.csv
$PY poc/faultline_dp2/poc/mkreport.py $D/data/polished1.csv $D/data/polished1_report.json

# 2b. the frontier-target probe (§4)
$PY $D/poc/shrink_toward.py $D/data/c1_merged.csv --out $D/data/shrunk_frontier.csv \
  --gap poc/koboshchan_coordinator/data/gap_vs_frontier.csv --target frontier
$PY poc/alln_fixed_lp/poc/alln_fixed_lp.py --incumbent $D/data/shrunk_frontier.csv \
  --report $D/data/c1_report.json --output $D/data/lp_fixed/results_shrunk.json \
  --wins $D/data/lp_fixed/wins_shrunk.json --n-list 442,447,485,530,577,743,798 \
  --rounds 12 --jobs 7
$PY $D/poc/run_lpslack_seeded.py --merged $D/data/shrunk_frontier.csv \
  --report $D/data/c1_report.json --n-list 442,447,485,530,577,743,798 \
  --out $D/data/lpslack/frontier_results.json --jobs 7
# factor ladder: same two commands with --factor 1-1e-6 .. 1-5e-2 → lp_fixed/ladder_*.json

# 3. one donor-cascade round (scan -> merge -> DP -> donors -> build+certify)
bash $D/poc/round.sh $D/data/polished5.csv $D/data/polished5_report.json r6 \
     442,447,485,530,577,798 $D/data/pockets_pareto4.json

# 4. further polish rounds reuse the proven optimiser with our own start rows
$PY $D/poc/run_lpslack_seeded.py --merged $D/data/polished6.csv \
  --report $D/data/polished6_report.json --n-list 442,447,485,530,577,798 \
  --out $D/data/lpslack/iter5_results.json --jobs 6 --outer-iters 10 \
  --fixed-rounds 10 --tr-sched 5e-4,2e-4,5e-5

# 5. deliverable + independent re-certification from final.csv bytes
$PY $D/poc/mk_final.py $D/data/polished_final.csv $D/final.csv
$PY $D/poc/verify_final.py $D/final.csv      # literal + pandas, vs fresh main AND vs C1
$PY $D/poc/mk_table.py $D/final_verify.json  # the table in §3
```

`run_lpslack_seeded.py` imports `poc/swarm_lpslack/poc/lpslack.py` **unchanged**
and calls `optimize_group` with our rows and our gate side; nothing about the
geometry, the shrink or the double certification is re-implemented. For the
sub-pocket scan the brief suggested copying `scan_pockets.py`; `round.sh` calls
the **original** `poc/pocket_lift_scan/poc/{scan_pockets,merge_pockets,dp,build_cert}.py`
instead, so the gadget semantics and the seam builder can never drift from C1's
certified path — only the inputs (polished donor rows) change.

## 6. Files

| path | what |
|---|---|
| `poc/mk_c1.py` | C1 `data/built/*.json` -> incumbent-schema CSV |
| `poc/assemble.py` | best **certified** candidate per `n` over many `wins.json` (literal **and** pandas) |
| `poc/extract_wins.py` | normalise `alln_fixed_lp` / `lip_slp` / `lpslack` outputs to one `wins.json` shape |
| `poc/shrink_toward.py` | uniform shrink to a factor or to the frontier side (frontier probe) |
| `poc/run_lpslack_seeded.py` | run the proven `lpslack` optimiser from arbitrary rows |
| `poc/round.sh` | one donor-cascade round: scan -> merge -> DP -> donors -> build+certify |
| `poc/mk_donors.py` | donor table = fresh main with the polished rows substituted |
| `poc/mk_final.py` | `final.csv` writer (`id,x,y,deg`, `s`-prefixed, 75-digit degrees) |
| `poc/verify_final.py` | re-certify `final.csv` from its own bytes vs **fresh main** and **C1** |
| `poc/mk_table.py` | markdown table for §3 |
| `data/c1_merged.csv`, `data/c1_report.json` | the C1 groups and their official 50-digit sides |
| `data/polished{1..7}.csv`, `data/polished_final.csv`, `*_report.json` | per-round "best certified per n" tables |
| `data/lp_fixed/`, `data/lipslip/`, `data/lpslack/` | optimiser outputs (`results.json`, `wins.json`, `iter*.json`, `frontier_results.json`) |
| `data/shrunk_frontier.csv`, `data/shrunk_f*.csv`, `data/lp_fixed/ladder_*.json` | the frontier-shrink probe and its factor ladder |
| `data/pockets_pareto2.json`, `pockets_pareto3.json`, `pockets_pareto4.json`, `pockets_pareto_r{6,7}.json` | pocket libraries after adding each round's polished rows as donors |
| `data/pred{2,3,4}.json`, `data/pred_r{6,7}.json` | DP predictions (composition per `n`) |
| `data/built2/`, `built3/`, `built_r6/`, `built_r7/` | built + certified groups (radians, decimal strings) |
| `data/donors3.csv`, `donors_r6.csv`, `donors_r7.csv` | donor rows for `build_cert.py` (main + polished) |
| `final.csv` | the deliverable groups (`id,x,y,deg`) |
| `final_verify.json` | `verify_final.py` output (both modes, both references) |
| `logs/` | every run log quoted above |

**Not checked in (regenerable, one command each):** the donor CSVs
(`data/donors3.csv`, `donors_r6.csv`, `donors_r7.csv`, 33 MB each — rebuild with
`poc/mk_donors.py POLISHED.csv OUT.csv`), the raw scan outputs
(`data/pockets_{imp,p2,p3,r6,r7}_o*.json`, ~18 MB — the loop in `poc/round.sh`),
and the individual shrink-factor CSVs of the frontier ladder (rebuild with
`poc/shrink_toward.py --factor ...`). Everything derived from them is kept in
`data/pockets_pareto*.json`.

## 7. Where it stops / what to do next

* The cascade converged on `n = 442/485/530/577` (the `D = 1` family). Their
  excess `f = L - K` went `0.533023 -> 0.532729` (442), `0.533023 -> 0.532437`
  (485), `0.533023 -> 0.532367` (530, 577). The per-round gains on those four `n`
  were `1.9e-4, 4.6e-5, 2.6e-4..3.1e-4, 3.0e-5..5.0e-5, 3.0e-6..4.9e-6, 1.7e-6`
  — **not** monotone: every DP+build re-assembles a *fresh* rigid composition,
  which re-arms 1e-4-scale slack, and only the following polish round can spend
  it. A round is now worth ~1e-6 (score), unless a *new* `g = 1` gadget with a
  smaller footprint appears — in particular one with `w, h <= 20`, which would
  also reach `K = 20` (`n = 401`, worth ~9.6e-6 score).
* `n = 447` is stuck at `dL = 6.2e-9` (its gadget is a `(11,21,+6)` cut out of a
  30x30 row; the pocket side is empty, so nothing to squeeze).
* `n = 743` never moved under any of the three optimisers: 5 runs from its C1
  row (2-3 trust-region schedules, 8-10 outer iterations each) plus 1 run from
  the frontier-folded start, all `NO_WIN`.
* `n = 798` gives dust: 3.6e-6 in round 1, then 4.9e-7, 2.7e-8, 3.2e-9.
* The DP is closed again after round 8: the only remaining predicted cell is
  `n = 585` at 1.06e-8, which `build_cert.py` cannot realise (575 of 585 squares).
* The mechanism is not specific to the C1 rows: **any** seamed composition should
  be re-scanned and re-lifted after a free-angle polish. Running this loop on the
  `poc/faultline_dp2` / `poc/gadget_squeeze` compositions (and on the *whole*
  portfolio's `D = 1..3` family rows, not just the 7 C1 `n`) is the obvious
  scale-up and is where the next 1e-4 of score most likely sits.
