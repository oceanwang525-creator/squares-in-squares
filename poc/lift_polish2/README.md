# lift_polish2 — E2: is the polish→re-cut→re-lift cascade converged? (D2 scaled up)

Status: **positive, but not for the reason E2 predicted.**

* **9 certified strict wins, total score gain `-3.766008e-4` vs fresh main**
  (new nominal score **22.1525808549**). Every group is strict in **both**
  transports (literal decimal and a real `pandas.read_csv`/`to_csv` round trip)
  against **my own fresh main snapshot**
  (`poc/pell_corrugation/data/border-scan/merged.csv`, SHA256
  `9fe7cff8eecd92a1baffad52161c9c5ac2707cc04247e68f747d120ba15a7373`,
  regenerated 50-digit table `data/main_report.json`, score
  **22.152957455700296**, 1000/1000 accepted). Deliverable: `final.csv`
  (9 groups / 5254 rows) + `final_verify.json`.
* **`-1.970323e-4` of that gain is new here**; `-1.795685e-4` is a *restore*
  (see §1 — the E2 brief assumed the D1/D2 lifts were already merged upstream;
  **they are not**).
* The cascade itself **is converged**: rounds 1-2 paid, rounds 3-6 produced dust
  (`≤3e-8`). Its single real payment is one row that had **never** been through
  `lpslack` (`n=492`, **dL = -4.3530e-3** in round 1) — E2's specific hunch about
  n=492 was right and it is worth **2.5e-4 of score on its own**, but it is a
  one-shot, not a cascade.
* The re-cut/re-lift lever is **closed**: the sub-pocket DP finds **0** cells on
  the polished rows at step 0.125 (two rounds, four phases each) *and* at step
  0.0625 (two phases, targeted at the two rows that moved).
* The frontier residual is **not** reachable locally, and this time the reason is
  measured: the SLP's frontier-folded iterate sits 1.6e-4..6.5e-4 above the
  frontier in *float* but hides a **5.3e-4..7.4e-4 overlap**; repairing it needs
  a `1e-3` uniform expansion, which lands **8e-3..1.1e-2 worse** than our rows.

## 1. Ground truth first: fresh main does *not* contain the D1/D2 lifts

`mkreport.py` on the current `merged.csv` gives **22.152957455700296**
(matching the cycle-4 number), but the per-`n` sides of the nine lift targets are
the *pre-lift* values:

| n | fresh main `L` | best known before this POC | source | main − best |
|---:|---:|---:|---|---:|
| 442 | 21.533023264753 | 21.532729164093 | `lift_polish/final.csv` | 2.94e-4 |
| 447 | 21.656854249492 | 21.656420959283 | `lift_polish/final.csv` | 4.33e-4 |
| 485 | 22.533023221516 | 22.532436630049 | `lift_polish/final.csv` | 5.87e-4 |
| **492** | 22.707106781187 | 22.705882354153 | `pocket_lift_scan2` | **1.22e-3** |
| 530 | 23.533023413333 | 23.532367090942 | `lift_polish/final.csv` | 6.56e-4 |
| 577 | 24.533023524429 | 24.532367091853 | `lift_polish/final.csv` | 6.56e-4 |
| 740 | 27.705900674795 | 27.705882419967 | `pocket_lift_scan2` | 1.83e-5 |
| 743 | 27.753898760680 | 27.753852984553 | `pocket_lift_scan` | 4.58e-5 |
| 798 | 28.740473074686 | 28.740330092571 | `lift_polish/final.csv` | 1.43e-4 |

`data/start.csv` (`data/start_report.json`) is therefore "best known per `n`,
not main", worth `-1.795685e-4` of score by itself. Two `n` (492, 740) are D1
builds that `lift_polish` never polished; the other seven are `lift_polish`'s own
final rows.

I also checked the converse (`poc/harvest.py`): scanning **all 155**
`poc/*/final.csv` plus 32 `built*/n*.json` against main finds **exactly these 9
`n`** and nothing else — main is the best known row at the other 991 `n`. So
there is no extra maintenance harvest hiding in the portfolio.

## 2. What was run each round

Round = { `lpslack` (imported unchanged from `swarm_lpslack`, several
trust-region/iteration schedules) ‖ fixed-angle LP (12 rounds, no trust region,
also 6 random tie-break seeds) ‖ Lipschitz SLP (8-10 rounds) } → `assemble.py`
(best **certified** side per `n`) → `mkreport.py` → { re-cut sub-pockets of the
improved rows at step 0.125/0.0625 → Pareto merge → staircase DP → donor table →
`build_cert.py` } → next round's start rows.

| round | polish | cascade (scan → DP → build) | paid |
|---:|---|---|---|
| 1 | lpslack ×2 schedules + LP + SLP from `start.csv` | 4 phases @0.125 on the 9 polished rows | 492 **-4.353e-3**, 485 -7.68e-6, 530 -1.54e-6, 577 -1.53e-6, 442 -9.3e-8, 740 -4.8e-8, 798 -4.6e-9 |
| 2 | lpslack ×2 from `polished1` | 4 phases @0.125 on `polished2` | 485 -6.54e-6, 492 -6.1e-9, 740 -7.8e-9, 798 -5.2e-9 |
| 3 | lpslack (10 iters, tr 5e-4,5e-5) from `polished2` | — | 485 -3.2e-8 |
| 4 | lpslack from `polished3` | — | 485 -2.7e-9 |
| 5 | lpslack from `polished4`: 485 (default), 492 ×2 (tr `2e-3..5e-5`, tr `5e-5..5e-7`), LP w/ 6 seeds on 492, SLP on 492 | — | **nothing** (all `NO_WIN`) |
| 6 | lpslack (tr `1e-3,5e-4,1e-4`) on 442/447/740/743/798 | step **0.0625**, phases `(0.046875,0.046875)` and `(0.109375,0.046875)`, on 485+492 | 740 -5.5e-9, 798 -3.6e-9 (dust, below the 1e-9 claim bar) |

Two DP runs (`pred_r1.json`, `pred_r2.json`) and one fine DP (`pred_fine.json`)
all report **`DP strictly better at 0 n`** (555 cells evaluated each). The scans
did produce 4696 new oriented shapes and 19-23 "improved existing shape" items
(e.g. `(11,18,+6)` of n=740 `f 0.705887436 → 0.705882361`), but after the Pareto
merge the library is **1333 items** and the DP output is bit-identical — the
supply of pockets is not the constraint.

The fixed-angle LP never wins from any row in any round (`data/lp_fixed/*`, also
with `--seeds 6 --tie-tol 1e-5` on n=492: `wins=0`), and neither does
`lip_slp` (`data/lipslip/*`: `wins=0`, 9/9 groups in 3.7 s). Only `lpslack`
moves these rows — the same asymmetry `lift_polish` reported.

## 3. Per-n round table (50-digit sides from `mkreport.py` per round)

| n | fresh main | start (D1/D2) | round 1 | round 2 | round 3 | round 4 | final | dL vs main (lit = pan) | dL vs start | frontier | residual to frontier |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 442 | 21.53302326475 | 21.53272916409 | 21.53272907063 | 21.53272907063 | 21.53272907063 | 21.53272907063 | 21.53272907062 | -2.9419413e-4 | -9.3468e-8 | 21.52292294 | 9.80613e-3 |
| 447 | 21.65685424949 | 21.65642095928 | 21.65642095928 | 21.65642095928 | 21.65642095928 | 21.65642095928 | 21.65642095928 | -4.3329021e-4 | 0 (tie) | 21.64836585 | 8.05511e-3 |
| 485 | 22.53302322152 | 22.53243663005 | 22.53242895078 | 22.53242240724 | 22.53242237481 | 22.53242237216 | 22.53242237216 | -6.0084936e-4 | -1.4258e-5 | 22.52290251 | 9.51986e-3 |
| **492** | 22.70710678119 | 22.70588235415 | **22.70152938756** | 22.70152938142 | 22.70152938142 | 22.70152938142 | 22.70152938142 | **-5.5773998e-3** | **-4.3530e-3** | 22.69054048 | 1.09889e-2 |
| 530 | 23.53302341333 | 23.53236709094 | 23.53236555488 | 23.53236555488 | 23.53236555488 | 23.53236555488 | 23.53236555488 | -6.5785845e-4 | -1.5361e-6 | 23.51730492 | 1.50606e-2 |
| 577 | 24.53302352443 | 24.53236709185 | 24.53236555723 | 24.53236555723 | 24.53236555723 | 24.53236555723 | 24.53236555723 | -6.5796720e-4 | -1.5346e-6 | 24.51730489 | 1.50607e-2 |
| 740 | 27.70590067479 | 27.70588241997 | 27.70588237210 | 27.70588236434 | 27.70588236434 | 27.70588236434 | 27.70588235880 | -1.8315997e-5 | -6.1168e-8 | 27.69693762 | 8.94474e-3 |
| 743 | 27.75389876068 | 27.75385298455 | 27.75385298455 | 27.75385298455 | 27.75385298455 | 27.75385298455 | 27.75385298455 | -4.5776127e-5 | 0 (tie) | 27.75007811 | 3.77488e-3 |
| 798 | 28.74047307469 | 28.74033009257 | 28.74033008795 | 28.74033008272 | 28.74033008272 | 28.74033008272 | 28.74033007915 | -1.4299555e-4 | -1.3419e-8 | 28.73622866 | 4.10141e-3 |
| **Σ** | | | | | | | | **-3.7660081e-4** | **-1.9703236e-4** | | |

(`final` column = `final.csv` re-certified from its own bytes by
`poc/verify_final.py`: literal and pandas sides are **identical to all printed
digits** for every group. The `740`/`798` finals include the round-6 dust below
the 1e-9 claim bar; the claimed wins are the `dL vs main` column.)

Nominal union with fresh main: `22.152957455700296 - 3.766008e-4 =
22.1525808549`. `n=447` and `n=743` are pure restores: **no** polish run moves
them at all (three `lpslack` schedules each, plus the 12-round fixed-angle LP in
two rounds, the Lipschitz SLP and the frontier-folded start), so their claim is
identical to `lift_polish`'s.

## 4. Where the money came from

* **`n=492` is 67 % of the whole POC gain (`-4.3530e-3`, score `-1.964e-4`).**
  It is the D1 two-pocket seam (`(13,5,+3)`<-585 + `(9,16,+5)`<-686, rigidly
  built at `22 + 0.705882354`). One `lpslack` pass — which frees *all* angles, so
  the rigid pocket bound `f = 0.705882` dissolves — takes it to
  **`22 + 0.701529381`**. This is the only row in the run that had never been
  polished, and it is exactly the E2 hunch. It converged in that single pass:
  three further schedules (incl. `tr 2e-3..5e-5` and `tr 5e-5..5e-7`), the
  fixed-angle LP with 6 random tie-break seeds, and the Lipschitz SLP all return
  `NO_WIN` on it.
* **`n=485`** is the only genuine multi-round cascade (`-1.43e-5` over four
  rounds: `7.68e-6, 6.54e-6, 3.2e-8, 2.7e-9`) — and it dies at round 5. Its
  gadget is the `D=1` family's `(18,21,+1)`-type seam; the decrescendo is
  exactly the "each `build` re-arms ~1e-6, the next polish spends it" pattern
  `lift_polish` described, and it is now exhausted.
* `n=530/577` paid once (`-1.54e-6` each) and are dead afterwards; `442/740/798`
  paid ≤1e-7 and are dead; `447/743` never moved.

## 5. The frontier residual (step 3 of the brief) — measured, not just negative

`shrink_toward.py` folds each row to its per-`n` `frontier` side
(`poc/koboshchan_coordinator/data/gap_vs_frontier.csv`) and `lpslack` is started
from that folded row (gate = our current certified side). All nine `NO_WIN`
(`data/lpslack/frontier.json`, `frontier_rest.json`):

| n | frontier | our certified `L` | residual (certified) | SLP's best *float* side | float residual | LADDER repair: first certifiable side |
|---:|---:|---:|---:|---:|---:|---:|
| 442 | 21.52292294 | 21.53272907 | 9.8061e-3 | 21.5233803 | 4.574e-4 | 21.5439017 (+1e-3 expansion) |
| 447 | 21.64836585 | 21.65642096 | 8.0551e-3 | 21.6487972 | 4.313e-4 | — |
| 485 | 22.52290251 | 22.53242237 | 9.5199e-3 | 22.5233272 | 4.247e-4 | — |
| 492 | 22.69054048 | 22.70152938 | 1.0989e-2 | 22.6911865 | 6.460e-4 | 22.7127155 |
| 530 | 23.51730492 | 23.53236555 | 1.5061e-2 | 23.5179479 | 6.430e-4 | 23.5404628 |
| 577 | 24.51730489 | 24.53236556 | 1.5061e-2 | 24.5179217 | 6.168e-4 | 24.5414367 |
| 740 | 27.69693762 | 27.70588237 | 8.9447e-3 | 27.6973099 | 3.723e-4 | — |
| 743 | 27.75007811 | 27.75385298 | 3.7749e-3 | 27.7502385 | 1.604e-4 | — |
| 798 | 28.73622866 | 28.74033008 | 4.1014e-3 | 28.7363973 | 1.686e-4 | — |

**Why the float residual is an illusion.** `lift_polish` reported these float
sides but could not explain them; `poc/frontier_repair.py` does. It imports
`lpslack.py` unchanged and only replaces its shrink-only repair with a ladder
of *uniform expansions* `(1+d)`, `d = 0 … 1e-2`, certifying each in both
transports. Measured minimum separating gap of the "best" float iterate:

| n | min sat gap of the SLP iterate | smallest `d` that certifies | that side | vs our certified row |
|---:|---:|---:|---:|---:|
| 442 | **-5.261e-4** | 1e-3 | 21.5439017 | +1.12e-2 |
| 492 | **-5.736e-4** | 1e-3 | 22.7127155 | +1.12e-2 |
| 530 | **-7.393e-4** | 1e-3 | 23.5404628 | +8.10e-3 |
| 577 | **-7.091e-4** | 1e-3 | 24.5414367 | +9.07e-3 |

i.e. the folded start's overlaps are **never repaired by the SLP** (the "best
float side" is the folded configuration almost untouched: `min gap ≈ -(shrink
factor)`, `4.2e-4..7.4e-4`), and the cheapest configuration the certifier accepts
is **8e-3..1.1e-2 worse** than what we already have. The residual to the
all-team frontier therefore needs a different construction, not local movement —
the same conclusion as cycles 1-3, now with the overlap magnitude on record.

## 6. Verdict on E2

**E2 is half right and half wrong, and the difference is worth knowing.**

* Right: the *portfolio* was not converged — one `lpslack` pass over rows that
  had never been polished is worth `4.4e-3` of `dL` at n=492. Any row in the
  portfolio that is a *rigid composition* (built by a DP/`build_cert`-style
  assembler) must be treated as unpolished until proven otherwise: freeing the
  angles removes the rigid pocket bound outright.
* Wrong: the *cascade* (polish → re-cut → re-lift) is converged. Six polish
  rounds and three DP rounds gave `-4.353e-3` in round 1, `-6.5e-6` in round 2,
  then `≤3e-8`. The re-cut/re-lift half is **structurally dead** here: two
  independent scans of the polished rows (step 0.125 × 4 phases, step 0.0625 × 2
  phases) leave the 1333-item Pareto library and the DP prediction bit-identical
  — consistent with D1 §4 (a built row is pocket+grid, and a polished row's
  sub-rectangles are dominated by pockets the library already had).
* Consequence for the coordinator: **merge `final.csv` (3.766e-4) — 1.796e-4 of
  it is a restore that has been sitting in `lift_polish`/`pocket_lift_scan*`
  finals since cycle 3 and is not in main.** Do not expect more rounds of this
  loop to pay; the next 1e-4 needs a *new construction*, not more polish. If the
  loop is ever re-run, spend it on freshly built composition rows, not on rows
  that have already been polished once.

## 7. Run commands

```bash
PY=.venv/bin/python ; D=poc/lift_polish2 ; M=poc/pell_corrugation/data/border-scan/merged.csv

# 0. fresh-main 50-digit side table (148 s; score 22.152957455700296)
$PY poc/faultline_dp2/poc/mkreport.py $M $D/data/main_report.json

# 1. best-known-per-n start rows (main + all poc/*/final.csv + all built*/n*.json)
$PY $D/poc/mk_start.py $D/data/start.csv --prov $D/data/start_prov.json
$PY poc/faultline_dp2/poc/mkreport.py $D/data/start.csv $D/data/start_report.json
$PY $D/poc/harvest.py $D/data/harvest.csv --prov $D/data/harvest_prov.json   # audit: only these 9 n

# 2. round 1 polish (all 9 n, two lpslack schedules; imported unchanged)
$PY poc/lift_polish/poc/run_lpslack_seeded.py --merged $D/data/start.csv \
  --report $D/data/start_report.json --n-list 442,447,485,492,530,577,740,743,798 \
  --out $D/data/lpslack/r1a.json --jobs 9 --outer-iters 8 --fixed-rounds 8 --tr-sched 5e-4,5e-5
$PY poc/lift_polish/poc/run_lpslack_seeded.py --merged $D/data/start.csv \
  --report $D/data/start_report.json --n-list 442,485,492,530,577 \
  --out $D/data/lpslack/r1b.json --jobs 5 --outer-iters 10 --fixed-rounds 10 \
  --tr-sched 1e-3,5e-4,1e-4
$PY poc/alln_fixed_lp/poc/alln_fixed_lp.py --incumbent $D/data/start.csv \
  --report $D/data/start_report.json --output $D/data/lp_fixed/r1.json \
  --wins $D/data/lp_fixed/r1_wins.json --n-list <same 9> --rounds 12 --jobs 6
$PY poc/lip_slp/poc/lip_slp.py --incumbent $D/data/start.csv \
  --report $D/data/start_report.json --output $D/data/lipslip/r1.json \
  --wins $D/data/lipslip/r1_wins.json --n-list <same 9> --rounds 8 --jobs 6

# 3. assemble the best certified side per n, then the next round's side table
$PY poc/lift_polish/poc/assemble.py --base $D/data/start.csv \
  --wins $D/data/lpslack/r1a.json $D/data/lpslack/r1b.json --out $D/data/polished1.csv
$PY poc/faultline_dp2/poc/mkreport.py $D/data/polished1.csv $D/data/polished1_report.json

# 4. one cascade round: scan (4 fine phases, step 0.125) -> Pareto merge -> DP
#    -> donors -> build+certify;   round.sh deletes the raw scans and donor CSV
bash $D/poc/round.sh $D/data/polished1.csv $D/data/polished1_report.json r1 \
     442,447,485,492,530,577,740,743,798 poc/lift_polish/data/pockets_pareto_r7.json 4

# 5. rounds 2-6: same two commands with polished{N} and the new schedules
#    (2: r2a/r2b, 3: r3a, 4: r4, 5: r5_492big/r5_492small/r5_485 + lp 6 seeds,
#     6: r6_sweep 1e-3,5e-4,1e-4 + step-0.0625 scan of 485/492)
$PY poc/pocket_lift_scan2/poc/scan_pockets.py $D/data/polished5.csv \
   $D/data/scan2/fine_0.046875.json --step 0.0625 --fmax 1.10 --cap 2000 --procs 2 \
   --offx 0.046875 --offy 0.046875 --rows 485,492
$PY poc/pocket_lift_scan2/poc/merge_pockets.py --gadlib poc/faultline_dp2/data/gadlib.json \
   --out $D/data/pockets_fine.json $D/data/pockets_r2.json $D/data/scan2/fine_0.046875.json \
   $D/data/scan2/fine_0.109375.json
$PY poc/pocket_lift_scan2/poc/dp.py extra --lib poc/faultline_dp2/data/gadlib.json \
   --pockets $D/data/pockets_fine.json --report $D/data/report_full_r5.json \
   --out $D/data/pred_fine.json

# 6. frontier probe (folded start) + the expansion-repair analysis
$PY poc/lift_polish/poc/shrink_toward.py $D/data/polished1.csv \
   --out $D/data/shrunk_frontier.csv --gap poc/koboshchan_coordinator/data/gap_vs_frontier.csv \
   --target frontier
$PY poc/lift_polish/poc/run_lpslack_seeded.py --merged $D/data/shrunk_frontier.csv \
   --report $D/data/polished5_report.json --n-list 442,492,530,577 \
   --out $D/data/lpslack/frontier.json --jobs 4
$PY $D/poc/frontier_repair.py $D/data/shrunk_frontier.csv \
   --report $D/data/polished5_report.json --n-list 442,492,530,577 \
   --out $D/data/lpslack/repair_rest.json --jobs 3

# 7. deliverable + independent re-certification from final.csv's own bytes
$PY poc/lift_polish/poc/mk_final.py $D/data/polished5.csv $D/final.csv
$PY $D/poc/verify_final.py $D/final.csv      # literal + pandas, vs fresh main AND vs start
$PY $D/poc/mk_table.py                       # the table in §3
```

## 8. Files

| path | what |
|---|---|
| `final.csv` | **deliverable**: 9 complete groups (442,447,485,492,530,577,740,743,798), 5254 rows, `id,x,y,deg`, `s`-prefixed 75-digit degrees |
| `final_verify.json` | re-certification from `final.csv`'s own bytes: literal + pandas sides, `dL` vs fresh main and vs start |
| `poc/mk_start.py` | best-known group per n over main + every `poc/*/final.csv` + every `built*/n*.json` (`data/start_prov.json` = provenance) |
| `poc/harvest.py` | audit that main is best-known everywhere else (found exactly the same 9 n) |
| `poc/round.sh` | one cascade round: parallel fine scan → Pareto merge → report patch → DP → donors → build+certify (raw scans + donor CSV deleted afterwards) |
| `poc/frontier_repair.py` | **new** — `lpslack` with an *expansion* repair ladder; measures the overlap hiding behind the frontier-folded float side |
| `poc/verify_final.py` | re-certify `final.csv` in both transports vs fresh main **and** vs the D1/D2 start |
| `poc/mk_table.py`, `poc/extract_groups.py` | table generator; group extractor |
| `data/main_report.json` | fresh main, 50-digit, score 22.152957455700296 (148 s) |
| `data/start.csv`, `start_report.json`, `start_prov.json` | D1/D2 best-known rows and their official sides |
| `data/polished{1..5}.csv` + `*_report.json` | per-round best certified per n (+`.detail.json` = per-n candidate log) |
| `data/lpslack/` | `r1a,r1b,r2a,r2b,r3a,r4,r5_*,r6_sweep.json`, `frontier*.json`, `repair_*.json` |
| `data/lp_fixed/`, `data/lipslip/` | the fixed-angle LP (12 rounds, no trust region) and Lipschitz SLP runs — all `wins=0` |
| `data/pockets_{r1,r2,fine}.json`, `data/pred_{r1,r2,fine}.json` | Pareto libraries and DP output (`0` predicted cells each) |
| `data/scan2/` | the two step-0.0625 scan outputs (322 KB) |
| `data/harvest.csv`, `harvest_prov.json` | the portfolio-vs-main audit |
| `data/table.md` | §3 table, generated |
| `logs/` | every run quoted above |

Not kept (regenerable, one command each): the donor CSVs (33 MB), the raw
step-0.125 scans (~18 MB), and the four `polished*_report_full_*.json`.
