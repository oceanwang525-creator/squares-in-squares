# koboshchan coordinator — squares-in-squares

Role: **coordinator only**. I brainstorm hypotheses, rank them against the
per-`n` frontier, spawn one subagent per hypothesis into its own `poc/{idea}/`
folder, review outcomes, and manage git/PRs. I do not run packing experiments.

## Ground truth (2026-11-14)

| quantity | value |
|---|---|
| incumbent `poc/pell_corrugation/data/border-scan/merged.csv` | **22.153257205012313** |
| merge-branch `merged-22.15325720501.csv` | identical SHA256 `61004c6f…` |
| official certification | 1000/1000 accepted, missing 0, rejected 0 |
| goal | **≤ 22.0999** (need Δ ≤ −0.053357) |
| Oct-29 leaderboard per-puzzle union (`leader_curve_analysis/data/perpuzzle_best.csv`) | 22.067663505 |
| all-team frontier envelope (`frontier_nov12/docs/per_n_gap.csv`) | 22.050884 |

So the target sits about **52 % of the way** from us to the all-team
frontier. The gap to the real leaderboard union is 0.0894.

Data handed to subagents: `data/incumbent-22.15325720501.csv`,
`data/gap_vs_frontier.csv` (columns `n,K,D,ours,frontier,leader,dL,gap_score,gap_leader`).

## Where the gap lives (analysis)

- Gap mass by `K=floor(sqrt(n))`: K≤15 ≈ 0; K16-19 0.012; K20-23 0.018;
  K24-27 0.027; **K28-31 0.046**. The problem is *large-n construction
  quality*, not small/mid `n`.
- `c = L − K` per `D = n − K²`: ours is roughly K-independent at large K
  (D=1 → 0.5208, D=7 → 0.6087..0.6341, D=8 → 0.634), while the frontier's `c`
  **decreases with K** (D=1 → 0.5022, D=7 → 0.6059, D=8 → 0.6059). The
  frontier has better family constructions; ours appears to be “K×K grid plus
  a fixed-excess corner gadget”.
- The portfolio is already a local optimum under fixed-angle LP, Lipschitz
  free-angle SLP, contact-topology search, clustered SA, replica-exchange PT,
  MILP set-packing windows, insertion/deletion, periodic strip cells and
  public-engine basin hopping (loops 1-40 in `poc/loop_log.md`). **More local
  polish cannot pay.** New *constructions* are required.
- `poc/rectangle_gadgets` shows the tracked rows ARE gadget compositions, and
  the frontier wins by a **lower `f` for the same `(w,h,g)` pocket**. This is
  the cleanest lever and motivates H1/H2/H4 below.

## Hypotheses, cycle 1 (ranked by expected gain)

| # | idea | folder | expected | status |
|---|---|---|---|---|
| H1 | Realise the frontier small-`m` `K²+D` family (D=6,7,8) and lift | `poc/smallD_lift` | 0.01–0.03 | running |
| H2 | Rectangle-gadget f-squeeze on binding shapes | `poc/gadget_squeeze` | 0.005–0.02 | running |
| H3 | Non-45° tilted-band construction for D=1,2,4 | `poc/dsmall_band` | 0.005–0.015 | running |
| H4 | Fault-line DP over an expanded small-gadget library | `poc/faultline_dp2` | 0.01–0.03 | running |
| H5 | Minus-E corner-shear from a (K+1)² grid | `poc/minusE_shear` | 0.005–0.015 | running |

Each subagent's mandate: first pass on 1–3 `n` (~15 min). No sign of a
certified dL ≤ −1e-9 → write up the failure in its README, leave a
header-only `final.csv`, stop. A real win (certified dL ≤ −1e-9, ideally
≤ −1e-5, in **both** literal-decimal and pandas read_csv/to_csv transport) →
scale up.

## Outcome log

- cycle 1 launched 2026-11-14 06:xx UTC (5 subagents). Results appended here as
  they land.
