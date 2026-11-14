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

- cycle 1 launched 2026-11-14 06:xx UTC (5 subagents).
- **H3 `dsmall_band` — negative (documented).** A rigid "upright grid + one
  free-angle tilted band" reaches at most `ncol²` (the plain grid) at every
  target side: 900/841/961 at n=901/842/962, short by 1; 324/361 at n=328/365,
  short by 4. The incumbent's upright fill is a free-form polished patchwork, not
  a lattice — feeding the incumbent's own tilted block into any rigid fill gives
  only 850 of 901 squares. The missing square is not a band re-angling.
- **H5 `minusE_shear` — negative (structural proof).** Any minus-E assembly that
  keeps an intact upright `K×K` lattice has side ≥ `K+1`; measured LP floor is
  `K+1+2..3e-8` (no translational slack) and the deepest overlap-free inward
  slide is `d=0`. Our n=492 row already sits at the `22 + 1/√2 = 22.7071` zipper
  optimum, which is 8.6e-3 *better* than the whole minus-E family.
- **H1 `smallD_lift` — dust wins.** Composition lifts of existing D=1 rows as
  `(20,20,1)` gadgets improve n=442/485/530/577 by `dL ≈ -5e-6..-8e-6` and
  n=853 by `-1e-7`.
- **H2 `gadget_squeeze` — 5 certified wins (dust) + demand table.** Squeezing the
  binding pockets found **no** lower `f` (the pockets are rigid jams: 0-2 of 22
  hop trials re-polish feasibly, returning the same `f`). But the composition DP
  over a richer library certifies n=442/485/530/577 (the `(20,17,+1)` proper
  sub-pocket of n=401) and n=585. Total score gain `1.23e-6`. The valuable
  output is `docs/demand.txt`: the big-gap rows have **saturated seam budgets**
  and need a *replacement* pocket with better gain density at `f≈0.60-0.67`
  (small footprint `w+h+1≤12`, `g≥2`). Notably n=367 could pay 1.1e-3 if a
  `(3,2,+1)` existed at `f≤0.6709`.
- **H4 `faultline_dp2` — 4 certified wins (dust) + mechanism.** Reproduced DP
  closure against the current incumbent. The `(20,17,+1)` **proper sub-pocket**
  of n=401 lifts to K=21..24 because the D=1 family is *non-monotone* in K
  (corrects loop 33). Recomputed pay lines: new `(4,*,3)`-type gadgets are worth
  ≤1e-4 total, not the stale 0.01-0.03. Also found a `dtheta` sign bug in
  `rectangle_gadgets/poc/rgad.py`. Next step: exhaustive proper-pocket scan.

## Hypotheses, cycle 2

Cycle 1 showed the frontier gap is *not* closable by rigid constructions,
isolated gadget squeezing, or composition of the existing library. The only
mechanical source of new strict wins left is **proper sub-pocket reuse**
(a sub-rectangle of a row that lifts to a non-monotone K), plus a definitive
test of whether small high-density pockets exist.

| # | idea | folder | expected | status |
|---|---|---|---|---|
| C1 | Exhaustive proper-pocket scan (all rows × all window placements) + DP lift | `poc/pocket_lift_scan` | 1e-6–1e-4 | queued |
| C2 | Exact minimal-rectangle search for pay-line pockets `(3,2,+1)`, `(4,7,+3)`, small `g=2` | `poc/small_pocket_exact` | up to 1e-3 (n=367) | queued |
| C3 | Long-budget hybrid lpslack+basin-hop on the top-8 gap rows at the frontier side | `poc/deep_gap_search` | low | queued |

## Honest assessment

Closing 0.053 to reach 22.0999 means matching top-team **private** free-form
search on roughly half of the 1000 puzzles. Loops 1-40 and cycle 1 show the
portfolio is a local optimum under every descent/stochastic/rigid-composition
method available here, and the frontier's advantage is a K-staircase of better
*dense free-form* packings at `K ≥ 20`. The mechanical sources above are real
but are 1e-6..1e-3 scale, not 5e-2. I will keep cycling and banking certified
wins; the target remains far out.



