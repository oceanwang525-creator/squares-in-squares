# root coordinator ledger

## Current state

- Repository: `squares-in-squares-fork`.
- Fork remote: `https://github.com/oceanwang525-creator/squares-in-squares.git`.
- Upstream target: `https://github.com/koboshcom/squares-in-squares.git`.
- Cycle 1 sync: fast-forwarded local `main` from `5a09141` to `5f62a66` using `upstream/main`.
- Current merge artifact: `upstream/merge:merged-22.15325720501.csv`.
- Current best union in history: `poc/carried_lpslack_wins` lowers the merge score to
  `22.153257204712842`, dual-transport certified.
- Target score: `22.0999` or lower.
- Required additional certified score reduction from the best union: about `0.0533572`.
- Open PRs owned by the authenticated account: none at cycle start.

## Frontier gap summary

The strongest actionable table is `poc/frontier_nov12/docs/per_n_gap.csv`.
It is stale in the sense that later merges have improved some rows, but the
remaining target requires about 42% of the total positive gap recorded there.
The positive mass is concentrated in high `k`:

| k | recorded positive gap |
|---:|---:|
| 31 | 0.01548 |
| 30 | 0.01455 |
| 29 | 0.01320 |
| 28 | 0.01259 |
| 27 | 0.00990 |
| 26 | 0.00825 |

Large individual example gaps include n = 380, 801, 846, 684, 702, 592, 701,
979, 744, 877, 813, 928, 508, 259, 844, 682, 298, 632, 414, 548, 604, 445 and
555. Several are dust-scale by now, but the high-k family is still the right
place to search.

## Cycle 1 hypotheses

All five are ranked by expected gain and independence from the major dead ends
already documented in the repository. The largest known failures are local
LP/SLP polish, plain uniform shrink, single-square deletion, public harvesting,
45-degree rigid bands, ordinary window MIS at 367/911, ordinary parallel
tempering, soft-to-rigid hardening, and a four-block rigid pinwheel.

### 1. `cellgrain_reassemble`

**Rank:** highest.  
**Hypothesis:** large-n incumbents are frequently interlocked as two or more
nearly crystalline grains. A periodic mixed-angle grain cell can be optimized
independently, then reassembled at a slightly smaller box with a finite-end
seam solver. Unlike failed strip-cell work, this does not require one cell to
have more density than the current long arm; it searches for a different cell
whose finite block can replace a topological seam and pay for its ends.
**Quick test:** 1-3 high-gap n selected from 380, 684, 702, 801, 846.  
**Outcome:** negative, quick-test boundary reached.  Direct whole-cell SLP was
saturated on `846`, `380`, `702` and `801`; `684` produced only a dual-valid
improvement of `1.07e-15`, below the strict-win threshold and not exported.
Perturbed full/pocket/tilted-grain seeds and a C++ masked-melt probe on `846`
reached lower float sides but retained SAT penetration and failed exact
certification.  The agent committed `f493df1`; I removed the completed folder
from the working tree after that commit, preserving its history.

### 2. `contactbraid_reset`

**Rank:** high.  
**Hypothesis:** rather than deleting one square or relaxing the whole packing,
find a low-width braid of contacts crossing the box, remove an entire coherent
row/column of contacts, and solve a bounded reattachment LP. This is a
topology reset, not local descent. The target family is high-n rows where the
gap appears in a narrow shear corridor rather than a large pocket.
**Quick test:** 1-3 high-gap n selected from 508, 592, 632, 813, 928.  
**Outcome:** running.

### 3. `corner_lattice_borrow`

**Rank:** high.  
**Hypothesis:** several nearby n groups share repeated blocks. Instead of
keeping a donor fixed, allow a bounded set of donor squares to change integer
ownership between adjacent grains, then refill only the freed seam with exact
MILP. This differs from donor-slice and transplant work because ownership can
move in both directions and the refilled region is constrained by the new
contact graph rather than a rectangular crop.
**Quick test:** 1-3 high-gap n selected from 682, 744, 844, 877, 979.  
**Outcome:** negative, quick-test boundary reached.  Release scans on donors
`681, 743, 843, 876, 978` all stopped exactly one square short.  Tested 80
single-square releases per target, plus 30 each of `2x2` and `3x3` windows on
`877` and `979`.  No candidate reached a required group size, so no strict-win
claim was possible.  The agent committed `ed94194` before stopping.  I removed
the completed folder from the working tree after the failure commit, preserving
its history.

### 4. `dualwall_asymmetry_lp`

**Rank:** medium-high.  
**Hypothesis:** many high-n incumbents have strongly asymmetric wall contacts.
An exact two-wall LP with independent upright shifts and nonuniform boundary
growth may reveal a first-order path missed by uniform shrink and all-square
SLP. The solver should keep orientation groups fixed but allow blocks anchored
to opposite walls to move independently.
**Quick test:** 1-3 high-gap n selected from 259, 414, 445, 555, 604.  
**Outcome:** running.

### 5. `periodic_frontier_refit`

**Rank:** medium-high.  
**Hypothesis:** use the public per-n frontier only as a side target (never as
coordinates) and fit a small parametric periodic block to the incumbent's
contact pattern. Optimize the block side and phase in a smaller box, then
assemble multiple periods and certify the full n. This is distinct from failed
single-period strip cells and rigid 45-degree bands because both phase and
period can vary and the assembler may use different end pieces.
**Quick test:** 1-3 high-gap n selected from 380, 701, 846, 979.  
**Outcome:** running.

## Rules given to every subagent

- Work only inside `poc/{idea}/`.
- Create a README describing idea, commands and results.
- Create `final.csv` immediately, with the exact header `id,x,y,deg`; use a
  header-only file unless there is a certified candidate.
- Include complete groups only.
- Certify any claimed strict win with both the literal CSV check and the actual
  pandas `read_csv`/`to_csv` roundtrip.
- Commit failures as well as wins.
- Stop quickly if there is no sign of improvement on the 1-3 n quick test.
