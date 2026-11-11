# Neighbour deletion + shrink (negative)

Take our packing for n+k (k=1..4, merged.csv), delete k squares, scale centres down to a target container L_t < our L(n), then relax with all x, y, angle free (L-BFGS on the exact SAT overlap penalty from `poc/k2p10_fill2/poc/a_overlap.py`, followed by a margin-hardening ramp 1e-8..1e-6 and an exact feasibility check). Code: `poc/nd.py` (L_t = ours - 2e-4), `poc/nd_fine.py` (L_t = ours - 1e-6). Raw output: `data/run1.log`, `data/run2.log`.

Candidate sets per (n,k): fewest contacts, 4 corners, centre, random tilted, tilted neighbourhood, random neighbourhood (6 per k, 24 per target; far fewer than 200 because of the CPU budget). 300 L-BFGS iterations per candidate.

## Result

No win. For all 25 targets no candidate became overlap-free at L_t (even 1e-6 below ours), so there is no measured L below ours for any target. No CSV was built and `squares score` was not run, since there is nothing to swap in.

| n | ours L | leader L | best measured L | note (best candidate, k=1 every time; residual energy at ours-2e-4) |
|---|---|---|---|---|
| 971,911,910,972,908,793,861,803,969,747,862,913,695,973,741,854,497,496,909 | see per_n_comparison.csv | | none < ours | residual energy 3e-5..7e-4, max overlap depth 4e-4..3e-3 |
| 328, 686 | 18.6144, 26.7071 | | none < ours | worst: E 2e-3 and 9e-4 |
| 791, 634, 687, 492 | 28.6370, 25.7071, 26.7279, 22.7071 | 28.6064, 25.6823, 26.7064, 22.6905 | none < ours | n+1 already has the same L as n; after deleting one square, still E~1e-12, depth ~1e-6 at ours-1e-6 (rigid) |

Estimated gain: 0. Plain scaling of the n+k packing after deleting squares does not unjam it: the remaining packing is held by the same chains, and the deleted squares in the tried sets (k=1 always best, larger k worse) do not release the container-spanning constraint. Because L(n+k) > L(n) in our data except at 4 n, shrinking would need to beat the n+k structure by more than the gap to the leader, which these relaxations do not get close to.

## Not done / caveats

- Heavy-ball hardening was not used (L-BFGS plus margin ramp instead); a feasible-but-not-hardened result would still have shown as a near-zero energy, which was never seen except the 1e-12 cases above (depth ~1e-6, not feasible).
- Other repo packing sources (non-merged.csv) were not tried; only 24 candidates per target, 300 iterations (800 in the fine run). Machine load was ~15-20 during the run.
- No bisection on L was run, since even the smallest shrink (1e-6) failed.
