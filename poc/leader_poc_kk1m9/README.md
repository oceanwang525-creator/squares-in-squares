# leader_poc_kk1m9 (n = K(K+1)-9): negative, no packing beats the incumbent

Targets: n=411 L=20.77414 (ours 20.78694), n=453 21.77798 (21.79140), n=497 22.78029 (22.79709).

## What was found
- The incumbent n=411 in merged.csv is not a 45-degree grid+strip. It has 229 tilted squares in two lattice-like pockets (NW and SE) at about 0.45-0.48 rad (~26.5 deg = atan(1/2)) with per-pocket angles already polished, in an upright 182-square remainder. Only the first 15-square groups share angles. So the "K-dependent non-45 strip angle" is already partly in our packing; the gap to Cats is not a missing angle but their lower waste in the pockets/boundary.
- No template with K-independent offset was derived or built in the budget. No exact-overlap-checked new packing was produced.

## Measured
- `poc/relax.py 411 20.77414 240`: scale the incumbent n=411 by 20.77414/20.78694 about its centre, then minimise the squared SAT penetration + wall violation (all 411 squares, x/y/angle free, L-BFGS, ~4 min). Penalty stalled at 1.33e-4 (from ~1.44e-4 after the first 47 s) and did not approach 0, so no feasible packing at the target L. Nothing certified; no CSV written; merged.csv untouched.
- Bug note: the script's final "final pen" print is wrong (in-place shift of a view of v), and it saves a shifted array; the per-iteration penalties in the log are correct. The saved .npy was deleted since it was infeasible.
- Official `squares score` was not run (no candidate). Measured L for the target n: ours 20.78694 (unchanged), target not reached.

## Open
Local relaxation of the incumbent is the already-failed free-angle route and is consistent with the incumbent being at a local optimum. A real win needs a different pocket topology (not tried).
