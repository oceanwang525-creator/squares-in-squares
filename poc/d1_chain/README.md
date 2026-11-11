# d1_chain: structure of the D=1 departure at n=401 (L-K 0.526143 vs ours 0.535534). Result: negative, nothing built

Estimated gain: 0. No feasible packing below ours, no CSV written, merged.csv untouched, no `squares score` run (nothing overlap-free to score).

## Finding: our n=401 is not a single 45deg chain
Picture: `docs/inc401.png`. Ours (L = 20.535534 = 20-3+5/sqrt2, exact bbox check) is two separate 45deg diamond clusters (about 40 squares each, centres near (-6,3) and (3.5,-6) in centred coordinates, i.e. symmetric about the diagonal x=-y, not on it), embedded in an upright lattice with 1/sqrt2-offset staircase rows, plus a small corrugated strip along the top (x~0.7..9.7, y 7.7..9.8, angles off-lattice by ~1e-3) and upright columns at x~8.7-9.7. So "one chain along the diagonal" and "fan/kink of a chain" are the wrong model for this incumbent; the leader's 0.0094 gain is of unknown form (two-diamond, not chain).

## Experiment (`poc/fan.py`, `poc/common.py`, reuses k2p10_fill2 a_overlap penalty)
All 401 squares free (x,y,angle), container shrunk to target L, penalty L-BFGS x3 rounds, 240 s single thread, ~45 restarts. Starts: incumbent scaled; or each diamond rigidly rotated by N(0,0.03) rad (fan about its centre); or additionally per-square angle jitter 0.02 (angle profile); position jitter 0.01.
| target L | min E | min max-violation |
|---|---|---|
| 20.5355 (ours) | 3e-9 | 8e-6 (jittered start, not converged to 0) |
| 20.532 | 2.2e-5 | 3.5e-4 |
| 20.529 | 7.1e-5 | 6.1e-4 |
| 20.526143 (leader) | 1.44e-4 | 8.7e-4 |
Violation grows smoothly (~ proportional to dL, E ~ dL^2) for every start kind; no basin with E->0 below ours, so the optimiser never found a feasible fan/rotation. Converged angle profile: tilted squares stay at 45deg to within ~1e-3 rad (the relaxation does not spontaneously fan), consistent with a rigid-diamond/upright-lattice lock.
Not done: explicit n=442 run, parametric kink/2-row/end-pair models, SLP. Local relaxation from the incumbent topology is ruled out at the 20-min level; the leader needs a different topology (e.g. different diamond sizes/positions or square counts per diamond), not an angle tweak.
