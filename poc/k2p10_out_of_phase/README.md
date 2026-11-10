# k2p10_out_of_phase: negative, no out-of-phase band holds n squares at L below ours

Targets (leader L): n=635 25.69874, n=686 26.68672. Ours: 25.7071067812, 26.7071067812. **Measured L: none. Nothing was certified, no CSV written, merged.csv untouched.**

## Method
`poc/enumerate_cores.py` enumerates every core `(t, w, a1, a2, b1, b2)`: staircase-band core of side `L0=(a1+a2)/2+1+w/sqrt2`, plus a t-layer upright L-border (`L=L0+t`).
Column count `ell` is the exact wall/stair-limited value used by `pell_corrugation.build_staircase_band_45` (`ell = 1+floor(u_hi-u_lo)`), stair clearance `max(a)+max(b)<=L0` enforced.
Count = T(a1)+T(a2)+T(b1)+T(b2)+w*ell + (K^2-(K-t)^2). Parity of a1+a2 splits in-phase (even) from out-of-phase (odd, q1 theorem). Output of the full run: `docs/odd_parity_min_L.txt`.

## Results (uncorrugated L45; counts are upper bounds, ignoring corner drops)
| K / n | in-phase, count>=n, min L45 | out-of-phase, L45 < ours, best count | out-of-phase, count>=n, min L45 |
|---|---|---|---|
| 25 / 635 | 25.70711 (our plain w=1 band; 5 configs) | **632** at 25.69239 (t=0, w=13, a=(16,15), b=(9,9), ell=22) | 25.74264 (w=20/17, t=1/0), +0.0355 above ours |
| 26 / 686 | 26.70711 | **683** at 26.69239 (t=1, w=13, same core) | 26.74264, +0.0355 above ours |

- Out-of-phase needs frac(w/sqrt2)+1/2 near 0.707 to land at the target L; w=13 (frac .1924) is the only width that does, and it is 3 squares short at both K.
  The missing 3 squares need roughly +0.06 in L by area (3/(2L)), versus the 0.0147 headroom to ours; the 0.0063/0.0204 headroom to the leader is smaller still.
- Pocket fill: `poc/holes.py` scans the w=13 core (632, built exactly by `poc/build.py`) for free unit squares at angles 0/45, step 0.1, container 25.7070 (below ours): **0 free spots**. No border/pocket fill exists for the 3 missing squares.
- Every odd config with count>=n sits at L45 >= X.74264 (+0.0355 over ours).

## First-order shrink predicted by q1 vs squares lost
- Shrink: dL ~ -|delta|/2 with K_pinch*|delta| <= slack(D)=frac(D/sqrt2). For the w=13 core D=13, slack 0.192, ~21 transitions -> |delta|<=0.009 -> dL ~ -0.004 (theory only, unmeasured, applies to a 632/683-square config).
  For the count-feasible odd cores (slack 0.09-0.45, L45 25.7426) the same bound gives <=0.004-0.01, nowhere near the 0.0355 needed.
- Squares lost: 3 at the only L45 that is competitive. So predicted shrink (<=0.005) cannot pay for the lost squares (~0.06 in L), and w=1 odd parity gives m+0.207 with far fewer squares.
- The in-phase incumbent has zero first-order shrink (q1 case 1) and is already the minimal-L in-phase config at n; repolish confirmed rigidity (`poc/leader_poc_k2p10`).

## Not done / open
- Mixed constructions (asymmetric stair shapes, non-staircase flank pieces, 2 bands, partial rows of the border) were not enumerated; only the standard core+L-border family. The leader's 0.0084/0.0204 gain likely needs such per-n bespoke structure.
- No corrugation LP polish was run on the 632/683 near-misses (they do not hold n squares so could not count).
