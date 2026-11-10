# leader_poc_k2p4: Cats and boxes n=328 / n=365 (K^2+4)

Result: **negative. No packing below the incumbent was produced; no CSV written; merged.csv untouched.**

| n | target (Cats) | incumbent L (merged.csv) | best measured here | est. score gain |
|---|---|---|---|---|
| 328 | 18.599535465 | 18.614368938 | none below incumbent (SLP returns the incumbent unchanged) | 0 |
| 365 | 19.599535465 | 19.612860 (L-K 0.61286) | not attempted | 0 |

Potential if reproduced: (0.01483 + 0.01332) / sqrt(n) ~ 0.00082 + 0.00070 = ~0.0015 score.

## Findings
- Offset check on perpuzzle_best.csv: only n=328 and n=365 have L-K = 0.59953546(5); no other K^2+4 n shares it. L values 18.59953546480645 vs 19.599535464757366 differ by 5e-11 (numerical-polish noise), so the true constant is ~0.5995354647.
- No closed form found: integer-relation searches (a+b*sqrt(d), d=2,3,5,6,7,10, denominators <=24; integer polynomials of degree <=4 with coefficients in [-12,12]) all negative. It is probably a numerically optimised constant of a fixed rigid local gadget, not a Pell/trig constant. K-independence implies a fixed-size core plus K-dependent upright (rigid) filler.
- Incumbent for 328 (and 365) is a ~70-square diagonal band at two angle clusters (328: 0.7711 and 0.7997 rad, i.e. 44.2 and 45.8 deg; 365: 0.7678 and 0.8030) plus 0/90 deg squares. L-K differs between K (0.61437 vs 0.61286), so it is a per-n relaxation, not a K-independent template.
- `poc/slp.py`: sequential LP over x, y, angle of all squares (separating-axis constraints linearised, with both |cos|/|sin| sign branches imposed because parallel pairs sit on the kink of |sin(dtheta)|; trust region shrink, exact feasibility check). Started at the n=328 incumbent it finds **no improving step** (trust region shrunk to 1e-7): the incumbent is a strict local optimum under free angles for all squares. So polishing cannot reach 0.5995.
- `poc/hop.py`: dilate 6-9%, jitter tilted angles, SLP. 8 runs x 330 s ended at L-K 1.27 to 1.70 (not converged; SLP shrinks slowly without compaction moves). Not competitive.

## Not done / next
- The 0.0148 gap needs a different band structure (different angle pattern / count of tilted squares), not a local move. Plausible route: periodic-cell optimisation of the 2-angle band (cell fixed, K-independent) then tile it; not attempted within the ~30 CPU-min budget.
- n=365 not run. No official `squares score` runs were done since nothing beat the incumbent.

Files: poc/slp.py, poc/hop.py, data/hop_*.log/.npy, data/slp_n328.npy.
