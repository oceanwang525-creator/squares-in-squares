# leader_poc_k2p10 (K^2+10 / K^2+11, K=24..31): negative, no packing beats the incumbent

Targets: n=635 (25.69874), 686 (26.68672), 687 (26.70641). Incumbent: 25.707107, 26.707107, 26.727922.

## What the incumbent is
- n=635 and 686 are the plain 45deg w=1 staircase band (no L-border): 33/34 unit squares in one chain on the main diagonal (pitch 1 along u), two upright staircases (a=24 / 25) and 1-square corners. L = a + 1 + 1/sqrt2 exactly (`poc/pell_corrugation/data/border-scan/merged.csv`).
- Chain u-span has slack 0.527 (D=46, 46/sqrt2 = 32.527), but a1+a2 is even, so the tooth tips of the two flanks are in phase and the out-of-phase corrugation theorem (q1) gives no first-order shrinkage.

## Measured
`poc/repolish.py` runs the existing 4-sign-piece epigraph LP polisher (`pell_corrugation.polish_band`, free angles, trust 0.05 / 0.01 rad, 6 iterations, ~15 s each) on the full incumbent rows:

| n | incumbent L | polished L | target L |
|---|---|---|---|
| 635 | 25.7071067812 | 25.7071069033 | 25.69874 |
| 686 | 26.7071067812 | 26.7071069083 | 26.68672 |

No change (+1.2e-7 is the 5e-9 clearance); the 45deg w=1 chain is a first-order-rigid local optimum. Nothing was certified or written as CSV because nothing improved; merged.csv is untouched.

## Why the cheap guesses do not work (analysis, not measured packings)
- A tip-to-tip channel of width 1 needs L reduced by 0.0084 (635) / 0.0204 (686), i.e. channel width 0.988 / 0.971. Tips are spaced sqrt2 along u but chain squares have pitch 1, so only ~1 tip in 3 can sit in a chain gap; the rest pin squares at exactly 45deg.
- Odd a1+a2 (out-of-phase) at w=1 gives L=m+0.207, which holds fewer squares; the Pell-17 gain is only ~0.0014-0.002, 10x too small.
- The only w with frac(L45) near 0.69 is w=13, odd a1+a2: a=(16,15), b=9, l=22 at K=25 gives L45=25.6924 but only 632 squares (K^2+7), not 635. Not built or certified.
- A K-dependent drop of ~0.01/K likely needs per-n bespoke structure (Cats uses 328 distinct L-K values), not a one-parameter change to this band.

Open: a search over (w, a1, a2, b1, b2, t) with corrugation for counts >= K^2+10 at L below incumbent was not run (outside 30 CPU min; scan_structured_borders already covers staircase-band cores).
