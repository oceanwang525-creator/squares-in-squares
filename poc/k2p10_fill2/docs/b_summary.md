# b: fill the n-2 band cores (approach 3), n=635 -- negative

No overlap-free 635 layout found; nothing certified, merged.csv untouched. Measured L: none (ours 25.7071067812, leader 25.69874).

## Overlap module (poc/b_overlap.py, verified by poc/b_verify.py, output docs/b_verify.txt)
Vectorised SAT (4 axes, penetration depth = -max gap), cKDTree neighbours (r<sqrt2), wall violation, analytic gradient wrt x,y,angle.
Checks: pair gap vs independent polygon-projection reference max err 1.8e-15; energy vs reference diff 3e-14; gradient vs finite diff max err 6e-9; incumbent 635 (box 25.7071067812) energy 0, min gap 4e-16 (matches pell_corrugation.sat_check).

## Method (poc/b_fill.py, b_hop.py, b_runall.sh)
633-square 45deg cores (t=1; (w,a,b,l) = (8,18,18/6,25), (13,15/14,9/10,20), (25,6,6,18,18,8)), centres dilated to the box Lc (0 overlap at Lc=25.70), 2 extra squares at grid-searched least-overlap pockets (random among top 60, axis-aligned or 45deg), all 635 squares x,y,angle free, L-BFGS on sum depth^2 + wall^2, box hardened Lc+0.15 -> +0.05 -> Lc. Basin hopping (noise 0.02) on the best.

## Results (635 squares, residual overlap, no SLP stage reached because nothing got near feasible)
| cfg | Lc | trials | best E=sum depth^2 | max depth | pairs overlapping |
|---|---|---|---|---|---|
| w=13 | 25.70 | 3 (+8 hops) | 7.3e-3 | 0.0119 | 786 |
| w=25 | 25.70 | 3 (+1 earlier) | 1.06e-2 | 0.0172 | 238 |
| w=8 | 25.70 | 2 | 1.07e-2 | 0.0155 | 337 |
| w=8 | 25.705 | 3 | 1.69e-2 | 0.0198 | 645 |
| w=8 | 25.7071 | 2 | 1.89e-2 | 0.0207 | 572 |
Even at Lc=25.7071 (where a zero-overlap 635 exists) the method does not recover it.

## Where it stalls
Residual is a floor, not a slow descent: ~0.9 total depth (about 0.012 max) spread over hundreds of pairs, unchanged by hops. The two extra squares' area is pushed into diffuse overlap through the rigid 45deg staircase teeth; free area per pocket is far below 1, so L-BFGS from local pockets cannot open a unit hole. Larger L did not help (landscape roughness dominates). Not tried: moving pockets via combinatorial re-seeding of teeth, adding tilted pieces on the staircases beforehand.
