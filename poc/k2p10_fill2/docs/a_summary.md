# k2p10_fill2 (head a): n=635 last-2-squares fill, NEGATIVE

No 635-square overlap-free layout below 25.7071067812 was found. Nothing certified, no CSV written, merged.csv untouched. Measured L: none (ours 25.7071067812, leader 25.69874).

## Overlap module (poc/a_overlap.py)
Vectorised SAT (cKDTree pairs, 4-axis penetration depth), wall violation, penalty with analytic gradient (x, y, angle). Tests: depth vs scalar corner-projection reference max err 2.7e-15 (6418 pairs); overlap set equals brute force over all 44850 pairs; gradient vs finite difference err 1.4e-7; incumbent 635: bbox 25.707106781186745, max overlap depth 0, 0 overlapping pairs, 2.6 ms.

## Attempts (633 base: w=8, a=(18,18), b=(6,6), t=1, l=25, L45=25.65685, dilated to L)
- Free-space scan (poc/a_pockets.py, grid 0.05, upright and 45deg unit squares): best placement of one more unit square overlaps neighbours by depth 0.183-0.193 at both L=25.657 and L=25.705. No pocket anywhere; free area (~27) is all in 0.25-area staircase triangles and thin wall slivers, so pocket merging would need to move rigid blocks by ~0.19+ per pocket, and sliding is blocked by contact chains.
- Approach 1 (poc/a_relax.py): insert 2 squares at best-clearance spots (random picks of top 400), L-BFGS on full penalty (all x, y, angle free) at L=25.705, 8 trials: residual E 0.019-0.063, max depth 0.018-0.052, 350-540 overlapping pairs.
- Approach 2 (poc/a_corner.py, log a_corner_log.txt), L=25.7070, 65 corner squares (within 9 of BL/TR corner) set to 45deg, atan(3/4) angles, or jittered, 2 new squares in the corners, same relax: E 0.039-0.126, max depth 0.030-0.069 (plain random insertion 0.035-0.042). Tilting made it worse than plain.

Best residual anywhere: max overlap depth 0.0184 (E=0.019) at L=25.705; 633+2 squares, 635 required, overlap NOT zero, so not a valid layout.

## Where it fails
The deficit is not local: two squares need ~2 area, but the spare area is spread as 0.25 triangles plus the 0.05 L slack (~2.6 area spread along the perimeter), and local relaxation only smears 0.02-0.05 overlap over hundreds of contacts. Penalty relax from a rigid lattice is stuck in a local minimum; gains need a different topology (not refinement of this one). Not tried: basin hopping/large moves, bisect-in-L (pointless since L=25.705 already fails), MILP pocket merge.
