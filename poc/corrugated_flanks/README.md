# corrugated_flanks: free-flank 45deg band interface (n=635, 686). Result: negative, no packing built

Estimated score gain: 0. No 635/686 layout below our L was produced, no CSV written, `merged.csv` untouched, nothing sent to `squares score`. Measured L: none (target 635: 25.69874 <= L < 25.70711).

## Model (`poc/cell.py`, `poc/run_cell.py`)
Periodic strip in the band frame (u along band, v across), cell length P = k*sqrt2 (k staircase steps). Outer staircase layer on each side = rigid lattice with free u-phase; next staircase layer (k diamonds per side) and w rows of m band squares are fully free (x, y, angle). Penalty SAT overlap (`k2p10_fill2/poc/a_overlap.py`, 3 periodic images), L-BFGS, H (outer-to-outer distance) lowered in 0.01 steps with kicks, several random starts. Waste per length = ((H + sqrt2/2) P - squares) / P; rigid band = 0.7071 (H = w + 2 sqrt2). Resolution of H is 0.01. Independent scalar SAT check of the k=5,m=7 optimum: max depth 2e-15 (`poc/check.py`).

## Results (`docs/scan_w1_k3to7.jsonl`, `docs/scan_w23.jsonl`)
- Free phase, w=1, k=5, m=7: H = 3.7784 vs rigid 3.8284 (dH = -0.050), waste/len 0.667 vs 0.717 for the same cell with rigid H (0.707 continuous). Mechanism: flank squares tilt about -7 deg into short ramps (3-4 squares, 0.15 rise) separated by a jump. This is the corrugation, and it is a real first-order gain.
- But the two staircase lattices cannot take any relative u-phase. Phase of the nearest staircase layers differs by dphi = 0 or 0.5 (units of sqrt2) by lattice parity. Scan of dphi in steps of 0.125 (k=5,m=7): H = 3.838 (0), 3.818, 3.788, 3.798, 3.808 (0.5), 3.798, 3.788, 3.818.
- At the realizable phases, w=1, m = floor(k sqrt2): dphi=0.5 gives dH = -0.02 (k=5,m=7: 0.697 vs 0.717; k=7,m=9: 0.668 vs 0.798 is not comparable, fewer squares per length); dphi=0 gives dH = +0.01 (no gain, same as rigid within resolution). Cells with m < k sqrt2 -1 show large dH but only because squares are removed (m=k is just a shifted upright lattice); not a gain.
- w=2,3 (all band rows free), k=5,m=7: dH = -0.01 (dphi 0.5), +0.01 (dphi 0). Nothing beyond resolution. m=ceil(k sqrt2) is infeasible (no H found).
- Incumbent n=635 (w=1 chain, 33 tilted squares, 602 upright) has staircase phase measured dphi = 0 (both nearest layers at u/sqrt2 = 0.3536 mod 1). That is the class with no gain.

## Full 635 test (`poc/full635.py`)
Incumbent at L = 25.7071068 - delta, chain plus all squares within 3.0 of the diagonal free (173 squares), the two staircase blocks translated with their walls, penalty relaxation from 4 jittered starts per delta: best energy 3.4e-5 (delta 0.002) and 8.2e-4 (delta 0.01), scaling ~ delta^2 with no relief. No overlap-free state at any delta > 0. Squares count at the target L: 635 only at the rigid L = 25.70711 (incumbent); below it, overlap.

## Conclusion
Corrugated flanks give ~0.02-0.05 in transverse width for w=1, but only at phases (dphi ~0.25-0.75 away from 0) that the incumbent's lattice parity does not provide, and not at all for w>=2 within 0.01 resolution. A w=1 packing of the dphi=0.5 class (needs a different L/staircase offset; not found among the enumerated band configs, which hold n-2 at L < ours) is the only open lead; the ends of the chain have ~0.5 diagonal slack, enough for the 0.99/length density of the k=5,m=7 ramp pattern. Not built. The leader's 0.57/len is not reproduced (best realizable 0.697).
