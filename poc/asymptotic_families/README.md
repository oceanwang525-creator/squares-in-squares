# asymptotic_families: tilted-row strips vs rigid grid at s~25.7. Result: negative, nothing built or certified

Estimated score gain: 0. No 635-square layout below 25.7071067812 was found; no CSV written; merged.csv untouched.
Scripts: `poc/strip.py` (exact count + overlap check via `../k2p10_fill2/poc/a_overlap.py`), `poc/breakeven.py`. Outputs in `docs/`.

## Literature (recipes and constants)
- Erdős-Graham 1975: W(s)=O(s^(7/11)). Roth-Vaughan 1978: W(s) >= c*sqrt(s*frac(s)) (not O(s^a), a<1/2). Trivial grid: W~2s*frac(s).
- Chung-Graham 2009: O(s^((3+sqrt2)/7) log s); Wang 2016: O(s^(5/8)); Chung-Graham 2020 claimed O(s^(3/5)); Bui arXiv 2508.04603 and McClenagan arXiv 2602.01484 (2026) give full proofs of s^(3/5) (Bui: "Square packing with O(x^0.6) wasted area").
- Recipe (as summarised from 2602.01484; PDF 2508.04603 not readable here): grid fills S(x) except two rectangles of width h; tilted stacks of n squares of angle theta with sec(theta)=(n+tan theta)/h, theta ~ h^(-1/2); each end is a trapezoid of width ~sqrt(h) packed with near-horizontal strips of height ~2/theta using two alternating "shrink by one square per step" algorithms (angles phi~sqrt(2 theta)). Waste: x/sqrt(h) (tilted stacks, O(x*theta)) + h^(3/4) (trapezoid) -> h~x^(4/5), W~x^(3/5).
- Neither paper gives numeric constants or a crossover s (all implied constants unspecified; Erdős-Graham/Chung-Graham the same).
- Break-even: not derivable from the literature. If W=c*s^0.6, beating 0.8s+5 needs s^0.4 > c/0.8 roughly; c is unknown (the proofs need theta small, phi~sqrt(2theta) small, sqrt(h) many iterations, i.e. h >> 100, s >> 1000). Honest statement: break-even s is unknown and certainly far above 32; no theorem-level evidence of any gain at n<=1000, consistent with k2p10_curve_geometry (no s^0.6 decay visible at n<=1000).

## Smallest construction implemented (single strip, columns tilted by theta)
Strip height h=j+frac(L), width L, columns of m=j+1 squares, theta from m cos t + sin t = h, column spacing 1/cos t, end triangles wasted; upright block fills the other 25-j rows. Compare N*m with j*25.
- L=25.7071067812 (n=635 needs gain +10): best gain over j=1..25 is -3 (j=3: theta=0.698, N=18, 72 vs 75; j=1,2,4: -5). Every j has negative gain. Layouts overlap-free (0 overlaps, within [0,L] box), but hold fewer than the grid. L=25.69874: best also -3.
- Lower bound on needed gain is +10, so deficit is 13 squares.
- Scaling of this model (`docs/breakeven.txt`, f=0.7071): gain over grid first >0 at s~100 (+12), +44 at 200, +3400 at 6400, but waste stays ~0.83 s (e.g. s=800: 825 vs 0.8s+5=646; s=25600: 21264 vs 20485), i.e. always above the leader's linear fit within the tested range. The end triangles (~h*m*sin t per end) and the cos t column-spacing loss make the single strip linear in s; the s^0.6 gain requires the trapezoid-end algorithms above, which I did not implement (they need h>>100).

## Verdict
No tilted-row strip variant beats the rigid grid per area at s~25-32. The asymptotic constructions are irrelevant for n<=1000 (break-even unknown, far above 32). The leader's 0.8K+5 waste is a local/finite-size effect, not this asymptotic family.
