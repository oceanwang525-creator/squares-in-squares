# wildcard_idea: catalog block stitching (variant of option c)

Idea: take every incumbent packing with n_c < 400 as a core of side L_c, wrap it with an
upright L-border of thickness t (adds 2*t*floor(L_c)+t^2 squares, L = L_c + t), and compare
with the incumbent L at the resulting n (<= 1000). This is the cheapest form of "stitch known
small packings into a large square"; a core+border is the only stitch that adds no seam waste.

Result: `poc/border_wrap_audit.py` (float AABB screen on merged.csv, ~seconds):
- Wins: 0. Exact ties (|dL| < 1e-9): 3885, so the incumbent already contains every such wrap.
- Closest strictly worse wraps: core 227 t=1 (+0.000178), core 328 t=1 (+0.0015), core 205 t=1 (+0.0016).
- No CSV written, no official-metric run (nothing to certify). Estimated score gain: 0.

Honest negatives: only one-sided-fraction L-borders were tried. Not tried: rectangular
sub-blocks with different cores per side (needs best-known rectangle packings, absent from the
repo), knapsack allocation across several cores, tilted-wedge fills (option a), or two-orientation
strips (option b). The near-miss at 227+1 (0.000178) suggests nothing structural; a polish of
that core (n=227 -> 227+2*15+... L~15.x) was not attempted since polish from incumbents is a listed failure.
