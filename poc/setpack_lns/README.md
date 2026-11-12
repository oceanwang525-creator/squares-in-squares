# Exact set-packing LNS at the n=911 frontier side (negative)

## Idea

Generate a discrete candidate set of unit-square placements and orientations in
the box `L=30.6609125615235`, build the conflict graph, and select a
maximum independent set (non-overlapping squares) with exact MILP subproblems
in sliding windows (large-neighbourhood search). Any selection of >=911 is
certified with the official metric.

- candidates: 27,967 (0.5 grid, 8 orientations 0..75 deg);
- conflict edges: 1,954,325 (exact SAT, touching allowed);
- seed: the 30x30 upright grid (900);
- window MILP: `scipy.optimize.milp` (HiGHS), binary variables, pairwise
  conflict constraints, fixing the selection outside the window.

## Measured result: negative

- 400 windows of 5x5: best count **900**, no window improves the grid.
- 150 windows of 8x8: best count **900** again.

So the 900-grid is a local optimum under exact set packing in 8x8 windows; the
frontier's 911-square packing requires a coordinated global pattern that no
window sub-MILP finds. 0 wins, `merged.csv` unchanged.

Script: `poc/setpack_lns.py`; raw run in `docs/result.txt`.
