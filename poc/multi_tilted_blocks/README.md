# multi_tilted_blocks: n=908 block structure, MIS test (negative, no packing produced)

## Block structure of our n=908 row in pell_corrugation/data/border-scan/merged.csv (L = 30.63702940)
- 704 upright + 204 tilted. The tilted squares form two connected clusters of 102 (poc/extract.py):
  - NW: x in [0.50,10.03], y in [15.42,29.22]; 59 squares at 24.10 deg (atan(17/38)), 10 at 25.7, 3 at 23.9, 3 at 26.7, rest in between.
  - SE: x in [12.42,26.22], y in [2.50,12.03]; mirror image (angle -24.1 deg = 65.9 mod 90).
- Same structure as q3_public_leaders_anatomy.md (there 100+100, 13 columns). The central SW-NE diagonal is upright.
- Leader 908 = 30.605885 (ours 30.637029) and 969 = 31.605885 (= 908 + 1, i.e. one more L-border of 61 squares). 0.60588451 appears for no other n in perpuzzle_best.csv and is not a simple a + b*sqrt(m)/d form (searched m in {2,5,10,13,17,26,29,37,41}). No coordinates of the leader are in the repo, so its structure is unknown.

## Test
poc/mis.py: pool = upright lattices anchored at each of the 4 box corners (3600) + tilted lattice patches (angles +-24.1 deg, +-atan(1/2); 2x2 phases; regions NW, SE, SW corner, NE corner), 16784 candidates, 685k conflict edges (fast SAT from k2p10_fill2/poc/a_overlap.py), HiGHS MILP max independent set, 240 s limit, L=30.63.
- Result: 458 squares found, LP bound 1421 (far above 908). The edge-formulation MILP does not converge, so it says nothing about feasibility of 908. (A first run reported 8480 squares because of an index bug in the constraint matrix; fixed, discard it.)

## Squares vs L
| L | squares found | needed |
|---|---|---|
| 30.63 | 458 (MILP, not converged) | 908 |
Nothing measured at 908/909. No CSV was built, `squares score` not run, merged.csv untouched. Estimated gain: none shown.

## Open
Pool-MIS needs a clique-cover formulation or a seeded start from the incumbent's 2 blocks plus a third block; not done in budget.
