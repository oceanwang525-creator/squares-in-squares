# k2p10_band_search: negative (n=635, 686)

No band configuration fits n squares below our L. Nothing was relaxed, certified or written as CSV (nothing reached count >= n); merged.csv untouched. Estimated gain: 0.

## Method
1. `poc/enum45.py`, `poc/enum45_L.py`: exhaustive enumeration of 45deg staircase-band cores + t-layer L-border over (w<=39, a1, a2, b1, b2, t), with exact square counts (same rules as `pell_corrugation.build_staircase_band_45`: corner drops, max column count l) at every L45 in [target, ours], and for free L >= L45 (extra transverse slack wasted, longitudinal span grows). Count formula reproduces the incumbent (635 at 25.70711 w=1, 686 at 26.70711 w=1).
2. `poc/bandmilp.py`, `poc/sweep.py`: slanted-band candidate pool (4 corner-anchored upright lattices region-restricted by side of band line, plus w columns of tilted squares at angle th, free offset and longitudinal phase) and HiGHS max-independent-set (3 s limit each). Swept th in {45, 43.6 (20-21-29), 42, 40, 36.87 (3-4-5), 33, 30, 28.07 (8-15-17), 22.62 (5-12-13), 16.26 (7-24-25)}, w in {2,3,4,5,6,8,10,13}, 13 offsets x 5 phases, at L = target.

## Best squares count vs L (needs >= n)
| n | L | best count | config |
|---|---|---|---|
| 635 | 25.69874 (target) | 633 | rigid 45deg w=8, a=(18,18), b=(6,6), t=1, l=25 (L45=25.65685; same count for every L up to 25.7071) |
| 635 | 25.69239 (w=13, a=(15,14), b=(9,10), t=1) | 633 | rigid 45deg (the "632" candidate is actually 633) |
| 635 | 25.70711 (ours) | 635 | w=1 incumbent |
| 686 | 26.68672 (target) | 684 | rigid 45deg w=8, a=(18,18), b=(6,6), t=2; also w=13 (26.69239) and w=25 (26.67767): 684 |
| 686 | 26.70711 (ours) | 686 | w=1 incumbent |

MILP sweep at target L (coarse phase grid, so it undercounts the rigid optimum by ~1): 635 -> 633 (45deg w=13), 632 (w=8,10); 686 -> 684 (45deg w=8,10). All non-45deg angles: <= 626 (635) and <= 677 (686); the grid K^2 is 625 / 676.

## Conclusions
- Every 45deg staircase band with L in [target, ours) is exactly 2 squares short (n-2) for both n; the deficit is flat across the whole interval, so spare L (up to 0.05) does not buy columns. Only the w=1 incumbent reaches n.
- Rational/Pell-type tilt angles (3-4-5, 5-12-13, 8-15-17, 7-24-25, 20-21-29) and free angles 30-42deg lose 7-60 squares versus 45deg: tooth waste does not match the 45deg 1:1 staircase.
- Residual lead: the n-2 configs (L45 ~ 25.657 / 26.657, 0.05 below ours) would need two extra squares from pocket merging/tilted corner pieces; not attempted (free area per tooth gap is far below 1). The MILP is a lattice pool, so it does not rule out unstructured non-lattice packings.
- Caveat: sweep grids are coarse (phase 0.2, offset 0.25, 3 s MILP), no 686 run above target L, no t>3 in the free-L enumeration.

Files: `poc/*.py`, `docs/enum45_*.json`, `docs/enumL_*.txt`, `docs/sweep_*.json`.
