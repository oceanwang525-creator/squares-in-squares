# k2p10_fill2: fit the last 2 squares into the n-2 45° band configs (n=635)

Result: negative. No 635-square overlap-free layout below our L = 25.7071067812 was found (leader: 25.69874). Estimated gain: 0. `merged.csv` untouched, no CSV written, nothing certified.

Start: 45° staircase band cores holding exactly 633 squares (e.g. w=8, a=(18,18), b=(6,6), t=1) that sit down to L45 = 25.69874 (see `../k2p10_band_search`).

| Approach | Files | Best result |
|---|---|---|
| Pocket merging / insert 2 squares, all x,y,angle free, L=25.705 | `poc/a_pockets.py`, `a_relax.py` | max overlap depth 0.0184, hundreds of overlapping pairs |
| Tilted corner pieces (45°, atan(3/4), jitter) | `poc/a_corner.py` | worse: depth 0.030-0.069 |
| Penalty + hardening + basin hops, 3 cores, Lc 25.70-25.7071 | `poc/b_fill.py`, `b_hop.py` | energy 7.3e-3, depth 0.012, 786 overlapping pairs |

Findings:
- The 633 cores have no pocket for a unit square (best insertion still overlaps by 0.18-0.19); free area is in 0.25-area staircase triangles and wall slivers.
- Relaxation only smears the overlap over many contacts; even at Lc=25.7071, where a zero-overlap 635 exists, it did not recover one. Bisecting L is pointless.
- Untried: basin hopping with block moves, MILP pocket merge, SLP/4-sign LP polish (nothing got near feasible).

Reusable: `poc/a_overlap.py` and `poc/b_overlap.py` (vectorised SAT with KD-tree, analytic gradient; verified against a reference to ~1e-15, incumbent 635 gives zero overlap). Details in `docs/a_summary.md`, `docs/b_summary.md`.
