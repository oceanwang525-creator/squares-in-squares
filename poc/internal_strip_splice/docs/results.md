# Measured strip-splice results

The donor and comparison geometry came from the tracked incumbent at
`b57e78b89784a87904f0793ec388276741d9f8e8`.
The input SHA-256 was
`1f9c0fd3b7a88e5a0f6c44b567ff2ef354872812c6b0ad8e37807bd1a64e965c`.

| Search | Cases | Near-capacity cases | Strict candidates |
|---|---:|---:|---:|
| Internal slabs of integer width 1–4 | 2,076 | 562 | 0 |
| Fractional slabs on 24 frontier targets | 1,335 | 38 | 0 |
| Fixed-center splits with collision graphs | 432 | — | 0 |
| Adaptive per-square seam assignment | 40 | — | 0 |

The integer scan used donors from 100 through 1,000. The fractional scan
considered cut widths of 0.55–2.35 units on each axis, with at most 12 squares
of difference between the removed area and the target removal count. It
used the frontier gap table to choose target sizes and measured their
actual incumbent bounding boxes from the input coordinates. The graph stage
tested nine splits on each of the 48 closest cut pairs, using a
0.25-second MILP budget for each non-bipartite graph. All 432 tested graphs
were non-bipartite. The straight-cut and fixed-split stages together took
175.846 seconds in one process.

The adaptive stage used the 40 retained closest targets, a 0.6-unit shoulder
around each slab and a 2-second MILP budget per target. All 40 optimizations
reached the integer count optimum in the bounded placement model. Four
targets gained one square and each still fell short by one square.

| Target | Donor | Prior capacity | Adaptive capacity |
|---:|---:|---:|---:|
| 40 | 106 | 38 | 39 |
| 68 | 104 | 66 | 67 |
| 70 | 152 | 68 | 69 |
| 86 | 127 | 84 | 85 |

The adaptive stage took 7.010 seconds. The 3-by-3 grid control contracted to
four squares in a side of 2, and the seam graph solver matched brute force
on 49 random graphs of up to seven vertices.

No proposal had both enough squares and a strict float-side reduction, so
the candidate certifiers were never called and there is no candidate CSV.
The capacity bounds apply to the tested translation options and cut
locations. They do not bound a seam search with independently rotated or
continuously displaced squares. The measurements are in
[scan.json](../data/scan.json) and [assignment.json](../data/assignment.json).
