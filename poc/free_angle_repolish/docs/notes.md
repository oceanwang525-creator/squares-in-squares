# Method notes

## LP replication

The solver copies the fixed-topology LP from `poc/gallery_repair/poc/repair_gallery.py`:

- pair set: `cKDTree(centers).query_pairs(1.6)`;
- one active axis per pair, chosen by `argmax(|projection| - radius)` over the four edge normals, with the sign of the projection;
- constraint `a . (x_j - x_i) >= radius + 1e-9`;
- two wall extents per center coordinate, `x >= ext` and `x <= L - ext`, where `ext = 0.5*(|cos|+|sin|)`;
- objective `min L`; bounds on every center are the merged coordinate +/-0.03, and `L >= 0`;
- `scipy.optimize.linprog`, HiGHS, tolerances 1e-9, 30 s limit.

Unlike the repair, the pair set and active-axis sign pattern are rebuilt at every trial angle, and the wall extents and pair radii use the trial angles.

## Coordinate descent

For each target:

1. Solve the baseline LP at the merged coordinates and angles.
2. For each pass, visit the tilted squares in a fixed-seed random order.
3. For each square, try the eight ladder values +/-0.002, +/-0.005, +/-0.01, +/-0.02 rad.
4. For each trial, copy the angle array, change one angle, rebuild the LP, and solve.
5. Keep the best trial only if its LP side is lower than the current side by more than 1e-9.
6. Stop early when a pass makes no improvement. Otherwise run at most 3 passes.

The center bounds stay anchored to the merged coordinates for the whole descent, so the 0.03 trust region is never re-centered.

## Emission and certification

`Decimal.from_float` captures the exact binary float, then `mpmath` at 75 dps multiplies centers by the same `1.0000000001` safety factor used by `repair_gallery.py` and prints 65 significant digits. Angles are printed from `Decimal.from_float` without dilation. Each target is written as a one-puzzle CSV with ids `0..n-1`.

Certification is the unchanged official `read_submission` + `certify` on the literal CSV and on `pandas.read_csv(path).to_csv(index=False)`. The incumbent is the matching `L` string in `merged-literal-official.json`. A win requires both candidate strings to be strictly smaller.

## Observed run

Command:

```sh
python3 poc/free_angle_repolish/poc/free_angle_repolish.py --targets 490,535,631,293,932 --passes 3
```

Wall time about 12.5 minutes on CPU. n=490 and n=535 each accepted exactly one +0.005 rad move in pass 1; pass 2 made no further improvement. n=293, n=631 and n=932 accepted no angle move and stopped after pass 1.

The LP side for n=490 moved from 22.63709157595389 to 22.63706688773023. The emitted, certified L moved from 22.637091576117623 to 22.637066887893938, a reduction of 2.4688e-5. n=535 behaved the same way, reducing by 2.4692e-5.

## Reproducibility

The random order uses `numpy.random.default_rng(0)` independently for each target. Re-running should reproduce the same accepted moves. Outputs are deterministic except for the normal floating-point variability of HiGHS.
