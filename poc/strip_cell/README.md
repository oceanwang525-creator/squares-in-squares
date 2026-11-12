# Periodic strip-cell capacity (negative)

## Idea

The corner arm of the large-`n` packings holds 167 squares in
`7.664 x 23.664` (linear density `7.057`/unit, area density `0.921`). A periodic
free-angle cell of `m` squares in width `W` and period `P` would tell us whether
a denser band exists. Optimize fixed-`W` SAT-penalty Adam cells with periodic
images at `x +/- P`, hard walls at `y=0,W`, and warm-started shrink stages.

## Measured result: negative

At the arm's linear density `m/P = 7.057`:

- `m=14, P=1.9838`: no feasible cell at any `W >= 8.0`, for 12 independent
  seeds and 45 shrink stages each.
- The obstruction is discrete: a pattern with 2 upright columns per period
  needs period `>= 2.0`, because a unit square's horizontal extent is at least
  `1` in every orientation. `P=1.9838` cannot contain two columns, and a single
  column would need `W=14`.
- Every failed cell leaves a residual pair penetration when the optimizer
  untilts the seed; tilting cannot reduce a square's minimum horizontal extent.

## Interpretation

The arm's `7.057 > 7.0` density is a **finite-end effect**: the plain upright
7-column strip gives exactly `7.0`, and the extra ~6 squares come from the
interlocks with the corner blocks at the arm ends. A periodic cell is strictly
less capable than the finite arm, so this route cannot reproduce the frontier
gap. This also explains why the fixed-wall arm compression is jammed.

Script: `poc/strip_cell.py`. Run data: 12 seeds, 14-square cells, 32-thread
process pool. No CSV written; `merged.csv` unchanged.
