# Parametric frontier continuation

## Idea

For each fixed family `D = n - K^2`, fit one rational-slope band template
across adjacent `K` instead of optimizing every row independently.  The shared
parameters are the slope `p/q` and band width `W`; the end staircases `T1/T2`
may adapt to `K`.  This asks whether a single reusable construction improves
multiple family members.

The first probe is analytic and deliberately cheap.  For a rational-slope
single-band template, count exact upright lattice cells in the two staircase
regions and the tilted unit-square chain in the band.  A tuple is retained only
if it satisfies at least two adjacent `K` at a side strictly below the current
incumbent.  Capacity is a screening result, not a certified win.

## Run

```sh
python3 poc/parametric_frontier_continuation/poc/probe.py
```

The output is written to `docs/probe.json`.

## Result

**Negative.** No shared `(p,q,W)` tuple reached even one of the twelve tested
high-families at a side below its incumbent.  The closest screen was the
`(1,1,3)` 45-degree template:

| D | K range | best count deficit per row |
|---|---:|---:|
| 7 | 24-29 | 3, 2, 2, 2, 2, 2 |
| 7 | 30-31 | 7, 7 |
| 8 | 24-29 | 4, 3, 3, 3, 3, 3 |
| 8 | 30-31 | 2, 2 |

The apparent two-square near misses for `(1,1,3)` do not survive geometry.
Expanding that template to actual centers at `n=791` produced 98 overlapping
pairs between the upright staircase cells and the three tilted rows, before
attempting certification.  The capacity formula is only a screen; it is not a
valid non-overlap construction.

The `D=4` pair also failed the shared-family screen: the best counts were 324
for `n=328` (deficit 4) and 360 for `n=365` (deficit 5), with different
parameters.  There is no common tuple to lift.

`final.csv` is header-only because no valid strict win was found.

