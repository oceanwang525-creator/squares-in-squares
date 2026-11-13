# Notes

## Geometry fix

The previous implementation normalized a pocket by subtracting the minimum
`x` and `y` of its tilted-square polygon bounds. Its upright exit supports were
transformed with the same rotation and reflection but never received that
translation. After a later corner shift, the supports were still displaced by
the negative of the pocket's original origin relative to the tilted squares.

This implementation stores the pocket and supports in one normalized bundle:

```text
low = polygon_bounds(tilted_squares).lower_left
tilted_squares -= low
axis_supports  -= low
```

Every dihedral transform then normalizes the transformed bundle once using only
the transformed tilted-square bounds. The same placement offset is applied to
both arrays.

## Replay

The legacy summaries preserve source count, component id, transform index,
phase index and corner anchor for their highest-ranked cells. The replay
reconstructs those choices against the current incumbent and reevaluates them
at the incumbent side minus a small margin. This tests the support bug without
repeating the full legacy product.

The exact refill uses the candidate set and corner reductions in
`poc/highn-mondrian/poc/fillfast.py`. No area estimate is reported as a packing.

## Exact set-packing correction

The first sparse MILP assembly transposed the conflict-matrix row and column
indices. It produced a false 1014-square result. The corrected implementation
keeps each conflict as one row with two columns. The one-window smoke test and
a corrected 300-window search both remain at 969/971.

## Search negatives

The following compact-motif variants were measured after the support fix:

- a motif pair while keeping all 908 squares fixed: no legal placement;
- only the 202 tilted 908 squares plus exact upright fill: best 955;
- removing only upright conflicts around one motif, 318 tasks: best 965/971;
- AABB motif area widened to 4.0, 11,147 motifs and 33,441 tasks: the first 300
  tasks still peaked at 965, so the run was stopped;
- all 64 dihedral combinations and four extra half-grid phase pairs for
  `908 A x A`: best 955 at every tested phase.

The dual-pocket replay itself is also negative: the best corrected
`972 x 971` cell reaches 965, and a small rotation sweep reaches 966. These are
probes, not submissions.

## Certification gate

If the scan reaches the target count, it writes a candidate CSV and certifies
the side with `poc/baseline`. The result is still not called verified until the
literal and pandas round-trip official runs both accept all 1000 rows.
