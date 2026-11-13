# Notes

## Geometry

Pockets are extracted with `corner-pocket-chirality`'s component and support
logic. The pocket and its upright exit supports are normalized as one rigid
bundle, so every rotation, reflection, translation and corner placement applies
to both arrays.

The left pocket is anchored at the lower-left corner and the right pocket at
the upper-right corner. The phase scan is a Cartesian product, not a shared
phase: each pocket independently receives `(0, 0)`, `(0, 0.5)`, `(0.5, 0)` or
`(0.5, 0.5)`.

## Refill

Every non-overlapping pair is passed to the unchanged `fillfast` candidate
generator and exact MIS reduction/MILP kernel from `highn-mondrian`. The
default scan does not add source support points to the candidate graph. This
keeps the graph small and matches the older `phase_scan` result basin.
`--support-candidates` is available as a separate mode.

The refinement mode can evaluate only geometry/phase records from a prior
summary. It also accepts upright lattice offsets. This was used to add the
default corner lattices plus all 16 quarter phases on the top `973 x 973`
poses.

## Measured commands

Representative scans:

```sh
python poc/cross-pocket-phase/poc/scan.py \
  --target 971 --first-source 971 --first-component 0 \
  --second-source 908 --second-component 1 \
  --phase-step 0.5 --jobs 10 --reach 8 --milp-limit 10 \
  --summary poc/cross-pocket-phase/data/scan-971a-908b-half.json \
  --best-out poc/cross-pocket-phase/data/probe-971a-908b.csv

python poc/cross-pocket-phase/poc/scan.py \
  --target 972 --first-source 973 --first-component 0 \
  --second-source 973 --second-component 1 \
  --phase-step 0.5 --jobs 10 --reach 8 --milp-limit 10 \
  --summary poc/cross-pocket-phase/data/scan-973a-973b-target972-half.json \
  --best-out poc/cross-pocket-phase/data/probe-973a-973b-target972.csv
```

The `791` run uses `908:0 x 908:0` at the `791` incumbent side. The complete
scan produced `780/791`; the two source sides differ by about 2, but the same
two-pocket template does not survive the smaller box.

## Strong negatives

- The `971` independent phase scans all stop at `959/971`, including both
  `908` components and eight dihedral transforms per side.
- `908 x 908` at target 971 remains `955/971`, confirming the earlier
  shared-phase result.
- Mixing a larger `973` pocket with `971` reaches `964/971`; adding tilted
  squares alone does not recover the seven-square gap.
- Both `973 x 973` and `972 x 972` reach `965/972`.
- Sixteen additional upright lattice offsets do not improve that `965/972`
  peak.
- Keeping the complete upright skirt from `908` while inserting the two
  `971` pockets leaves too many locked contacts; exact refill probes only
  reach the low 900s. This is a boundary on the core-replacement idea.

## Certification

Only a candidate that reaches the target square count is written as a packing.
It is still not a win until a full 1000-group union passes the official metric
both literally and after pandas `read_csv` / `to_csv`.
