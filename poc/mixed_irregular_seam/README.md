# Mixed irregular seam

## Idea

The rectangular staircase composition rule is saturated: it requires the
bounding rectangles of successive pockets to be disjoint except for one row or
column. This POC relaxes that rule. Two donor crops may interleave in the same
corner band as long as their actual square polygons do not overlap. The
remaining upright cells are selected by the exact MIS in `fillfast.py`.

The search differs from `highn-mondrian` by allowing the two crop bounding boxes
to overlap and by measuring the occupied integer-grid cells from the actual
transformed polygons. Capacity is therefore not inferred from `(w,h,g)`.

## Method

For a target side `L`:

1. extract the highest-gain corner crops from the incumbent portfolio;
2. normalize and transform each crop with all eight dihedral transforms;
3. place one crop from the lower-left and one from the upper-right on sampled
   phase lattices;
4. accept the pair only if their true square polygons are disjoint;
5. compute the union of integer grid cells touched by the actual crop polygons;
6. prefilter by the exact cap `K^2 + n_a + n_b - |union|`;
7. run the exact upright MIS on surviving pairs;
8. certify complete groups in literal decimal and pandas transports.

## Quick targets

- `367`
- `791`
- `969`
- `974`

## Run

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
$PY poc/mixed_irregular_seam/poc/search.py \
  --submission poc/pell_corrugation/data/border-scan/merged.csv \
  --targets 367,791,969,974 \
  --min-crop 4 --max-crop 18 \
  --keep-crops 1600 --pair-limit 8000 --workers 16 \
  --milp-seconds 3 \
  --summary poc/mixed_irregular_seam/docs/quick.json \
  --final poc/mixed_irregular_seam/final.csv
```

The run is a bounded quick screen. A real result requires a complete strict win
in both certification modes. Otherwise `final.csv` remains header-only.

## Results

Negative at the quick-test boundary.

The first corner-crop pass used the full selected crop, including upright
squares. For `n=367` its largest geometric bound after 1,500 screened pairs was
only `360`, below the required `367`; fixing those upright squares loses more
cells than the donor fragment contributes.

The second pass used the actual pocket geometry from
`poc/pocket_lift_scan2/data/pockets_pareto3.json` and kept only fragments whose
real box fit the target. Twenty high-density single/pair placements were
completed for each of `n=367` and `n=791`:

| n | pairs screened | largest pre-overlap bound | exact-refill candidates | wins |
|---:|---:|---:|---:|---:|
| 367 | 20 | 492 | 0 | 0 |
| 791 | 20 | 969 | 0 | 0 |

The large bounds come from overlapping bounding boxes, but every corresponding
pair failed the actual square-polygon SAT test. No non-overlapping placement
reached the target count; therefore no candidate entered exact MIS and no group
was eligible for certification. `final.csv` remains header-only.

This negative is bounded: it does not prove that arbitrary non-rectangular
seams are impossible. It shows that the natural lower-left / upper-right
placements of high-density donor pockets are either disjoint in their full
boxes or overlap their fixed squares. Unlocking this family would require
cutting the pockets into smaller fragments or allowing continuous movement of
the fixed-pieces themselves, not another placement-phase sweep.

Raw results:

- `docs/quick367.json`
- `docs/quick791.json`
