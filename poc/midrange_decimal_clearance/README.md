# Decimal clearance sweep for n=100–200

Some certified layouts keep a small separation margin to survive CSV rounding.
Shrink centre coordinates by a few parts in 10^15 while leaving every square
angle unchanged. Try several translations before the pandas round trip to
change where each coordinate falls relative to a floating-point boundary.
The sweep does not change the layout geometry at search scale.

The official evaluator must accept each retained layout and report a strictly
smaller side in both the original decimal CSV and the pandas 3.1.0 CSV. The
validation script also constructs the full portfolio and verifies that each
unchanged puzzle has identical literal and pandas inputs, and every changed
puzzle still passes both checks after the full-file round trip.

Only the winning rows are included here. The upstream `merged.csv` is the input
and was not edited. Nothing was submitted to Kaggle.

See [measured results](docs/results.md) and [replacement rows](data/improved_rows.csv).

```sh
incumbent=poc/pell_corrugation/data/border-scan/merged.csv
proof=poc/uniform_shrink/data/kat-union-proof
python poc/midrange_decimal_clearance/poc/search.py "$incumbent" "$proof" /tmp/clearance
python poc/midrange_hull_rotation/poc/certify.py "$incumbent" "$proof" \
  /tmp/clearance-proof /tmp/clearance/clearance_candidates.csv
```

Requires Python 3.12, NumPy, pandas 3.1.0, and the dependencies of the vendored
official evaluator in `poc/baseline/poc`.
