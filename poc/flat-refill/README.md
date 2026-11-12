# Flat-row exact refill

## Idea

Five high-`n` flat rows (`627, 678, 843, 902, 963`) have 75-95 tilted
squares. Replacing their upright squares with `fillfast` exact MIS finds
`n + 1` squares at the current side. The natural question is whether the
surplus square lets the same tilted pocket shrink below the current side.

## Method

For every target:

1. keep all tilted squares from the current certified packing;
2. try a ladder of tiny box reductions;
3. re-fill upright squares with `fillfast` exact MIS;
4. convert fillfast degrees back to radians;
5. try a minimal uniform dilation ladder;
6. require literal and pandas-roundtrip official `L` to be strictly lower
   than the incumbent.

Run:

```sh
python poc/flat-refill/poc/scan.py \
  --submission "/abs/path/to/submission.csv" \
  --fillfast-dir "/abs/path/to/refs/evand/search/packer" \
  --summary /tmp/flat-refill.json
```

## Result

The full result is in `docs/NOTES.md`. The family is negative: the `n + 1`
capacity comes from contact rounding, and every certifiable candidate after
dilation is at or above the current side.
