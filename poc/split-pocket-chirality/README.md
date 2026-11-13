# Split-pocket corner chirality

## Idea

The `n=908` packing contains two tilted pockets of 101 squares. The completed
`n=908` core plus exact upright refill reaches 969 squares in the `n=971` box;
the two long pockets leave a fragmented central boundary that is short by two
squares.

This POC splits each 101-square pocket into two 50/51-square rigid halves. Four
roughly 9-by-9 pockets can then be assigned independently to the four corners,
with all eight dihedral transforms and half-unit phases. The residual upright
region is refilled by the unchanged exact MIS implementation.

## Files

- `poc/scan.py` - pocket splitting, four-corner placement search and exact fill.
- `docs/NOTES.md` - run settings and measured results.

No `final.csv` is included unless the target packing is certified.
