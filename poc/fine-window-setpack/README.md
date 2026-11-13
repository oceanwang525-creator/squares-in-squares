# Fine-grid window set packing

## Idea

The `n=908` core plus exact upright refill reaches 969 squares in the
`n=971` target box, leaving two squares to find. Earlier exact window search
used 0.5-unit centres and 5-degree angles. The remaining voids are close to one
unit square in area, so that grid can miss a feasible contact.

This POC runs an exact maximum independent set solve inside windows centred on
the largest voids, using 0.1-unit centre grids and 0.5-degree angle samples.
Squares outside each window stay fixed. A window may replace several seed
squares, which is the topology-changing move the previous insertion search
lacked.

## Files

- `poc/search.py` - fine candidate generation and exact local MIS.
- `docs/NOTES.md` - commands and measured results.

No `final.csv` is included unless the target packing is certified.
