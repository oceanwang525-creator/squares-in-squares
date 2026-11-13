# n=973 dual-pocket transplant

## Idea

The certified `n=971` packing has side about `31.66432`, while the tracked
`n=973` packing is `31.69661` and the published frontier is about `31.66664`.
This POC keeps the two tilted `n=971` pockets fixed, inserts a compact source
motif, and refills the upright region with exact MIS at the frontier side.

This is a cross-`n` topology transplant rather than local polishing. Reaching
973 at a side below the incumbent would be a material structural win.

## Files

- `poc/scan.py` - focused runner around the exact `core_motif_fill` search.
- `docs/NOTES.md` - settings and measured results.

No `final.csv` is included unless the target packing is certified.
