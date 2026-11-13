# n=911 dual-pocket transplant

## Idea

The tracked `n=911` side is about `30.69562`, while the published frontier is
about `30.66091`. The certified `n=908` row is already near `30.63407` and has
two 101-square tilted pockets.

This POC fixes those two pockets inside the frontier box, inserts compact
motifs extracted from nearby solved rows, and refills the upright region with
exact MIS. The target is a material cross-`n` topology change, not local
polishing.

## Files

- `poc/scan.py` - focused runner around `core_motif_fill`.
- `docs/NOTES.md` - settings and results.

No `final.csv` is included unless the target packing is certified.
