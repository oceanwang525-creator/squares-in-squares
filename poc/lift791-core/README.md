# 791-core lift

## Idea

The certified `n=791` and `n=908` sides differ by almost exactly two units
(`2.000000004`), and 387 squares coincide exactly. This suggests a reusable
horizontal/vertical border-growth construction rather than independent
packings.

This POC uses all 791 squares as a fixed rigid core, then runs the exact
upright MIS refill at a target side. It next removes one or two core squares
and refills again, allowing a small local topology change without discarding
the recurring core.

Files:

- `poc/release_fill.py` - exact core-square release and refill search.
- `poc/transform_fill.py` - dihedral and translation-phase scan with wall checks.
- `data/results.json` - compact measured counts for all runs.

## Method

Every candidate is regenerated with the exact `fillfast` conflict graph and
MILP-kernel refill. A candidate is promoted only when it reaches the target
count, has no overlaps, is not a grid fallback, and passes both official
metric transports against the current incumbent.

## Result

Negative. The fixed `791` core reaches `907/908` and `968/971`; single-square
release, valid dihedral/translation phases, and compact motif replacement do
not improve those counts. Full measurements are in
[docs/NOTES.md](docs/NOTES.md).

No `final.csv` is included because no target count or strict incumbent win was
certified.
