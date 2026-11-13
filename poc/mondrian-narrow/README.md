# Mondrian narrowing before composition

## Idea

The public SQUISH certificates identify the operation missing from
`poc/highn-mondrian`: narrow each rectangular sub-packing *inside its own
rectangle* before placing two pieces in opposite corners. The initial local
scan composed the crops at their original bounding boxes, so it could not use
a pair whose boxes fit only after one or both pieces were repacked.

## Method

For a target `n`, the scan:

1. enumerates the highest-gain bottom-left and top-right corner crops;
2. ranks crop pairs by `(gain_A + gain_B)` and the amount of rectangle
   reduction needed to make both pairs of dimensions fit;
3. holds each crop inside a shrinking rectangular wall and applies a
   free-angle SLP continuation, followed by a margin-tight fixed-angle repair;
4. places the narrowed pieces in opposite corners and uses the exact
   staircase-candidate MIS in `fillfast` to refill all upright squares;
5. emits a group only when the unchanged official metric is strictly below
   the incumbent in both literal and pandas transport.

The narrowing LP freezes a zero-width existing contact to zero instead of
requiring positive clearance. This matters because the saturated crops contain
many simultaneous contacts; a positive-margin feasibility LP rejects even
their unchanged geometry.

## Result

Limited negative. A corrected `n=908` smoke scan with donors
`791,908,971,972`, crops up to 32, and one top pair found one candidate pair,
but no feasible narrowing variant for either crop. This isolates the failure
at the highest-ranked crop's wall/slack condition rather than at refill.

The corrected scanner now ranks by projective post-shrink capacity and
enumerates shrink splits crossing integer lattice boundaries. The negative is
not global: only one crop pair and one shrink target were tested. The next
useful scan is multiple smaller seam pieces, not a saturated 200-square pocket.

## Files

- `poc/narrowscan.py` - crop-pair ranking, wall SLP, exact refill and official
  double-transport gate.
- `docs/NOTES.md` - implementation details and measured results.
- `docs/LITERATURE.md` - paper-derived structural directions and measured
  consequences.
- `data/smoke-908.json` - the bounded corrected-scanner negative above.
