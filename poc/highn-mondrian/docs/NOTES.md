# Notes

This POC is the high-`n` continuation of the classical Mondrian composition
idea used by SQUISH. The key capacity identity for two bounding-box-disjoint
corner crops and an upright grid of side `K = floor(L)` is:

```text
capacity = K^2 + (c_A - touched_cells_A) + (c_B - touched_cells_B)
```

The identity is exact when the two corner rectangles are disjoint and the grid
is anchored at the lower-left corner. A grid square intersecting either crop
bounding box is removed; all remaining grid squares are valid because the crop
squares are contained in those boxes.

The first sweep deliberately starts from existing teammate packings. It does
not harvest external submissions and does not modify `merged.csv`.

## First scan

The scan used all incumbent groups and extracted corner crops with side bounds
4 through 20. It retained the 3000 highest-gain crops for each vertical half
before pairing, so this is a bounded scan rather than a proof over every crop
pair.

Measured capacities:

```text
n     grid   need   gain   total   short
686    676     10      5     681      5
687    676     11      6     682      5
908    900      8      3     903      5
910    900     10      4     904      6
911    900     11      5     905      6
969    961      8      3     964      5
970    961      9      3     964      6
971    961     10      4     965      6
972    961     11      5     966      6
973    961     12      5     966      7
```

No candidate was emitted: every target fell below its capacity threshold.
Static composition is therefore not the promising part of Mondrian at this
frontier.

## Paper and certificate check

The public SQUISH certificate repository reports 24 small-`n` Mondrian com-
positions in its October 2026 submissions. Its READMEs describe two pieces
placed in opposite corners with whole rows and columns of unit squares filling
the rest. The 2026-10-11 update says all but one of its nine new compositions
uses pieces **narrowed in their own rectangle before composing**. It then
applies `seam repair`, `nearby search`, and `deep squeeze`.

The repository exposes exact certificates, not the search implementation.
For this POC the useful implication is concrete: a static crop footprint is
not the right invariant. The piece must first be searched inside a narrower
rectangle, then the joined layout gets another search.
