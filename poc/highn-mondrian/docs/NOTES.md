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

## Exact refill

`poc/seamscan.py` replaces the bounding-box grid count with `fillfast`'s
staircase candidates and exact upright MIS. Results:

```text
target  scope                              capacity  need  short
971     top 200 pairs, original orientation       965   971      6
971     top 20 pairs, 8 transforms, 0/.5 phase    965   971      6
908     top 20 pairs, 8 transforms, 0/.5 phase    903   908      5
```

The `971` exact MIS selected 561 simplicial lattice candidates and left no
MILP kernel. Its result therefore equals the bounding-box estimate rather
than merely timing out. The transformed scans use the 32 highest-bound
variants per pair and are bounded, not exhaustive.

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
