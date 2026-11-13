# Literature scan

## SQUISH certificates

`itsnaka/squish-certs` is the most directly useful public source. The October
2026 submission notes give a concrete composition pipeline:

1. cut known packings into rectangular sub-packings;
2. narrow a piece inside its own rectangle;
3. place two pieces in opposite corners;
4. fill the remaining whole rows and columns;
5. repair the seam;
6. run nearby search and deep squeeze.

The published certificates also bound the useful scale. The October 11
compositions narrow pieces with 37, 42, 47, 52, 57, 58, 63, and 68 squares.
They do not narrow a 200-square saturated pocket. This is relevant to the
high-`n` frontier: the n=908 dual-pocket core has 100-square tilted pockets,
and the exact local scan already found those boxes at their contact jam.

## Anchored square and rectangle packings

Balas, Dumitrescu, and Toth, arXiv:1603.00060, reduce anchored rectangle and
square packing to maximum-weight independent set and give a PTAS. The useful
algorithmic point here is structural rather than a new packing: after the
moving pockets are fixed, the residual upright cells should be solved as an
exact MIS, which is what `fillfast` already does. A fractional LP count alone
is not a candidate certificate.

## Asymptotic unit-square constructions

Bui, arXiv:2508.04603, and McClenagan, arXiv:2602.01484, construct large-square
packings with sublinear waste using grid-plus-boundary families. Their
asymptotic scaling does not directly improve a fixed `n <= 1000`, but they
support the same design principle visible in the incumbent: an upright bulk
plus a lower-dimensional tilted boundary or pocket. The practical search space
is therefore the interface topology, not another bulk-wide free-angle polish.

## Iterated local search

Ye, Huang, and Lu, arXiv:1306.0694, combine continuous local optimization with
tabu search and iterated perturbation for circle packing. The transferable
point is to treat a composed layout as a local minimum and perturb a whole
piece or seam, rather than applying a smaller trust radius to every square.
The local experiments already show that direct compression of the saturated
high-`n` rows is jammed at `1e-5`, so the perturbation must change piece
membership or the seam before numerical repair.

## Rectangular staircase blocks

Arslanov, "On the packing of unit squares into a square," Electronic Journal
of Combinatorics 28(4), 2021, DOI `10.37236/8586`, gives a parametric family of
squeezable packings inside long rectangles. The directly reusable pieces are:

```text
26 squares in (4,8)
38 squares in (5,9)
43 squares in (5,10)
58 squares in (6,11)
64 squares in (6,12)
```

The general lemma packs `4k^2 + 6k - 2` squares in `(2k, 2k + 4)`. Composing
four corner blocks proves `s(k^2-k) < k`, but the present incumbents at those
counts are already `0.067` to `0.084` below the integer side, so the theorem's
completed square is not a direct score win. The useful part is the topology:
two interleaved staircases with orientations `(1,0)` and
`((4k^2-1)/(4k^2+1), 4k/(4k^2+1))` leave a controlled seam. Treat these
rectangles as donors or as a source of seam gadgets, not as a replacement for
the current high-`n` rows.

## Coordinate-first exact placement

Huang and Korf, "Optimal Rectangle Packing: An Absolute Placement Approach,"
JAIR 47 (2013), DOI `10.1613/jair.3735`, first fixes all `x` coordinates,
converts the instance to a perfect packing by adding rectangles, and then
branches only on `y` placements. This is a better fit for the remaining
upright refill than a generic two-dimensional MILP: the tilted pocket fixes
the horizontal obstacle profile, while reachable upright `x` coordinates are
finite data-driven events. The existing `fillfast` candidate set is a useful
special case, but not the full coordinate-first search.

Iori et al., "Exact solution techniques for two-dimensional cutting and
packing," EJOR 289 (2021), DOI `10.1016/j.ejor.2020.06.050`, separates position
models from interval and covering models. For this POC the appropriate exact
back end is a position model over the finite seam coordinates, not a
full-grid covering model.

## Consequence for this POC

The next measurement should first establish whether a crop with realistic area
slack can be narrowed by even `1e-3`. If the 200-square high-`n` pockets fail
that screen, continuing the same narrowing path is lower value than searching
smaller Mondrian pieces and using exact refill at the resulting seam.

The corrected scanner is a limited `n=908` smoke test: donors
`791,908,971,972`, crops up to 32, and one top pair. It found one candidate
pair, but neither crop admitted a feasible narrowing variant. This does not
prove the saturated pocket is rigid; the failed crop was only the highest
ranked one. It does show that choosing candidates by raw gain overestimates
the useful pieces. The next scan should rank by post-shrink capacity and
enumerate multiple crop sizes on both sides of the seam, as the corrected
implementation now does.
