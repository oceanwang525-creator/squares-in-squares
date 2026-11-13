# Research notes

## SQUISH

The public certificate repository is `itsnaka/squish-certs`. Its October
2026 submissions report 24 new packings between `n=131` and `n=307`. The
README for the October 9 submission describes the construction as:

> known packings are cut into rectangular pieces (sub-packings), two pieces
> are placed in opposite corners of a box with whole rows and columns of unit
> squares filling the rest, and where the composed layout sits just above the
> record, a search that repairs its seams can bring it below.

The October 11 update adds that all but one new composition uses pieces
**narrowed in their own rectangle before composing**. This is the concrete
missing step in the current `highn-mondrian` scan.

The published finishing pipeline is:

1. cut a known packing into rectangular sub-packings;
2. narrow each piece inside its own rectangle;
3. place two pieces in opposite corners;
4. fill remaining rows and columns;
5. repair the seam;
6. run nearby search;
7. run a deep squeeze.

The reported squeeze moves none of the October 11 layouts by more than
`7e-7`, so the important gain comes before the squeeze.

## Gensane and Ryckelynck

The 2005 "Packing squares into a square" paper combines randomized physical
relaxation with a final exact packing step. Its useful high-level sequence is
random walking, inflation, collective perturbation, and billiard-style
repair. The repo already has extensive single-square stochastic search; the
missing part of this paper is collective perturbation of a whole sub-block.

## Soft-to-rigid

arXiv `2610.14463` parameterizes a packing as a soft overlap problem and
gradually increases the hard-core weight before a rigid feasibility phase.
The team has already tested this family in `poc/soft_to_rigid`; it is logged
negative for `n=51,68,86`, so it is not a priority here.

## Practical takeaway

The next experiment should not enumerate more static corner stencils. It
should test whether existing crops can be narrowed by a collective
transformation while retaining all pair separations, then use the established
`tight_lp` and `free_slp` polishers on the joined layout. The first test set
should be `n=908,910,911,969..973`, because the current scan's shortfall is
only 5-7 squares and the frontier pockets already expose the relevant
chirality and channel phases.
