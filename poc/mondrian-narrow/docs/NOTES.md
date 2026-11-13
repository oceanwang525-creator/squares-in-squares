# Notes

## Geometry

Each crop is normalized to its tight axis-aligned bounding box. A wall solve
searches for an overlap-free packing of its original squares in a smaller
rectangle `(0, W) x (0, H)`. Existing zero-gap contacts use a zero right-hand
side, so the LP is feasible on a saturated incumbent; only genuine
penetrations must acquire clearance.

The free-angle rows linearize each square's four corners against the currently
nearest separating face. The fixed-angle repair is an exact LP in centres and
is used after every accepted free-angle proposal. Pair and wall feasibility
are checked directly before the piece is accepted.

## Composition

The bottom piece is placed at `(0,0)` and the top piece at
`(side-W, side-H)`. Their rectangles are disjoint when both coordinate sums
fit. The remaining upright squares are selected by `fillfast`'s exact
staircase-candidate MIS; a heuristic capacity estimate is never treated as a
win.

## Literature

The public `itsnaka/squish-certs` October 2026 submissions report 24 new
packings between `n=131` and `n=307`. Their pipeline is:

1. cut a known packing into rectangular sub-packings;
2. narrow each piece in its own rectangle;
3. place two pieces in opposite corners;
4. fill whole rows and columns;
5. repair the seam;
6. run nearby search and deep squeeze.

Eight of their nine October 11 compositions use pieces narrowed before
composition. The final squeeze moves at most `7e-7`, so the gain is primarily
topological rather than a long numerical polish.

## Corrected feasibility screen

The initial top-crop ranking used raw corner gain. That can select a crop whose
rectangle needs an impossible amount of wall motion before it can contribute.
The corrected scan:

- ranks by projected post-shrink capacity;
- includes shrink splits that cross integer lattice boundaries;
- enforces post-shrink area capacity;
- removes the old `current_gain >= need` rejection;
- supports explicit donor sets.

Smoke command:

```text
python poc/mondrian-narrow/poc/narrowscan.py \
  --targets 908 --donors 791,908,971,972 \
  --max-crop 32 --top-pairs 1 --stages 8 --max-solves 20
```

Measured result: one candidate pair and zero narrowing variants. The failure
was inside the rectangular wall solve, before exact upright refill. It is
logged as a limited negative, not a claim that all 908 crops are rigid.

## Literature

See `docs/LITERATURE.md` for the actionable Arslanov staircase-block,
Huang-Korf coordinate-first, and Iori et al. exact-model notes.
