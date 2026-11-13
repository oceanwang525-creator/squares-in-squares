# High-n Mondrian corner composition

## Idea

SQUISH's Mondrian construction cuts known packings into rectangular
sub-packings and places two pieces in opposite corners of a larger box. Whole
upright rows and columns fill the remainder; a seam repair then recovers the
small gap left by the composition.

Earlier local work tested one donor crop in one corner plus an L-shaped grid
(`poc/donor_slices`) and a DP over already-known clean-cut gadgets
(`poc/rectangle_gadgets`). This POC tests the missing second corner: two
independent corner crops can contribute a positive area gain over the same
upright grid.

## Method

For target side `L`, let `K = floor(L)`. A crop with `c` squares in a tight
`w by h` rectangle replaces all unit-grid cells touched by that rectangle.
Its net gain is

```text
g = c - touched_grid_cells
```

Two crops in opposite corners are bounding-box disjoint. Their total capacity
is therefore `K^2 + g_A + g_B`; no pairwise rectangle intersection needs to be
enumerated. The search:

1. cuts every tracked packing at each corner on integer widths and heights;
2. deduplicates crops by tight dimensions and capacity;
3. searches opposite-corner pairs at a side strictly below the incumbent,
   retaining the top 3000 bottom crops and top 3000 top crops by gain;
4. fills every remaining upright-grid cell;
5. keeps a composed packing only if it reaches the target count;
6. requires the unchanged official metric to pass both the literal CSV and a
   pandas `read_csv` / `to_csv` round trip before any row is emitted.

This is a bounded top-gain scan, not an exhaustive pair enumeration. If the
composition has surplus capacity, only the required number of grid squares is
emitted.

## Targets

The first sweep targets the high-`n` `k^2 + D` boundary where the current
incumbent uses one or two stock pockets:

```text
D = 8:  908, 969
D = 10: 686, 910, 971
D = 11: 687, 911, 972
D = 12: 973
```

These require only 8-12 squares above the `K by K` upright grid, exactly the
regime where two accumulating corner gains can change the topology.

## Result

Negative. Ten high-`n` targets were scanned and none reached its required
capacity. The best two-corner totals were:

| n | upright grid | required gain | best gain | capacity | short |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 686 | 676 | 10 | 5 | 681 | 5 |
| 687 | 676 | 11 | 6 | 682 | 5 |
| 908 | 900 | 8 | 3 | 903 | 5 |
| 910 | 900 | 10 | 4 | 904 | 6 |
| 911 | 900 | 11 | 5 | 905 | 6 |
| 969 | 961 | 8 | 3 | 964 | 5 |
| 970 | 961 | 9 | 3 | 964 | 6 |
| 971 | 961 | 10 | 4 | 965 | 6 |
| 972 | 961 | 11 | 5 | 966 | 6 |
| 973 | 961 | 12 | 5 | 966 | 7 |

The static two-corner family is 5-7 squares short throughout this frontier.
SQUISH's published certificates identify the missing operation: narrow each
piece inside its own rectangle before composition, then repair the new seam.
The raw bounded scan is in `data/mondrian-summary.json`.
