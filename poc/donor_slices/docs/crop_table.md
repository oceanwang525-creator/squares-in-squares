# Crop table

The cut is an integer rectangle `[x0, x1] x [y0, y1]`. The tight box is measured
from the vertices of the kept squares. `G` is the maximal upright grid in the
remaining L-shape. `N = c + G` is the composite capacity. The exact side is
taken from all vertices; the placement side includes the 1e-7 clearance.

| Target | Role | Donor | Cut | `c` | Tight `w x h` | Orient | Grid right | Grid top | `G` | `N` | Exact side |
|---:|---|---:|---|---:|---|---|---|---:|---:|---:|---:|
| 645 | below incumbent | 697 | `[0,26] x [1,27]` | 644 | `25.869769561 x 25.869769561` | identity | `0 x 25` | `25 x 0` | 0 | 644 | `25.869769661` |
| 645 | capacity match | 698 | `[0,26] x [0,26]` | 645 | `25.881109226 x 25.881109226` | identity | `0 x 25` | `25 x 0` | 0 | 645 | `25.881109326` |
| 925 | below incumbent | 986 | `[0,31] x [0,31]` | 923 | `30.859361960 x 30.859361960` | identity | `0 x 30` | `30 x 0` | 0 | 923 | `30.859362060` |
| 925 | capacity match | 988 | `[0,31] x [0,31]` | 925 | `30.875854187 x 30.875854187` | identity | `0 x 30` | `30 x 0` | 0 | 925 | `30.875854287` |
| 987 | below incumbent | 999 | `[0,22] x [0,22]` | 484 | `22 x 22` | identity | `9 x 31` | `22 x 9` | 477 | 961 | `31.000000100` |
| 987 | capacity match | 988 | `[0,31] x [0,32]` | 956 | `30.875854187 x 31.875854188` | identity | `1 x 31` | `30 x 0` | 31 | 987 | `31.875854288` |

The `n=987` below-incumbent row is the only one with a non-empty grid. Its
capacity is 961, 26 short of the target. The capacity match uses a taller crop
and a one-column grid; its capacity is exactly 987, but its side is above the
incumbent.

The emitted CSV sides are measured from the written rows, not from the placement
table. They are `25.881109226069280`, `30.875854187299760` and
`31.875854287299759` for `n=645`, `925` and `987`. The small differences from
the placement side are the 1e-7 crop-grid clearance.
