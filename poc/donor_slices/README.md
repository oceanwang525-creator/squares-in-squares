# Donor slices

Cut an axis-aligned crop out of a larger certified packing, place it in a corner
of the target container, and fill the remaining L-shape with an upright unit
grid. The idea is measured, not assumed: at every side below the incumbent the
crop-plus-grid capacity is below the target, and every emitted target-`n`
packing is worse than the incumbent. Estimated gain is zero.

The portfolio is `poc/pell_corrugation/data/border-scan/merged.csv` (1000
groups, 500500 rows). The incumbent sides are the literal strings in
`merged-literal-official.json`. Certification uses the unchanged
`sys.path.insert(0, 'poc/baseline/poc/src')` metric with `from squares.core
import read_submission, certify`.

## Construction

`poc/crop.py` enumerates axis-aligned crop rectangles on an integer cut grid.
A square is kept only when its whole vertex bounding box lies inside the cut
rectangle. The emitted crop is the tight fractional bounding box of the kept
squares, so mixed-angle donors are measured from their actual vertices. The
crop is reported with its capacity `c`, exact width `w`, and exact height `h`.
The eight dihedral orientations are exposed; capacity and side depend only on
the two dimension orders, so the search keeps identity or a quarter turn.

`poc/composite.py` places the crop in a corner and fills the two remaining
strips with the maximal upright grid. For container side `L`:

```
G(L, w, h) = floor(L - w) * floor(L) + floor(w) * floor(L - h)
N(L)       = c + G(L, w, h)
```

The exact container side is taken from all crop and grid vertices, never from a
nominal box. A 1e-7 clearance separates the crop from the grid. The search is
deterministic: donor `n'` runs over `target + 1 .. 1000`, cut widths and heights
run over `ceil(sqrt(target)) - 10 .. ceil(sqrt(target))`, and orientations are
deduplicated by bounding-box dimensions.

For each target the search records two things:

- the maximum `N(L)` with exact side strictly below the incumbent;
- the minimum exact side with a capacity-matched composite `N = target`.

`poc/run.py` writes the capacity-matched composite to `data/`, certifies the
literal CSV and the `pandas.read_csv(...).to_csv(index=False)` copy, and records
the measured strings in `data/results.json`.

## Capacity arithmetic

| `n` | incumbent `L` | max `N` below incumbent | shortfall | best capacity match `L` |
|---:|---:|---:|---:|---:|
| 645 | `25.87031812251975364740446552983030849778887781162520` | 644 | 1 | `25.88110922606928005961566084581980832075793593806084` |
| 925 | `30.86321837316950898175795253708386667312879581004381` | 923 | 2 | `30.87585418729976029554514981301238663036201614886522` |
| 987 | `31.86630526508366147711534673847566523363639134913683` | 961 | 26 | `31.87585428729975900403701846625347367262293118983507` |

The prior donor-slice capacity arithmetic recorded 640, 920 and 961 below the
incumbent (shortfalls 5, 5, 26). This broader enumeration finds 644, 923 and
961, so the measured shortfalls are 1, 2 and 26. The deviation is +4 at
`n=645` and +3 at `n=925`; `n=987` is unchanged. The negative conclusion is the
same on all three targets.

### Why grid quantization consumes the margin

At `n=645`, the best crop below the incumbent is donor 697, cut
`[0,26] x [1,27]`, with 644 squares and tight box `25.869769560605`. The
incumbent side is `25.870318123`, leaving only `0.000548562` of room. Both
`floor(L - w)` and `floor(L - h)` are zero, so the L-shape grid contributes
nothing. The next crop with 645 squares has tight box `25.881109226`, already
`0.010791104` above the incumbent. The integer cut cannot add the missing
square without crossing the incumbent side.

At `n=925`, the best crop below the incumbent is donor 986, cut
`[0,31] x [0,31]`, with 923 squares and tight box `30.859361960`. The remaining
`0.003856413` is again below one grid pitch: `G = 0`. A 925-square crop needs
tight box `30.875854187`, which is `0.012635814` above the incumbent.

At `n=987`, the best below-incumbent composite is a 22x22 crop (484 squares)
from donor 999 plus a 477-square grid, total 961, at side `31.000000100`. The
grid quantum is coarse: below side 32 the counts are fixed at
`9 * 31 + 22 * 9 = 477`. Adding the missing 26 squares requires another grid
column and row, which first fit at side 32. A capacity-matched composite exists
only above the incumbent: a 956-square crop from donor 988 plus 31 grid squares
at side `31.875854287`, which is `0.009549022` above the incumbent.

## Measured evidence

| `n` | incumbent `L` | candidate `L` | `delta L` | `delta score` |
|---:|---:|---:|---:|---:|
| 645 | `25.87031812251975364740446552983030849778887781162520` | `25.88110922606928005961566084581980832075793593806084` | `0.01079110354952641221119531598949982296905812643564` | `0.00042489928732209481290343013428589750074908278244285` |
| 925 | `30.86321837316950898175795253708386667312879581004381` | `30.87585418729976029554514981301238663036201614886522` | `0.01263581413025131378719727592851995723322033882141` | `0.00041546300935840799755793160111672498572568818508254` |
| 987 | `31.86630526508366147711534673847566523363639134913683` | `31.87585428729975900403701846625347367262293118983507` | `0.00954902221609752692167172777780843898653984069824` | `0.00030394872601830407263425125227642245775058601639270` |

All deltas are positive, so every candidate is worse. The measured score change
over the three groups is `+0.001144311022698806883095612988`.

## Certification

Both transports use the unchanged official metric. The literal file is read
with `read_submission`; the pandas copy is exactly
`pd.read_csv(path).to_csv(index=False)` under pandas 3.1.0.

| `n` | literal certified `L` | pandas certified `L` | literal | pandas | same decimal |
|---:|---:|---:|:---:|:---:|:---:|
| 645 | `25.88110922606928005961566084581980832075793593806084` | `25.88110922606928005961566084581980832075793593806084` | ok | ok | yes |
| 925 | `30.87585418729976029554514981301238663036201614886522` | `30.87585418729976200000000000000000000000000000000000` | ok | ok | no |
| 987 | `31.87585428729975900403701846625347367262293118983507` | `31.87585428729976000000000000000000000000000000000000` | ok | ok | no |

The pandas transport changes only the last digits at `n=925` and `n=987`; both
copies still certify and both remain above the incumbent.

## Result

Zero wins. The capacity arithmetic loses on all three targets, and the measured
certified sides lose on all three. This is a negative construction result, not a
near-record. No GPU, CUDA, or Kaggle submission is involved.

## Limits

- Crops use an integer cut grid; a rectangle boundary is an integer. Fractional
  tight boxes are measured after the cut, not searched continuously.
- Donors are capped at `n=1000`. The local portfolio has no gallery or data
  above 1000; donor `n=1000` is a 32x32 upright grid.
- The search is bounded to donors `target + 1 .. 1000` and cut dimensions
  `ceil(sqrt(target)) - 10 .. ceil(sqrt(target))`. It is deterministic, not an
  exhaustive search over all donor subsets.
- The L-shape grid is upright and axis-aligned. Tilted grid fills, non-corner
  crops, and donor mutations are outside this POC.
- The reported `L` is measured from actual vertices. No theoretical or nominal
  container side is reported as measured.

## Reproduce

```sh
python3 poc/donor_slices/poc/run.py
```

Outputs are `data/composite-645.csv`, `data/composite-925.csv`,
`data/composite-987.csv`, their `.pandas.csv` copies, and
`data/results.json`. The run took about 303 seconds of CPU on this machine.
