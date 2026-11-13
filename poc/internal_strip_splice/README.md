# Internal strip splice

A larger packing can be contracted along internal seams while keeping the
pieces on either side rigid. This can carry a dense tilted region into a
smaller square when the donor has straight corridors through its upright
regions.

The search removes squares crossing a vertical or horizontal slab and
translates the surviving right and upper pieces inward. It then tries to
keep some of the squares crossing those seams by solving a conflict graph
for several nearby splits, or by assigning individual squares to a side.
The donor and comparison source is the tracked
`poc/pell_corrugation/data/border-scan/merged.csv`.

The search found zero strict candidates across 3,411 cut-width pairs, 432
fixed-split graph variants and 40 adaptive seam assignments. The adaptive
model recovered one square in four cases, each still one square short.
See the [measured results](docs/results.md) and [method](docs/method.md).

From the repository root, use the shared baseline environment with NumPy,
SciPy and pandas 3.1.0 installed.

```sh
python poc/internal_strip_splice/poc/splice.py
python poc/internal_strip_splice/poc/assign.py --cases 40
```

The scripts write compact capacity reports to `data/scan.json` and
`data/assignment.json`. They emit a separate candidate CSV for any strict
per-puzzle improvements that pass both official checks.
