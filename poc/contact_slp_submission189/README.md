# Contact-constrained refinement of submission189

This experiment refines centers and angles together with sequential linear
programming. Each LP selects a separating axis for nearby pairs, constrains the
corner support and container walls, and minimizes the enclosing square side.
The nonlinear geometry is repaired and checked before an entire puzzle can
replace the supplied layout.

The new layouts are **593, 814, 873, 915, 921, 925, and 934**. The local score
decrease from the supplied `submission189.csv` is approximately
**0.000000419528**, from **22.189454432234** to **22.189454012706**.
The other 993 puzzle groups preserve the input coordinates.

- `data/submission.csv` contains all 1,000 puzzles and 500,500 coordinate rows.
- `docs/search_gains.json` records the seven changes against the supplied input.
- `docs/lower_sides.csv` lists every strictly lower measured side against the
  upstream `pell_corrugation/data/border-scan/merged.csv` receipts.
- `docs/literal.json`, `docs/pandas.json` and `docs/summary.json` contain the
  official evaluator output for the saved file and the pandas round trip.

See [reproduction notes](docs/reproduce.md) for the validation and search commands.
