# Later long-escape high-n strict wins

These are the final trust-`0.014` rows compared against the latest upstream
`merged.csv`. Literal values are shown; pandas-roundtrip values are in
[long-escape-r30-gate.csv](../data/long-escape-r30-gate.csv).

| n | old L | new L | delta L |
| ---: | ---: | ---: | ---: |
| 747 | `27.810663622416302` | `27.810663472083957` | `-3.403e-12` |
| 791 | `28.634070751578994` | `28.634070746019363` | `-5.514e-11` |
| 868 | `29.898657165780871` | `29.898656959892449` | `-2.391e-07` |
| 910 | `30.679149464635975` | `30.679149450839650` | `-1.602e-08` |
| 936 | `30.997239988761749` | `30.997239689800019` | `-2.990e-07` |

The interim trust-`0.008` pass improved `868`, `908`, `910`, `912`, and `936`
before this final pass. Exact SLP reports are in the `data/` directory.
