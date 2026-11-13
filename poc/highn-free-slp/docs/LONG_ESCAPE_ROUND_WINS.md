# Later long-escape high-n strict wins

These are the final trust-`0.012` rows compared against the latest upstream
`merged.csv`. Literal values are shown; pandas-roundtrip values are in
[long-escape-r29-gate.csv](../data/long-escape-r29-gate.csv).

| n | old L | new L | delta L |
| ---: | ---: | ---: | ---: |
| 747 | `27.810663622416302` | `27.810663472087358` | `-1.140e-12` |
| 791 | `28.634070751578994` | `28.634070746065896` | `-5.513e-11` |
| 816 | `28.978163041811776` | `28.978163041266074` | `-5.457e-10` |
| 868 | `29.898657165780871` | `29.898656984298634` | `-1.815e-07` |
| 910 | `30.679149464635975` | `30.679149450900756` | `-1.374e-08` |
| 936 | `30.997239988761749` | `30.997239690027351` | `-2.987e-07` |

The interim trust-`0.008` pass improved `868`, `908`, `910`, `912`, and `936`
before this final pass. Exact SLP reports are in the `data/` directory.
