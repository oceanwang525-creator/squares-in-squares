# Later long-escape high-n strict wins

These are the final trust-`0.010` rows compared against the latest upstream
`merged.csv`. Literal values are shown; pandas-roundtrip values are in
[long-escape-r28-gate.csv](../data/long-escape-r28-gate.csv).

| n | old L | new L | delta L |
| ---: | ---: | ---: | ---: |
| 791 | `28.634070751578994` | `28.634070746108842` | `-5.470e-12` |
| 816 | `28.978163041811776` | `28.978163041438868` | `-3.729e-12` |
| 868 | `29.898657165780871` | `29.898657005243942` | `-1.605e-07` |
| 910 | `30.679149464635975` | `30.679149450960350` | `-1.368e-08` |
| 936 | `30.997239988761749` | `30.997239690218212` | `-2.985e-07` |

The interim trust-`0.008` pass improved `868`, `908`, `910`, `912`, and `936`
before this final pass. Exact SLP reports are in the `data/` directory.
