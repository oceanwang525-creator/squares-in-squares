# Dual-pocket certified union

This POC contributes a complete 1,000-puzzle packing built from exact local
coordinates. It strictly improves 153 `L` values over
`poc/pell_corrugation/data/border-scan/merged.csv`. It also has 47 microscopic
regressions totalling `4.47e-10` score, so the net official improvement is
`0.0102987925531` (`22.199718022946` -> `22.189419230393`).

The largest gains are:

| n | this `L` | baseline `L` | score gain |
| ---: | ---: | ---: | ---: |
| 132 | 11.976048383614462 | 11.985680198845833 | 0.000838342 |
| 452 | 21.763243563002430 | 21.776303382600013 | 0.000614282 |
| 592 | 24.799033589726804 | 24.812906111345477 | 0.000570157 |
| 157 | 12.993302611355388 | 13.000000000000000 | 0.000534510 |
| 693 | 26.799033589751160 | 26.812015736350467 | 0.000493151 |

The full per-`n` list is in [docs/WINS.md](docs/WINS.md). Both literal-decimal
and actual pandas `read_csv`/`to_csv` official-metric reports are checked in
under `data/`.

The latest coordinated topology pass also improves `n=791`
(`28.635045763742` -> `28.634070802306`) and `n=910`
(`30.679153216406` -> `30.679151422149`).

Run from the repository root after following the shared baseline setup:

```sh
python poc/dual-pocket/poc/certify_union.py
```

The script certifies all 1,000 literal rows, repeats the certification after
an actual pandas numeric round trip, and regenerates the differences from the
stored official baseline report.

The dual-pocket search itself did not produce the submitted `n=971` layout.
Its negative search bounds and the exact coordinate provenance are recorded in
[docs/NOTES.md](docs/NOTES.md).
