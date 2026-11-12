# Dual-pocket certified union

This POC contributes a complete 1,000-puzzle packing built from exact local
coordinates. It strictly improves 154 `L` values over
`poc/pell_corrugation/data/border-scan/merged.csv`. It also has 47 microscopic
regressions totalling `4.47e-10` score, so the net official improvement is
`0.0104253842874` (`22.199718022946` -> `22.189292638658`).

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

A self-generated contact-SLP refinement from PR #2 adds strict wins at
`n=593, 814, 873, 915, 921, 925, 934` for a further `4.1946e-7` total gain.
The latest coordinated layout also tightens `n=973`
(`31.696651125563` -> `31.696609281126`).
The fixed-revision `sqk` corner SLP further tightens `n=910` and `n=687`.
The coordinated `n=969` layout also clears the baseline by a strict export-sized
margin, bringing the final strict-win count to 154.
Corner SLP on the next gap band tightens `n=587` and `n=694`.
The following band additionally tightens `n=593`, `n=921`, and `n=981`.
Corner SLP on gap ranks 145–192 then tightens
`n=542, 740, 806, 864, 915`.
The next band tightens `n=595, 644, 651, 869`, merged against the latest
parallel baseline to avoid overwriting its `n=687` improvement.
The following band tightens `n=497`, `n=696`, `n=816`, and `n=925`.
The next band tightens `n=502`, `n=638`, `n=873`, and `n=934`.
The following band tightens `n=505`, `n=689`, and `n=756`.
The final scanned band tightens `n=814`.

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
