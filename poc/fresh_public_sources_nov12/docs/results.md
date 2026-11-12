# Measured results

## Portfolio

| check | score | accepted | missing | rejected |
|---|---:|---:|---:|---:|
| literal decimal strings | **22.178843629051656** | 1000 | 0 | 0 |
| actual pandas `read_csv` / `to_csv` transport | **22.178843629051656** | 1000 | 0 | 0 |

The final file is the per-`n` union of the team merge `e6f7b69`
(`22.19906754760468`) and this POC's public import (`22.178867829033475`):
188 `n` from the team file, 190 from the public import, 622 ties.
Certified reduction vs the team merge: `0.020223918553029`.
Certified reduction vs the earlier `22.199718022945788` incumbent:
`0.020874393894132`.
Merged file: `poc/pell_corrugation/data/border-scan/merged.csv`
SHA256: `4516c499130da19ac9b0c1ec01de2b65f59bfd10c47d82029b6c2882295ab2c9`.

The exact command used on the literal tracked file and its output:

```text
$ . .venv/bin/activate && python /tmp/opencode/score_both.py \
    poc/pell_corrugation/data/border-scan/merged.csv
rows 500500 n 1000
LITERAL {"score": 22.178843629051656, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
literal seconds 111.63696670532227
PANDAS {"score": 22.178843629051656, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
pandas seconds 110.8271632194519
```

The transport run used actual pandas **3.1.0**, the version required by the
repository's construction-transport tests.

`score_both.py` calls `squares.core.read_submission`, then
`squares.core.score` on the literal strings, then `squares.core.kaggle_roundtrip`
and `squares.core.score` again on the transported rows. Both runs use the
unchanged official v20 metric and report all 1000 puzzles accepted.

## Largest wins

| n | score gain | old L | new certified L | source |
|---:|---:|---:|---:|---|
| 496 | 0.000735 | `22.78113652136377930980926526503900869324189555999651` | `22.76475841297804775210631502115790640606744078791094` | `jonathan1506__50-top-2-6-22-1793-fault-dp-fq-stretch` |
| 629 | 0.000709 | `25.58378219015252996337328338986240000000000000000000` | `25.56600635501839001983868547097110457078056773132835` | `jonathan1506__50-top-2-6-22-1793-fault-dp-fq-stretch` |
| 452 | 0.000614 | `21.77630338260000907905113636934431744240829327135969` | `21.76324356278875740574577340813110536773820954410551` | `jazivxt__boxed-in-sub` |
| 592 | 0.000570 | `24.81290611134547751678880895910034661540782546113358` | `24.79903358948413998104952910395281172694003752845572` | `jazivxt__boxed-in-sub` |
| 589 | 0.000553 | `24.77817459305213233161255787892255427638375652219692` | `24.76475841298041093166026577085029901659226005849078` | `jonathan1506__50-top-2-6-22-1793-fault-dp-fq-stretch` |
| 630 | 0.000527 | `25.60460147573027041536406697384683803275411247401850` | `25.59137814952230000000000000000000000000000000000000` | `jonathan1506__50-top-2-6-22-1793-fault-dp-fq-stretch` |
| 411 | 0.000525 | `20.78694409447977800000000000000000000000000000000000` | `20.77630338260000907905113636934431744240829327135969` | `jonathan1506__50-top-2-6-22-1793-fault-dp-fq-stretch` |
| 905 | 0.000519 | `30.58378219015221148893789606041668642667931142532617` | `30.56816641128635487715784708680916830216843425868995` | `jonathan1506__50-top-2-6-22-1793-fault-dp-fq-stretch` |
| 541 | 0.000513 | `23.78416169719151100000000000000000000000000000000000` | `23.77223267172285105341598444768314760716978612650508` | `jonathan1506__50-top-2-6-22-1793-fault-dp-fq-stretch` |
| 966 | 0.000502 | `31.58378219015222709571700547975052058088080185569984` | `31.56816641128636067440500824258689895211850250060641` | `jonathan1506__50-top-2-6-22-1793-fault-dp-fq-stretch` |

All 188 entries, including source and old/new certified L, are in
`docs/wins.json`. Every replacement was certified individually before the
merged file was written. The union with the team merge was then re-certified
from scratch; the full per-`n` official reports are
`docs/final-literal-official.json` and `docs/final-pandas-official.json`.

## Source files scanned

- `jonathan1506/50-top-2-6-22-1793-fault-dp-fq-stretch`
- `prvsiyan/a-square-deal-research-to-verified-packings`
- `malyshevdanil/squares-in-squares-living-atlas-of-tight-packings`
- `jazivxt/boxed-in-sub`
- `lucifer19/think-select-reclose`
- `nilanjansaha123/squares-gpu-seam-dp-soft-rigid-highs`
- `kmmmvs/squares-public-merge-daily`
- `busyaprime/lb-22-1945500-sacred-geometry-of-1000-squares`
- `mzishran/squares-in-squares-validated-merge-local-search`

Public authors retain credit. No Kaggle submission was made.

## Caveat

This POC imports public data. It is not a new geometric construction. The next
loops should use these packings as donors/seeds for local repair or new
families, and must be measured against this new 22.178843629 incumbent.
