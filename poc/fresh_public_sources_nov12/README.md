# Fresh public outputs, November 12 2026

Anonymous Kaggle-output sweep of kernels that appeared after the November 10
import. The best public submission found is
`jonathan1506/50-top-2-6-22-1793-fault-dp-fq-stretch` (public LB
`22.1792686666`). Nine submission CSVs were prefiltered per `n` by floating
bounding box, strict candidates were re-certified with the unchanged official
metric, and the best certified packing per `n` was merged into the tracked
incumbent.

A team merge (`e6f7b69`, score `22.19906754760468`) landed while this import
was being certified. Rather than discarding either side, the final tracked file
is the per-`n` minimum of the two: 188 `n` come from the team file, 190 from
this public import, and 622 are identical. This is still a strict win over both
parents at every replaced `n`.

## Measured result

The merged portfolio scores **22.178843629051656** on both literal-decimal and
actual `pandas.read_csv(...).to_csv(index=False)` transport, with **1000/1000
accepted, 0 missing, 0 rejected**. The team merge it was unioned against scored
`22.19906754760468` (reduction `0.020223918553029`); against the earlier
tracked incumbent `22.199718022945788` the reduction is `0.020874393894132`.

The full 1000-row file is the tracked
`poc/pell_corrugation/data/border-scan/merged.csv`. Its SHA256 at certification
time was `4516c499130da19ac9b0c1ec01de2b65f59bfd10c47d82029b6c2882295ab2c9`.

## What is here

- `poc/fetch_outputs.py` - anonymous download of the listed kernels' top-level
  `submission.csv` files (and optional anchor CSVs) into `data/`.
- `poc/merge_public.py` - scan downloaded CSVs, prefilter by floating bbox,
  certify strict candidates with `squares.core.certify`, and write a merged
  candidate plus a per-`n` provenance JSON.
- `docs/results.md` - measured commands, score output, and largest wins.
- `docs/wins.json` - all 188 replaced `n`, source kernel, old/new certified L.
- `docs/final-literal-official.json` and `docs/final-pandas-official.json` -
  full per-`n` official reports for the two transport checks.

`data/` is intentionally ignored; the downloaded public files are several
hundred MB and are not committed. The tracked `merged.csv` is the only large
artifact updated by this POC.

## Reproduce

```sh
. .venv/bin/activate
python poc/fresh_public_sources_nov12/poc/fetch_outputs.py --out /tmp/sis_nov12
python poc/fresh_public_sources_nov12/poc/merge_public.py \
  --incumbent poc/pell_corrugation/data/border-scan/merged.csv \
  --inputs /tmp/sis_nov12 --output /tmp/merged_nov12.csv \
  --wins /tmp/nov12_wins.json
squares validate /tmp/merged_nov12.csv --require-complete
```

The fetch step uses only anonymous Kaggle endpoints and downloads data; it
never submits anything. Public authors retain credit for their packings.
