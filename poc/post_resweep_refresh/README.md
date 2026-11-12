# Post-resweep public refresh (Nov 12 2026)

The team resweep `e6f7b69` scanned 149 kernels at 2026-11-12 05:05 UTC. Several
public kernels were created or updated after that snapshot, and the Loop 1
import only scanned `submission.csv` of nine kernels. This POC refreshes every
kernel that is newer than the resweep or missing from its log, scans all of
their submission-shaped CSVs, and merges certified per-`n` wins against the
22.178843629 union incumbent.

## Measured result

Two strict per-`n` wins were found, both in the refreshed
`malyshevdanil/squares-in-squares-living-atlas-of-tight-packings` submission:

| n | old L | new certified L | delta L |
|---:|---:|---:|---:|
| 487 | `22.56493573331747` | `22.56493573053154992587340685122802808519938290659997` | `2.7859200741e-9` |
| 733 | `27.565624198303112` | `27.5656241955522026` | `2.7509094e-9` |

The merged portfolio scores **22.178843628823806** on both literal-decimal and
actual `pandas.read_csv(...).to_csv(index=False)` transport, with **1000/1000
accepted, 0 missing, 0 rejected**. Previous union: `22.178843629051656`.
Certified reduction: `2.2784924593596167e-10`.

The tracked file is `poc/pell_corrugation/data/border-scan/merged.csv`.
SHA256 at certification time:
`aa4fa2ae4fb4ee27f5230d3349eee839536b2efb0351c41d0f54e1c13cb75353`.

## What is here

- `poc/fetch_refresh.py` - anonymous download of current CSV outputs for the
  refreshed kernel list.
- `poc/scan_refresh.py` - float-bbox prefilter against the incumbent official
  report, then exact `squares.core.certify` on every strict candidate.
- `poc/merge_refresh.py` - row-preserving replacement of the winning `n`,
  followed by the full 1000-row literal and pandas checks.
- `docs/results.md` - measured commands and exact output.
- `docs/wins.json` - old/new certified L and source for both wins.
- `docs/final-literal-official.json`, `docs/final-pandas-official.json` - full
  per-`n` reports.

## Negative sub-result

Scanning all 34 full submission-shaped CSVs already downloaded by Loop 1
(including 24 `lucifer19` anchors, `submission_merged.csv`, and
`submission_ours.csv`) produced **0** candidates against the union. Final
submissions dominate their own anchors and variants at every `n`.

`kmmmvs/squares-public-merge-daily` currently exposes no output files, and
`saamhm/lb-22-257046-...` returns a 26-byte placeholder `submission.csv`.

## Reproduce

```sh
. .venv/bin/activate
python poc/post_resweep_refresh/poc/fetch_refresh.py --out /tmp/sis_refresh
python poc/post_resweep_refresh/poc/scan_refresh.py \
  --incumbent-report poc/pell_corrugation/data/border-scan/merged-literal-official.json \
  --inputs /tmp/sis_refresh --candidates /tmp/refresh_cands.json
python poc/post_resweep_refresh/poc/merge_refresh.py \
  --incumbent poc/pell_corrugation/data/border-scan/merged.csv \
  --candidates /tmp/refresh_cands.json --output /tmp/merged_refresh.csv
```

Only anonymous Kaggle output endpoints are used; nothing is submitted.
Public authors retain credit.
