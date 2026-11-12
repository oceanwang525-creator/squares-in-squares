# Midday public kernel refresh (Nov 12 2026)

The kernel list grew to 154 and four kernels produced new or changed CSV outputs
after the Loop 5 scan: `nilanjansaha123/squares-gpu-seam-dp-soft-rigid-highs`
(07:57 UTC), `samanyu1808/sis-submit` (07:57), `grennwolf/...santa-2025`
(07:29) and the new `seshurajup/1000-puzzles-ensemble-lb-22-2095` (06:56).
Fetch their current outputs, prefilter per `n` by floating bbox, certify strict
candidates against the tracked incumbent, and merge.

## Measured result

One strict win, from `grennwolf/.../submission_improved.csv`:

| n | score gain | old L | new certified L |
|---:|---:|---:|---:|
| 694 | `4.5268477398e-5` | `26.82724019360512230078618013011167012099681801383733` | `26.82604764628032482866099650919508489835152980397706` |

Literal and pandas transport certify the same L; no dilation needed.

The merged portfolio scores **22.17589232224942** on both transports, with
**1000/1000 accepted, 0 missing, 0 rejected**. Previous incumbent:
`22.175937590726818`. Certified reduction: `4.5268477398062724e-5`.

Merged file: `poc/pell_corrugation/data/border-scan/merged.csv`
SHA256: `eec27294fbd9991e5a7d65dc447070f89715869e27399fadd3c7d4ddefc2ef61`.

## What is here

- `poc/fetch_outputs.py` - anonymous download of current CSV outputs for the
  newest public kernels (or an explicit `--kernels` list).
- `docs/results.md` - exact commands and output.
- `docs/wins.json` - the win with both certified L values.
- `docs/final-literal-official.json`, `docs/final-pandas-official.json` -
  full per-`n` reports.

## Negative checks

- `nilanjansaha123/squares-gpu-seam-dp-soft-rigid-highs` `submission.csv`:
  0 candidates against the incumbent.
- `samanyu1808/sis-submit` `submission.csv`: 0 candidates.
- `seshurajup/1000-puzzles-ensemble-lb-22-2095` `submission.csv`: 0 candidates.
- The same `grennwolf` file re-scanned at a `1e-13` float margin still yields
  only `n=694`, so there is no dust-level win hiding there.

## Reproduce

```sh
. .venv/bin/activate
python poc/kernel_refresh_noon/poc/fetch_outputs.py --out /tmp/sis_noon \
  --kernels nilanjansaha123/squares-gpu-seam-dp-soft-rigid-highs,\
samanyu1808/sis-submit,grennwolf/squares-in-squares-top-1-santa-2025,\
seshurajup/1000-puzzles-ensemble-lb-22-2095
python poc/post_resweep_refresh/poc/scan_refresh.py \
  --incumbent-report poc/pell_corrugation/data/border-scan/merged-literal-official.json \
  --inputs /tmp/sis_noon --candidates /tmp/noon_cands.json
```

Data only; public authors retain credit. No Kaggle submission was made.
