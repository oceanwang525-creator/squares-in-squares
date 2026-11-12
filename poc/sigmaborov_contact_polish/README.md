# sigmaborov contact-polish kernel (Nov 12 2026)

A new public kernel appeared at 10:12 UTC:
`sigmaborov/squares-in-squares-certified-contact-polish` (public score
`22.2043440241`, so far above the tracked file). Its outputs include a 40 MB
`submission.csv`, a `frozen_public_baseline.csv`, and
`local_witnesses.json` with 37 per-`n` witness arrays.

Fetch the current outputs, prefilter per `n`, certify strict candidates against
the tracked `22.173995906837433` incumbent, and merge.

## Measured result

**3** strict wins, all from the kernel's `submission.csv` (the
`local_witnesses.json` arrays are the same three plus 34 that are at or below
the incumbent):

| n | score gain | old L | new certified L |
|---:|---:|---:|---:|
| 694 | `3.822e-10` | `26.825233542153100` | `26.825233532081742` |
| 551 | `2.416e-10` | `23.925171829038760` | `23.925171823365849` |
| 504 | `1.376e-10` | `22.911415008793174` | `22.911415005703776` |

Literal and pandas transport certify the same L for all three; no dilation
needed. The merged portfolio scores **22.17399590607584** on both transports,
with **1000/1000 accepted, 0 missing, 0 rejected**. Certified reduction:
`7.615905611345757e-10`.

Merged file: `poc/pell_corrugation/data/border-scan/merged.csv`
SHA256: `42421d82d582a6f05e738d13e830ef328161a39f469d00896f58ee521b29cdb5`.

```text
$ . .venv/bin/activate && python /tmp/opencode/score_both.py \
    /tmp/opencode/merged_sigma.csv
rows 500500 n 1000
LITERAL {"score": 22.17399590607584, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
literal seconds 112.31759309768677
PANDAS {"score": 22.17399590607584, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
pandas seconds 114.67661237716675
```

## Negative checks

- `frozen_public_baseline.csv`: 0 candidates (it is the older public baseline).
- The other 34 `local_witnesses.json` entries are at or above the incumbent;
  their float gaps range from `-1e-10` to `-4e-4` (n=988 is `4.19e-4` worse).
- The kernel's own public score (`22.2043`) is far worse than ours, but the
  per-`n` polish still yields three strict wins.

## Reproduce

```sh
. .venv/bin/activate
python poc/kernel_refresh_noon/poc/fetch_outputs.py --out /tmp/sis_sigma \
  --kernels sigmaborov/squares-in-squares-certified-contact-polish
python poc/kernel_refresh_morning/poc/scan_and_merge.py \
  --inputs /tmp/sis_sigma \
  --incumbent poc/pell_corrugation/data/border-scan/merged.csv \
  --report poc/pell_corrugation/data/border-scan/merged-literal-official.json \
  --candidates /tmp/sigma_cands.json \
  --output /tmp/merged_sigma.csv --wins /tmp/sigma_wins.json
```

Data only; public authors retain credit. No Kaggle submission was made.
