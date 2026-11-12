# External bank expansion (Nov 12 2026)

GitHub repository search surfaced a dozen independent square-packing banks that
the November 10 re-sweep never scanned, plus a `jlevy/squares` commit
(`45e2ecca`, 07:08 UTC) that landed after the Loop 3 refresh. This POC clones
them, harvests every witness/certificate/npz/CSV/JSON layout, and certifies
strict per-`n` candidates against the tracked 22.17659599299808 incumbent.

## Measured result

One strict win, from the newly landed `ozden-s269-issue962` witness:

| n | score gain | old L | new certified L | source |
|---:|---:|---:|---:|---|
| 269 | `0.00009301162632211604` | `16.90099457995132482016692563109368136049563534020245` | `16.89946907585504652716309549503674454576565242778760` | `jlevy/squares` `packing/resources/web/ozden-s269-issue962-2026-11-11/source/n-269-candidate.yaml` |

The merged portfolio scores **22.17650298137176** on literal decimal strings and
**22.176502981371762** after actual `pandas.read_csv(...).to_csv(index=False)`
transport, with **1000/1000 accepted, 0 missing, 0 rejected**. Previous
incumbent: `22.17659599299808`. Certified reduction:
`0.00009301162632211604`.

The tracked file is `poc/pell_corrugation/data/border-scan/merged.csv`.
SHA256 at certification time:
`7de2a8432ca6380678127734fb611cc5c4f090e151d9e804475b2c7ffb4eef05`.

## What was scanned

34,899 files across the twelve banks plus the refreshed `jlevy` tree:

- `JoostdeWinter/square-packing` (source of the de Winter witnesses)
- `TheSnakeFang/squarepack-certs` (90 MB of exact certificates)
- `DigitalEuan/squares_in_squares` (288 MB)
- `cubres/kaggle-squares-in-squares`, `raojohng/squares-in-squares`,
  `anopheles/packingproblem`, `TobiasHFS/squares-in-squares`,
  `omerfisler/kaggle-unit-squares-packing`, `rushikesh648/Squares-in-square-`,
  `mohsensbu-jpg/square_packing_records_karimi`,
  `chromoany/r68-square-packing`
- `jlevy/squares` at `45e2ecca` (07:08 UTC, newer than Loop 3's `455b5d20`)

## Negative checks

- The whole `kenaznia-doga/s<n>-upper-bound` family publishes sides that our
  portfolio already matches or beats at every `n` (156, 172, 179, 180, 199,
  204, 232, 268, 269 old, 270, 299, 303, 307). No geometry from those repos
  improves a tracked row.
- The `couzo-index-2026-10-15` certificates at n=442, 530, 531, 577, 578, 627,
  678, 786, 843, 902, 963 are literal wins of `1.1e-13`..`2.7e-13` but every
  one fails the pandas float64 transport (`certify` raises on the round trip).
  They are the flat `K+0.5355339059` rows where the last digits are not
  transport-stable; the exact wins are numerical dust and are not usable.
- The de Winter n=132/157 raw witnesses certify `1.10e-13`/`1.18e-13` better
  than the tracked dilated rows but again fail pandas transport, so the tracked
  `1e-14`-dilated rows remain the best transport-stable choice.
- `n=172` and `n=199` witnesses fail exact certification entirely (they are the
  regularized/drawing forms, not exact poses).
- `overlap-negative-control.yaml` is a deliberate invalid geometry and is
  rejected as expected.
- The full-portfolio fixed-angle LP probe (`poc/alln_fixed_lp`) found no
  translational slack anywhere, so the incumbent's own rows cannot be improved
  by re-solving at fixed angles.

## Reproduce

```sh
. .venv/bin/activate
python poc/external_bank_expansion/poc/fetch_banks.py --out /tmp/sis_banks
python poc/external_bank_expansion/poc/scan_and_merge.py \
  --repos /tmp/sis_banks \
  --incumbent poc/pell_corrugation/data/border-scan/merged.csv \
  --report poc/pell_corrugation/data/border-scan/merged-literal-official.json \
  --candidates /tmp/bank_cands.json \
  --output /tmp/merged_banks.csv \
  --wins /tmp/bank_wins.json
```

Data only; cloned repositories are never executed. Public authors retain
credit. No Kaggle submission was made.
