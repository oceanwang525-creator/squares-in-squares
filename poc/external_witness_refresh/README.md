# External witness refresh (Nov 12 2026)

The team resweep `../public_resweep_other` scanned four external GitHub packing
banks as of 2026-11-10. One of them, `jlevy/squares`, gained new commits on
2026-11-11 and 2026-11-12 (up to `455b5d20`, 06:06 UTC) with fresh witness and
certificate directories. This POC refreshes all four banks, harvests the new
witness/certificate files, certifies them against the tracked incumbent, and
merges the strict per-`n` winners.

## Measured result

Seven strict wins, all from new `jlevy/squares` witnesses:

| n | score gain | old L | new certified L | source |
|---:|---:|---:|---:|---|
| 132 | 0.000838342 | `11.985680198845833` | `11.976048383603517` | `de-winter-s132-upper-bound-2026-11-11` |
| 156 | 0.000169157 | `12.979165853834291` | `12.977053087497205` | `ozden-s156-upper-bound-2026-11-10` |
| 157 | 0.000534510 | `13.000000000000000` | `12.993302611343341` | `de-winter-s157-upper-bound-2026-11-11` |
| 180 | 0.000102488 | `13.916993522477846` | `13.915618502790026` | `ozden-s180-upper-bound-2026-11-10` |
| 237 | 0.000571495 | `15.902603638259612` | `15.893805580572574` | `ozden-s237-upper-bound-2026-11-11` |
| 268 | 0.000000000124 | `16.872029851111340` | `16.872029849084450` | `ozden-s268-upper-bound-2026-11-10` |
| 269 | 0.000031644 | `16.901513582189148` | `16.900994579951325` | `ozden-s269-upper-bound-2026-11-11` |

The merged portfolio scores **22.17659599299808** on both literal-decimal and
actual `pandas.read_csv(...).to_csv(index=False)` transport, with **1000/1000
accepted, 0 missing, 0 rejected**. Previous incumbent:
`22.178843628823806`. Certified reduction:
`0.002247635825722413`.

The tracked file is `poc/pell_corrugation/data/border-scan/merged.csv`.
SHA256 at certification time:
`f18356d4f428ea90791ae6a1f26b0ef4c2e864a6ea3c9d04008eec664dd7ee03`.

## Pandas repair

The exact rational witnesses at `n=132` and `n=157` pass the literal metric but
overlap after float64 transport. Both were dilated by `1e-14` about the bbox
centre; the certified L still improves by `9.6e-3` and `6.7e-3`. The other five
witnesses survive transport unchanged. `merge_witnesses.py` applies this
dilation ladder automatically and rejects any candidate that cannot pass both
checks.

## What is here

- `poc/fetch_external.py` - clone or refresh the four external banks
  (`Minuii/squares-in-squares`, `jlevy/squares`, `evand/square-packing`,
  `CacinieP/sis-cloud-artifacts`).
- `poc/scan_new.py` - harvest new witness/certificate/npz/CSV/JSON packings;
  handles rational corners, center-basis witnesses, algebraic-number-field
  scalars, `.cert` files, and Minuii `.npz`.
- `poc/merge_witnesses.py` - certify, repair pandas transport by minimal
  dilation, merge strict winners, and write provenance.
- `docs/results.md` - exact commands and output.
- `docs/wins.json` - all seven wins with source, dilation, and both certified
  L values.
- `docs/final-literal-official.json`, `docs/final-pandas-official.json` - full
  per-`n` reports.

## Reproduce

```sh
. .venv/bin/activate
python poc/external_witness_refresh/poc/fetch_external.py --out /tmp/sis_external
python poc/external_witness_refresh/poc/scan_new.py \
  --repo-root /tmp/sis_external \
  --incumbent-report poc/pell_corrugation/data/border-scan/merged-literal-official.json \
  --candidates /tmp/ext_cands.json
python poc/external_witness_refresh/poc/merge_witnesses.py \
  --incumbent poc/pell_corrugation/data/border-scan/merged.csv \
  --candidates /tmp/ext_cands.json \
  --output /tmp/merged_ext.csv --wins /tmp/ext_wins.json
```

Data only; cloned repositories are never executed. Public authors retain
credit. No Kaggle submission was made.
