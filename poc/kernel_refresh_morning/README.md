# Morning public kernel refresh (Nov 12 2026)

The public kernels are now iterating on top of the merged portfolio, and four
of them re-ran between 07:29 and 09:09 UTC:

- `prvsiyan/a-square-deal-research-to-verified-packings` (08:43)
- `jazivxt/boxed-in-sub` (08:51, public LB improved to `22.1779795658`)
- `jonathan1506/50-top-2-6-22-1783-fault-dp-fq-stretch` (09:02)
- `grennwolf/squares-in-squares-top-1-santa-2025` (07:29, unchanged file)

Fetch their current CSV outputs, prefilter per `n`, certify, and merge every
strict winner against the tracked `22.17529852480167` incumbent.

## Measured result

**114** strict per-`n` wins, each certified below the incumbent for both
literal decimal strings and actual pandas 3.1.0 transport, with no dilation
needed:

| source | winning `n` |
|---|---:|
| `prvsiyan` submission | 104 |
| `jonathan1506` submission | 22 |
| `jazivxt` submission | 20 |

(the sources overlap; 114 distinct `n` won overall)

The merged portfolio scores **22.173995906838663** on both transports, with
**1000/1000 accepted, 0 missing, 0 rejected**. Previous incumbent:
`22.17529852480167`. Certified reduction: `0.001302617963011434`.

Merged file: `poc/pell_corrugation/data/border-scan/merged.csv`
SHA256: `e5ef17f711ea2e674eb03b006dfa6b3c018fd3ac5e053cf7166896b0715cda92`.

```text
$ . .venv/bin/activate && python /tmp/opencode/score_both.py \
    /tmp/opencode/merged_loop10.csv
rows 500500 n 1000
LITERAL {"score": 22.173995906838663, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
literal seconds 123.66382241249084
PANDAS {"score": 22.173995906838663, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
pandas seconds 115.2159378528595
```

## Largest wins

| n | score gain | old L | new certified L | source |
|---:|---:|---:|---:|---|
| 642 | `3.4316e-4` | `25.826248643` | `25.817553669` | `jazivxt` |
| 504 | `1.6570e-4` | `22.915135075` | `22.911415009` | `jazivxt` |
| 862 | `1.2158e-4` | `29.830859962` | `29.827290504` | `jonathan1506` |
| 922 | `1.1755e-4` | `30.830859962` | `30.827290504` | `jonathan1506` |
| 507 | `7.0067e-5` | `22.961862385` | `22.960284706` | `jonathan1506` |

The full list with old/new L and source file is in `docs/wins.json`.

## Method

Each candidate `n` was prefiltered by floating bounding box below the
incumbent, then certified with the unchanged official metric on the literal
strings and after an actual `pandas.read_csv(...).to_csv(index=False)`
round trip. Only `n` where both certified L are strictly lower were replaced.
Unreplaced groups keep their exact previous strings.

## Negative checks

- `grennwolf/submission_improved.csv` was unchanged since Loop 8 and gave 0
  candidates.
- `prvsiyan/embedded_baseline.csv` (a different header/format) gave 0.

## Reproduce

```sh
. .venv/bin/activate
python poc/kernel_refresh_noon/poc/fetch_outputs.py --out /tmp/sis_morning \
  --kernels prvsiyan/a-square-deal-research-to-verified-packings,\
jazivxt/boxed-in-sub,\
jonathan1506/50-top-2-6-22-1783-fault-dp-fq-stretch,\
grennwolf/squares-in-squares-top-1-santa-2025
python poc/kernel_refresh_morning/poc/scan_and_merge.py \
  --inputs /tmp/sis_morning \
  --incumbent poc/pell_corrugation/data/border-scan/merged.csv \
  --report poc/pell_corrugation/data/border-scan/merged-literal-official.json \
  --candidates /tmp/morning_cands.json \
  --output /tmp/merged_morning.csv --wins /tmp/morning_wins.json
```

Data only; public authors retain credit. No Kaggle submission was made.
