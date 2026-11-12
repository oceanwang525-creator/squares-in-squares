# Measured results

## Portfolio

| check | score | accepted | missing | rejected |
|---|---:|---:|---:|---:|
| literal decimal strings | **22.17589232224942** | 1000 | 0 | 0 |
| actual pandas `read_csv` / `to_csv` transport | **22.17589232224942** | 1000 | 0 | 0 |

Previous incumbent: `22.175937590726818`.
Certified reduction: `4.5268477398062724e-5`.
Strict per-`n` wins: **1** (`n=694`).

Merged file: `poc/pell_corrugation/data/border-scan/merged.csv`
SHA256: `eec27294fbd9991e5a7d65dc447070f89715869e27399fadd3c7d4ddefc2ef61`.

The exact command used on the candidate file and its output:

```text
$ . .venv/bin/activate && python /tmp/opencode/score_both.py \
    /tmp/opencode/merged_loop8.csv
rows 500500 n 1000
LITERAL {"score": 22.17589232224942, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
literal seconds 109.61025762557983
PANDAS {"score": 22.17589232224942, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
pandas seconds 110.67009353637695
```

The transport run used actual pandas **3.1.0**. Both runs use the unchanged
official v20 metric.

## Win

| n | score gain | old L | new certified L | source |
|---:|---:|---:|---:|---|
| 694 | `4.5268477398e-5` | `26.82724019360512230078618013011167012099681801383733` | `26.82604764628032482866099650919508489835152980397706` | `grennwolf/squares-in-squares-top-1-santa-2025` `submission_improved.csv` |

Both literal and pandas certified L are identical. Full precision is in
`docs/wins.json`; the complete per-`n` reports are
`docs/final-literal-official.json` and `docs/final-pandas-official.json`.

Data only; public authors retain credit. No Kaggle submission was made.
