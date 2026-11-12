# Measured results

## Portfolio

| check | score | accepted | missing | rejected |
|---|---:|---:|---:|---:|
| literal decimal strings | **22.17650298137176** | 1000 | 0 | 0 |
| actual pandas `read_csv` / `to_csv` transport | **22.176502981371762** | 1000 | 0 | 0 |

Previous incumbent: `22.17659599299808`.
Certified reduction: `0.00009301162632211604`.
Strict per-`n` wins: **1** (`n=269`).

Merged file: `poc/pell_corrugation/data/border-scan/merged.csv`
SHA256: `7de2a8432ca6380678127734fb611cc5c4f090e151d9e804475b2c7ffb4eef05`.

The exact command used on the candidate file and its output:

```text
$ . .venv/bin/activate && python /tmp/opencode/score_both.py \
    /tmp/opencode/merged_n269.csv
rows 500500 n 1000
LITERAL {"score": 22.17650298137176, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
literal seconds 109.65653562545776
PANDAS {"score": 22.176502981371762, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
pandas seconds 108.64327836036682
```

The transport run used actual pandas **3.1.0**. Both runs use the unchanged
official v20 metric.

## Win

| n | score gain | old L | new certified L | dilation | source |
|---:|---:|---:|---:|---:|---|
| 269 | `0.00009301162632211604` | `16.90099457995132482016692563109368136049563534020245` | `16.89946907585504652716309549503674454576565242778760` | - | `ozden-s269-issue962-2026-11-11` |

The pandas transport certifies `16.89946907585504625752607602176401834818405424554676`,
which is also strictly below the old `16.90099457995132482`. No dilation was
needed. Full precision is in `docs/wins.json`; the complete per-`n` reports are
`docs/final-literal-official.json` and `docs/final-pandas-official.json`.

## Scan coverage

34,899 files from twelve cloned banks plus the refreshed `jlevy/squares`
HEAD `45e2ecca` (07:08 UTC). Prefilter at `float_side < incumbent L - 1e-13`
left 17 `n`; exact certification plus pandas transport left exactly one strict
win. The negative results are enumerated in the POC README.

Data only; no cloned code was executed. Public authors retain credit.
