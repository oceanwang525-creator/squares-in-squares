# Measured results

## Portfolio

| check | score | accepted | missing | rejected |
|---|---:|---:|---:|---:|
| literal decimal strings | **22.17399590607584** | 1000 | 0 | 0 |
| actual pandas `read_csv` / `to_csv` transport | **22.17399590607584** | 1000 | 0 | 0 |

Previous incumbent: `22.173995906837433`.
Certified reduction: `7.615905611345757e-10`.
Strict per-`n` wins: **3**.

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

## Wins

| n | score gain | old L | new certified L |
|---:|---:|---:|---:|
| 694 | `3.822e-10` | `26.825233542153100` | `26.825233532081742` |
| 551 | `2.416e-10` | `23.925171829038760` | `23.925171823365849` |
| 504 | `1.376e-10` | `22.911415008793174` | `22.911415005703776` |

Both literal and pandas certified L are identical for all three. Full
precision is in `docs/wins.json`; the complete per-`n` reports are
`docs/final-literal-official.json` and `docs/final-pandas-official.json`.

## Negative checks

- `frozen_public_baseline.csv`: 0 candidates.
- The other 34 `local_witnesses.json` arrays are at or above the incumbent;
  float gaps `-1e-10` to `-4e-4`.
