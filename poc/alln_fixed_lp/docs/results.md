# Measured results

## Portfolio

| check | score | accepted | missing | rejected |
|---|---:|---:|---:|---:|
| literal decimal strings | **22.176502968074328** | 1000 | 0 | 0 |
| actual pandas `read_csv` / `to_csv` transport | **22.176502968074328** | 1000 | 0 | 0 |

Previous incumbent: `22.17650298137176`.
Certified reduction: `1.3297431220727618e-8`.
Strict per-`n` wins: **2** (`n=595`, `n=875`).

Merged file: `poc/pell_corrugation/data/border-scan/merged.csv`
SHA256: `693aab989cc52cab0310337ab6bbf53319f7b8a759e27c1fa1081e8fb5cb3baf`.

The exact command used on the candidate file and its output:

```text
$ . .venv/bin/activate && python /tmp/opencode/score_both.py \
    /tmp/opencode/merged_lp.csv
rows 500500 n 1000
LITERAL {"score": 22.176502968074328, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
literal seconds 111.29040122032166
PANDAS {"score": 22.176502968074328, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
pandas seconds 124.45218443870544
```

The transport run used actual pandas **3.1.0**. Both runs use the unchanged
official v20 metric.

## Wins

| n | score gain | old L | new certified L |
|---:|---:|---:|---:|
| 595 | `1.2442839765e-8` | `24.86192565354330775433594822850320341001258076580554` | `24.86192535001240769152874135100165402121368801394200` |
| 875 | `8.545911458e-10` | `29.98873321773301106923132166975700539764409810823621` | `29.98873319247497383157810537769367948888875651734886` |

Both literal and pandas certified L are identical for both wins (no dilation).
Full precision is in `docs/wins.json`; the complete per-`n` reports are
`docs/final-literal-official.json` and `docs/final-pandas-official.json`.
`docs/scan.json` has the per-`n` LP status for all 1000 groups.

## Negative sub-results

- 995 of 1000 groups are at the fixed-angle LP optimum: the LP side returns a
  constant `~2.7e-8` above the incumbent, exactly the accumulated 1e-9 safety
  margins. No translational slack exists there.
- 3 groups give an LP solution that is below the incumbent by less than the
  1e-9 margin; the exact certifier correctly rejects them as overlaps
  (`certify-fail`).

Data only; no Kaggle submission was made.
