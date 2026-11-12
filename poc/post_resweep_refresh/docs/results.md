# Measured results

## Portfolio

| check | score | accepted | missing | rejected |
|---|---:|---:|---:|---:|
| literal decimal strings | **22.178843628823806** | 1000 | 0 | 0 |
| actual pandas `read_csv` / `to_csv` transport | **22.178843628823806** | 1000 | 0 | 0 |

Previous union incumbent: `22.178843629051656`.
Certified reduction: `2.2784924593596167e-10`.
Strict per-`n` wins: **2** (`n=487`, `n=733`), both from the refreshed
`malyshevdanil/squares-in-squares-living-atlas-of-tight-packings` submission.

Merged file: `poc/pell_corrugation/data/border-scan/merged.csv`
SHA256: `aa4fa2ae4fb4ee27f5230d3349eee839536b2efb0351c41d0f54e1c13cb75353`.

The exact command used on the candidate file and its output:

```text
$ . .venv/bin/activate && python /tmp/opencode/score_both.py \
    /tmp/opencode/merged_refresh.csv
rows 500500 n 1000
LITERAL {"score": 22.178843628823806, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
literal seconds 110.5462212562561
PANDAS {"score": 22.178843628823806, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
pandas seconds 111.27803015708923
```

The transport run used actual pandas **3.1.0**. Both runs use the unchanged
official v20 metric and report all 1000 puzzles accepted.

## Wins

| n | score gain | old L | new certified L | source |
|---:|---:|---:|---:|---|
| 487 | `1.2624208889700416e-10` | `22.56493573331747000000000000000000000000000000000000` | `22.56493573053154992587340685122802808519938290659997` | `malyshevdanil/squares-in-squares-living-atlas-of-tight-packings` |
| 733 | `1.0160715703895751e-10` | `27.56562419830311200000000000000000000000000000000000` | `27.56562419555220260000000000000000000000000000000000` | `malyshevdanil/squares-in-squares-living-atlas-of-tight-packings` |

Full precision is in `docs/wins.json`; the complete per-`n` official reports
are `docs/final-literal-official.json` and
`docs/final-pandas-official.json`.

## Negative checks

- All 34 full submission-shaped CSVs already downloaded in Loop 1 (24
  `lucifer19` anchors, `submission_merged.csv`, `submission_ours.csv`) gave
  **0** candidates against the union.
- `kmmmvs/squares-public-merge-daily` exposes no output files at scan time.
- `saamhm/lb-22-257046-auto-optimizer-packing-unit-squares` returns a 26-byte
  placeholder `submission.csv`.
- Refreshed `prvsiyan`, `alexchilton`, `penguin069`, `keshigami`, and
  `amirhezekiahespanto` submissions gave 0 strict candidates.

Public authors retain credit. No Kaggle submission was made.
