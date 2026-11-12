# Measured results

## Portfolio

| check | score | accepted | missing | rejected |
|---|---:|---:|---:|---:|
| literal decimal strings | **22.17529852480167** | 1000 | 0 | 0 |
| actual pandas `read_csv` / `to_csv` transport | **22.175298524801672** | 1000 | 0 | 0 |

Previous incumbent: `22.17589232224942`.
Certified reduction: `5.937974477501639e-4`.
Strict per-`n` wins: **1** (`n=269`).

Merged file: `poc/pell_corrugation/data/border-scan/merged.csv`
SHA256: `9195b04515268df0cd53143dd1442fdd63e7f32c0d3a170100665525f353926b`.

The exact command used on the candidate file and its output:

```text
$ . .venv/bin/activate && python /tmp/opencode/score_both.py \
    /tmp/opencode/merged_n269b.csv
rows 500500 n 1000
LITERAL {"score": 22.17529852480167, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
literal seconds 117.51222062110901
PANDAS {"score": 22.175298524801672, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
pandas seconds 123.27251815795898
```

The transport run used actual pandas **3.1.0**. Both runs use the unchanged
official v20 metric.

## Win

| n | score gain | old L | new certified L | dilation | source |
|---:|---:|---:|---:|---:|---|
| 269 | `5.9379744775e-4` | `16.89946907585504652716309549503674454576565242778760` | `16.88973007359563669924200100534631805369196464263972` | - | `kenaznia-doga/s269-upper-bound-1688973007` |

The literal certified L is `16.88973007359563669021593274506696006607022399025597`
and the pandas-transported L is
`16.88973007359563669924200100534631805369196464263972`; both are far below the
old row. Full precision is in `docs/wins.json`; the complete per-`n` reports
are `docs/final-literal-official.json` and `docs/final-pandas-official.json`.

## Feed scan

The live feed (`at` 2026-11-12T00:55-08:00) has 250 `results` and 292
`drawings`. Scanning every numeric side against the incumbent gives exactly one
candidate (n=269). The drawing poses are 5-decimal, so their `~4e-6` apparent
gaps are rounding and are rejected against the feed's own declared sides.

Data only; public authors retain credit. No Kaggle submission was made.
