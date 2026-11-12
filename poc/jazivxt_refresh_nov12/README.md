# jazivxt refresh (Nov 12 2026, ~11:00 UTC)

`jazivxt/boxed-in-sub` re-ran at 10:58 UTC. Its refreshed `submission.csv`
still scores far above the tracked file overall, but per-`n` it improves 17
puzzles, including two large ones.

## Measured result

| check | score | accepted | missing | rejected |
|---|---:|---:|---:|---:|
| literal decimal strings | **22.173824283226253** | 1000 | 0 | 0 |
| actual pandas `read_csv` / `to_csv` transport | **22.173824283226253** | 1000 | 0 | 0 |

Previous incumbent: `22.173995905833525`.
Certified reduction: `1.7162260727358832e-4`.
Strict per-`n` wins: **17**.

Merged file: `poc/pell_corrugation/data/border-scan/merged.csv`
SHA256: `76fa0719a943d089aca7703d81697b91c29964af4e25a6fb0bd164f56ebd8318`.

```text
$ . .venv/bin/activate && python /tmp/opencode/score_both.py \
    /tmp/opencode/merged_loop19.csv
rows 500500 n 1000
LITERAL {"score": 22.173824283226253, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
literal seconds 113.69081044197083
PANDAS {"score": 22.173824283226253, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
pandas seconds 121.43864583969116
```

## Largest wins

| n | score gain | old L | new certified L |
|---:|---:|---:|---:|
| 598 | `1.3483e-4` | `24.906458443` | `24.903161419` |
| 921 | `1.2895e-5` | `30.821403933` | `30.821012592` |
| 996 | `9.0235e-6` | `31.955474330` | `31.955189555` |
| 862 | `5.2018e-6` | `29.827290504` | `29.827137781` |
| 748 | `5.0657e-6` | `27.827290504` | `27.827151959` |
| 922 | `4.5627e-6` | `30.827290504` | `30.827151959` |

The remaining wins are `1e-8`-scale refinements at n=374, 504, 551, 597, 694,
696, 744, 791, 800, 905, 908; full precision is in `docs/wins.json`.

## Method

1. Fetch the kernel's current `submission.csv` (anonymous output API).
2. Prefilter per `n` by floating bbox below the incumbent.
3. Certify literal and actual pandas transport in parallel (32 processes);
   require both strictly below; no dilation was needed.
4. Re-certify all 1000 groups from scratch.

Data only; public authors retain credit. No Kaggle submission was made.
