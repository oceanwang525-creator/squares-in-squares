# Measured results

## Portfolio

| check | score | accepted | missing | rejected |
|---|---:|---:|---:|---:|
| literal decimal strings | **22.175937590726818** | 1000 | 0 | 0 |
| actual pandas `read_csv` / `to_csv` transport | **22.175937590726818** | 1000 | 0 | 0 |

Previous tracked (team `certified_union_nov12`): `22.175937592741963`.
Certified reduction: `2.0151448357305352e-9`.
Replaced `n` against the team file: **11**.
Intermediate union with the team refresh POC (before the team union landed):
`22.175937603064224`, 33 `n` replaced against Loop 6.

Merged file: `poc/pell_corrugation/data/border-scan/merged.csv`
SHA256: `53b76adab4dcbd6a0e76369eb37c7c83444ea0a429a5559d5bbc53571bbd4bd9`.

The exact command used on the final union candidate and its output:

```text
$ . .venv/bin/activate && python /tmp/opencode/score_both.py \
    /tmp/opencode/merged_union_team2.csv
rows 500500 n 1000
LITERAL {"score": 22.175937590726818, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
literal seconds 110.84041953086853
PANDAS {"score": 22.175937590726818, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
pandas seconds 115.0757224559784
```

The transport run used actual pandas **3.1.0**. Both runs use the unchanged
official v20 metric.

## Wins kept from the intermediate

| n | score gain | team file L | final L |
|---:|---:|---:|---:|
| 595 | `1.1403e-9` | `24.861925377827375` | `24.861925350012408` |
| 875 | `8.5388e-10` | `29.988733217733011` | `29.988733192474974` |
| 487 | `1.0675e-11` | `22.564935730531550` | `22.564935730295970` |
| 733 | `1.0180e-11` | `27.565624195552203` | `27.565624195276596` |
| 444 | `5.8587e-14` | `21.564935730297184` | `21.564935730295950` |
| 966 | `3.5436e-14` | `31.568166462695272` | `31.568166462694171` |
| 851 | `7.9953e-15` | `29.684143606359570` | `29.684143606359336` |
| 677 | `3.2668e-15` | `26.520815280171560` | `26.520815280171475` |
| 730 | `2.8129e-15` | `27.520815280171571` | `27.520815280171495` |
| 735 | `1.6230e-15` | `27.608712578448344` | `27.608712578448300` |
| 591 | `2.0395e-16` | `24.799033589484145` | `24.799033589484140` |

Full precision is in `docs/wins.json`. The 33 replacements made by the
intermediate against Loop 6 (largest: n=971 and n=687) are in
`docs/wins_vs_loop6.json`; the team's `certified_union_nov12` already contains
all of them, so they are not replaced again in the final file.

## Method

1. Compare the team refresh POC's official report per `n` with Loop 6 and copy
   strictly better rows; re-certify (result `22.175937603064224`).
2. The team's `certified_union_nov12` then landed at `22.175937592741963`,
   better overall. Adopt it as the new base.
3. Compare the two certified reports per `n` again; keep the 11 rows where the
   intermediate is still strictly better.
4. Re-certify the full 500,500-row union from scratch.

Complete per-`n` reports are `docs/final-literal-official.json` and
`docs/final-pandas-official.json`.
