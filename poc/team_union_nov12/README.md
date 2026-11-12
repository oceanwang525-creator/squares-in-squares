# Team union, November 12 2026

Two team artifacts landed while Loop 5/6 were being pushed:

1. `poc/public_refresh_nov12` (`bdf9805`, `dbd248f`) — a per-`n` public refresh
   with its own certified `data/submission.csv`.
2. `poc/certified_union_nov12` (`f8a97b7`) — an eleven-source union that also
   updated the tracked `merged.csv`.

This POC unions them with the Loop 6 tracked file and records the final
incumbent.

## Sequence and measured results

| step | file | score | note |
|---|---|---:|---|
| Loop 6 tracked | `merged.csv` (Loop 6) | `22.176502968074328` | 1000/1000 both transports |
| union with `public_refresh_nov12` | this POC's intermediate | `22.175937603064224` | 33 `n` taken from the team refresh POC |
| team `certified_union_nov12` | tracked `merged.csv` at `f8a97b7` | `22.175937592741963` | eleven sources, beats the intermediate |
| **final union of the two** | tracked `merged.csv` now | **`22.175937590726818`** | 11 `n` kept from the intermediate |

All four scores certify 1000/1000 accepted, 0 missing, 0 rejected, in both
literal decimal strings and actual pandas 3.1.0 `read_csv`/`to_csv` transport.

The final file is the per-`n` minimum of:

- `poc/certified_union_nov12`'s tracked file (the base), and
- this POC's intermediate file (Loop 6 + `public_refresh_nov12`).

Eleven `n` come from the intermediate (n=595, 875, 487, 733, 444, 966, 851,
677, 730, 735, 591); the other 989 keep the team file's rows. Certified
reduction against the team file: `2.0151448357305352e-9`.

Merged file: `poc/pell_corrugation/data/border-scan/merged.csv`
SHA256: `53b76adab4dcbd6a0e76369eb37c7c83444ea0a429a5559d5bbc53571bbd4bd9`.

```text
$ . .venv/bin/activate && python /tmp/opencode/score_both.py \
    /tmp/opencode/merged_union_team2.csv
rows 500500 n 1000
LITERAL {"score": 22.175937590726818, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
literal seconds 110.84041953086853
PANDAS {"score": 22.175937590726818, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
pandas seconds 115.0757224559784
```

## The 33 `n` from `public_refresh_nov12`

The intermediate union replaced 33 `n` of the Loop 6 file with the team refresh
POC's rows. The two largest:

| n | score gain | Loop 6 L | team refresh L |
|---:|---:|---:|---:|
| 971 | `3.4065e-4` | `31.674933279210167` | `31.664318325447744` |
| 687 | `1.0945e-4` | `26.727922061358046` | `26.725053394441424` |

n=971 was the largest leader gap in the Oct 29 per-puzzle snapshot
(`31.636789` for Cats and boxes); the team row closes `0.0106` of it. The
remaining 31 are in `docs/wins_vs_loop6.json`.

## Final 11 `n` kept from the intermediate

| n | score gain | team file L | final L |
|---:|---:|---:|---:|
| 595 | `1.1403e-9` | `24.861925377827375` | `24.861925350012408` |
| 875 | `8.5388e-10` | `29.988733217733011` | `29.988733192474974` |
| 487 | `1.0675e-11` | `22.564935730531550` | `22.564935730295970` |
| 733 | `1.0180e-11` | `27.565624195552203` | `27.565624195276595` |
| 444 | `5.8587e-14` | `21.564935730297184` | `21.564935730295950` |
| 966 | `3.5436e-14` | `31.568166462695272` | `31.568166462694172` |
| 851 | `7.9953e-15` | `29.684143606359570` | `29.684143606359336` |
| 677 | `3.2668e-15` | `26.520815280171560` | `26.520815280171476` |
| 730 | `2.8129e-15` | `27.520815280171571` | `27.520815280171495` |
| 735 | `1.6230e-15` | `27.608712578448344` | `27.608712578448300` |
| 591 | `2.0395e-16` | `24.799033589484145` | `24.799033589484140` |

Full precision for all 11 is in `docs/wins.json`; the 33 earlier replacements
are in `docs/wins_vs_loop6.json`. Complete per-`n` reports are
`docs/final-literal-official.json` and `docs/final-pandas-official.json`.

## Lesson

A team or public file can be worse overall and still hold the best individual
puzzle. After every team merge, diff the official per-`n` reports, not just the
totals, and rebuild the union before scoring the whole file.
