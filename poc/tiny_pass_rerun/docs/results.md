# Measured results

## Portfolio

| check | score | accepted | missing | rejected |
|---|---:|---:|---:|---:|
| literal decimal strings | **22.173995906837433** | 1000 | 0 | 0 |
| actual pandas `read_csv` / `to_csv` transport | **22.173995906837433** | 1000 | 0 | 0 |

Previous incumbent: `22.173995906838663`.
Certified reduction: `1.2292409126146337e-12`.
Strict per-`n` wins: **8**.

Merged file: `poc/pell_corrugation/data/border-scan/merged.csv`
SHA256: `308fc61effc296bebefa57617d095b5b81055e425541fbb25d9b55296c25d302`.

```text
$ . .venv/bin/activate && python /tmp/opencode/score_both.py \
    /tmp/opencode/merged_tiny.csv
rows 500500 n 1000
LITERAL {"score": 22.173995906837433, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
literal seconds 113.85193109512329
PANDAS {"score": 22.173995906837433, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
pandas seconds 115.38657689094543
```

## Wins

| n | source | old L | new certified L | dL |
|---:|---|---:|---:|---:|
| 904 | `wrap-845-t1` | `30.563136023802043` | `30.563136023765352` | `3.669e-11` |
| 442 | `delete-from-443` | `21.535533905932896` | `21.535533905932831` | `6.455e-14` |
| 732 | `wrap-679-t1` | `27.535533905932977` | `27.535533905932904` | `7.306e-14` |
| 678 | `delete-from-679` | `26.535533905932888` | `26.535533905932853` | `3.500e-14` |
| 577 | `delete-from-578` | `24.535533905932872` | `24.535533905932851` | `2.185e-14` |
| 964 | `wrap-679-t5` | `31.535533905933007` | `31.535533905932983` | `2.305e-14` |
| 627 | `delete-from-628` | `25.535533905932880` | `25.535533905932877` | `3.250e-15` |
| 492 | `delete-from-493` | `22.707106781186654` | `22.707106781186654` | `1.925e-17` |

Both literal and pandas certified L are identical for all eight.

## Negative checks

- The wrap candidates at n=442, 530, 577, 678, 786, 843, 902, 963 all returned
  `no gain` (the new side is `1e-13` above the incumbent), so no wrap beyond
  those listed is usable.
- Deletion across `n+1..n+5` only wins at the five flat pairs
  (442/443, 577/578, 627/628, 678/679, 492/493), and only by `1e-17`..`1e-13`.

The pass is dust-level by construction; it is re-run because it is nearly free
and every strict win is legitimate.
