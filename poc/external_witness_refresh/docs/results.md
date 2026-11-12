# Measured results

## Portfolio

| check | score | accepted | missing | rejected |
|---|---:|---:|---:|---:|
| literal decimal strings | **22.17659599299808** | 1000 | 0 | 0 |
| actual pandas `read_csv` / `to_csv` transport | **22.17659599299808** | 1000 | 0 | 0 |

Previous incumbent: `22.178843628823806`.
Certified reduction: `0.002247635825722413`.
Strict per-`n` wins: **7** (`132, 156, 157, 180, 237, 268, 269`), all from
new `jlevy/squares` witnesses.

Merged file: `poc/pell_corrugation/data/border-scan/merged.csv`
SHA256: `f18356d4f428ea90791ae6a1f26b0ef4c2e864a6ea3c9d04008eec664dd7ee03`.

The exact command used on the candidate file and its output:

```text
$ . .venv/bin/activate && python /tmp/opencode/score_both.py \
    /tmp/opencode/merged_ext.csv
rows 500500 n 1000
LITERAL {"score": 22.17659599299808, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
literal seconds 107.22176384925842
PANDAS {"score": 22.17659599299808, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
pandas seconds 111.73104333877563
```

The transport run used actual pandas **3.1.0**. Both runs use the unchanged
official v20 metric and report all 1000 puzzles accepted.

## Wins

| n | score gain | old L | new certified L | dilation | source |
|---:|---:|---:|---:|---:|---|
| 132 | `0.0008383419099903595` | `11.98568019884583300000000000000000000000000000000000` | `11.97604838360351707000000000000000000000000000000000` | `1e-14` | `de-winter-s132-upper-bound-2026-11-11` |
| 156 | `0.00016915668648959917` | `12.97916585383429121812496955869460475447034129085227` | `12.97705308749720471374579694023149154999999957837244` | - | `ozden-s156-upper-bound-2026-11-10` |
| 157 | `0.0005345098047061375` | `13.00000000000000000000000000000000000000000000000000` | `12.99330261134334150000000000000000000000000000000000` | `1e-14` | `de-winter-s157-upper-bound-2026-11-11` |
| 180 | `0.00010248791641222778` | `13.91699352247784607096072475183677132734124592490938` | `13.91561850279002596923132169153300740408948693736872` | - | `ozden-s180-upper-bound-2026-11-10` |
| 237 | `0.0005714952593815206` | `15.90260363825961200000000000000000000000000000000000` | `15.89380558057257415496177625241584455043389668550914` | - | `ozden-s237-upper-bound-2026-11-11` |
| 268 | `1.2381200484298827e-10` | `16.87202985111134002204460492503125069619342368675845` | `16.87202984908445014797932657906432731018208491044072` | - | `ozden-s268-upper-bound-2026-11-10` |
| 269 | `3.164412493058995e-05` | `16.90151358218914805829710418676554363923909981124033` | `16.90099457995132482016692563109368136049563534020245` | - | `ozden-s269-upper-bound-2026-11-11` |

Full precision, including both literal and pandas certified L, is in
`docs/wins.json`. The complete per-`n` reports are
`docs/final-literal-official.json` and `docs/final-pandas-official.json`.

## Pandas repair

The first full pandas run on the unrepaired merge accepted 998/1000 and
rejected `n=132` and `n=157` with `Squares overlap (certified)`. Both exact
rational witnesses were dilated by `1e-14` about the bbox centre; the
repaired file then passed 1000/1000 in both transports. `merge_witnesses.py`
tries the dilation ladder automatically and never writes a candidate that
fails either check.

## Negative checks

- The other three refreshed banks (`Minuii`, `evand`, `CacinieP`) had no
  commits newer than the resweep snapshot and yielded no new candidates.
- The new `jlevy` `.xz` exact-certification receipts yielded 0 candidates.
- The 56 changed non-dated `jlevy` JSON/YAML files added no wins beyond the
  seven above.
- Algebraic witnesses `n=70` and `n=207` were parsed with a robust bisection
  root finder; their float improvement is only `1e-14`, below the `1e-9`
  prefilter, so they are not strict wins.

Data only; no repository code was executed. Public authors retain credit.
