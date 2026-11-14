# Carried H75 LPSlack wins

## Provenance

This POC carries verified rows from H75 `lpslack` branch commit `fcdaaf0`
without editing another team's folder. The source branch produced 71
candidates against `merged-22.15325798628.csv`. Two were removed after the
current frontier superseded them: `n=740` and `n=861`.

The source construction alternates:

1. a selective soft-angle SLP over `(x, y, theta, L)`;
2. an unbounded fixed-angle centre LP after every SLP step; and
3. a uniform separating-axis centre shrink.

Only coordinates that beat the current merge in both the literal CSV and the
actual pandas `read_csv` / `to_csv` transport are included.

## Result

69 strict wins against `merged-22.15325720501.csv`. The largest per-`n`
improvements are:

| n | Before L | After L | dL |
|---:|---:|---:|---:|
| 793 | 28.65928114610129269 | 28.65923071103240149 | -5.043507e-5 |
| 978 | 31.75387397150769273 | 31.75385305931283115 | -2.091219e-5 |
| 970 | 31.63770113551068056 | 31.63768686928383889 | -1.426623e-5 |
| 499 | 22.82717217391051682 | 22.82716214030662807 | -1.003360e-5 |
| 971 | 31.64188135677304787 | 31.64187137042664394 | -9.986346e-6 |
| 700 | 26.90312716482531069 | 26.90311842677951513 | -8.738046e-6 |

The combined score reduction is `1.0073704799507465e-5`, giving
`22.15324713130520049253465719`.

## Verification

`poc/verify_final.py` checks:

- the exact four-column header;
- complete `n` groups with contiguous square ids;
- strict improvement in both literal and pandas-certified side lengths; and
- official certification of all 69 groups.

See `docs/verify_summary.json` for the machine-readable proof.
