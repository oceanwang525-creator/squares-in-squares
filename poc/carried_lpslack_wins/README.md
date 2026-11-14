# Carried H75 LPSlack wins

## Provenance

This POC carries the six certified rows from the unmerged H75 `lpslack`
branch commit `55a3886` so they can be proposed against the current
`upstream/main` without editing another team's folder.

The source construction alternates:

1. a selective soft-angle SLP over `(x, y, theta, L)`;
2. an unbounded fixed-angle centre LP after every SLP step; and
3. a uniform separating-axis centre shrink.

Only coordinates that beat the current merge in both the literal CSV and the
actual pandas `read_csv` / `to_csv` transport are included.

## Result

Six strict wins against `merged-22.15325798628.csv`:

| n | Before L | After L | dL |
|---:|---:|---:|---:|
| 740 | 27.70590090058340010 | 27.70590067479485247 | -2.257885e-7 |
| 796 | 28.70588863249223658 | 28.70588238054323134 | -6.251949e-6 |
| 853 | 29.70588590302593463 | 29.70588248048808351 | -3.422538e-6 |
| 861 | 29.81703563083121357 | 29.81703379352127731 | -1.837310e-6 |
| 909 | 30.63883438428050745 | 30.63882860042838675 | -5.783852e-6 |
| 911 | 30.67698912254937767 | 30.67698369769481214 | -5.424855e-6 |

The combined score reduction is `7.812671440564168e-7`, giving
`22.153257204712842`.

## Verification

`poc/verify_final.py` checks:

- the exact four-column header;
- complete `n` groups with contiguous square ids;
- strict improvement in both literal and pandas-certified side lengths; and
- official certification of the six groups.

See `docs/verify_summary.json` for the machine-readable proof.
