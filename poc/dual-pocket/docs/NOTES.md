# Notes

## Current contribution

`data/submission.csv` is the per-puzzle union of:

- upstream team baseline `poc/pell_corrugation/data/border-scan/merged.csv`,
  officially `22.173995906838663`;
- the aligned local exact-coordinate pool.

The union keeps a local row only when the official metric gives a strictly
smaller `L`. It has 14 wins, no regressions, and scores
`22.173891123045767` in both literal and pandas-round-trip certification.

The source coordinates come from repeated support/corner SLP passes on the
local portfolio. The dominant gains in this contribution are:

```text
n=687: 26.725053394441 -> 26.723271497397
n=747: 27.811200399348 -> 27.810663622416
n=595: 24.861925350012 -> 24.861685491745
n=740: 27.706646517490 -> 27.706455038657
n=934: 30.964145685768 -> 30.964143168755
```

## Contact-SLP workflow

The reusable scripts come from `poc/contact_slp_submission189/poc/`:

```text
refine_support.py
refine_corners.py
```

Each script builds a conservative linear relaxation from the current contact
graph, moves centers and angles inside a trust region, and exports a candidate
only after exact feasibility checks. Repeated passes on the hit puzzles are
material: the first scan changes the geometry, and later passes exploit the
new contacts.

The local pool was also diffed against the previous certified union. An
earlier `pool.write_submission()` pass had replaced a few historically better
rows; restoring the per-puzzle minimum recovered strict wins without changing
the official gate.

## Dual-pocket search

The exploratory search extracted pocket components from the exact coordinates
for `n=908`, `971`, and `972`, enumerated all eight dihedral transforms,
lower-left/upper-right placement, and 0/0.5 lattice phases, then refilled the
remaining upright squares with the exact `fillfast` MIS.

At the target side `30.605885`, the measured upper bounds were:

- `908 x 908`: 890 squares (`18` short);
- `971 x 908`: 896 squares (`12` short).

At the tighter `n=971` target side `31.636789`, the bounds were:

- `971 x 971`: 963 squares (`8` short);
- `972 x 971`: 964 squares (`7` short);
- `972 x 972`: 964 squares (`7` short).

These are bounds for the tested pocket components and placement family, not a
global impossibility result. The search found no certified `n=908` hit, so no
unverified dual-pocket candidate was promoted.

## Negative checks

- Contact SLP on 139 highly tilted puzzles outside the current 400-row target
  table found zero support wins and zero corner wins.
- Repeating an already scanned seed without first updating the geometry found
  no new candidates.
- A pool import that retained 167 extra local rows lowered the total score by
  only `9.8e-10` while creating 181 tiny regressions. It was rejected in favor
  of the strict 14-win union.

## Provenance and limits

The exact coordinates are local certified rows assembled before this
repository contribution. The POC does not include the private donor corpus and
does not claim every winning row is a new construction. It contributes the
complete certified union, the per-puzzle gate, and the search negatives.

No Kaggle submission is performed.
