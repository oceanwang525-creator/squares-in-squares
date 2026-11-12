# Notes

## What this POC contains

`data/submission.csv` is the locally best exact-coordinate union at the time of
the contribution. It is compared directly with the current border-scan baseline
and contains the complete 1,000 puzzles, not only the changed rows.

The largest new high-`n` item is `n=971`, which improves:

```text
31.67679531161114 -> 31.664318325447745
```

That layout has two 136-square tilted pockets with mean canonical angles about
`64.85` and `25.15` degrees. Its exact coordinates are local decimal strings;
no rounding or symbolic form is needed at certification time.

Later coordinated topology passes tighten two more high-`n` layouts:

```text
n=791: 28.635045763742 -> 28.634070802306
n=910: 30.679153216406 -> 30.679151422149
```

Seven additional self-generated contact-SLP refinements tighten
`n=593, 814, 873, 915, 921, 925, 934`; all seven are strict wins against the
coordinated `22.189419` baseline.

The latest coordinated update tightens `n=973`:

```text
31.696651125563 -> 31.696609281126
```

The fixed-revision `sqk` corner SLP then tightens:

```text
n=910: 30.679151422149 -> 30.679150788214
n=687: 26.725053394441 -> 26.725053393193
```

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
- `972 x 971`: 964 squares (`7` short).
- `972 x 972`: 964 squares (`7` short).

These are bounds for the tested pocket components and placement family, not a
global impossibility result. The search found no certified `n=908` hit, so no
unverified candidate was promoted into the submitted CSV.

## Provenance and limits

The winning rows come from the team's local exact-coordinate pool assembled
before this repository contribution. This POC does not include the private
donor corpus or claim that all strict-win rows are new constructions. It
contributes the complete certified union, the reproducible comparison gate,
and the dual-pocket search findings.

The strict Decimal comparison finds 153 wins and 47 tiny regressions. The
regressions total `4.4659e-10` score over all 1,000 puzzles; the official net
improvement remains `0.0103005745458`.

No Kaggle submission is performed.
