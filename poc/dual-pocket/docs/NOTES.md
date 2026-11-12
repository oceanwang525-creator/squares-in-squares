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

The synchronized portfolio also contains a strict `n=969` win
(`31.633982502102 -> 31.633982501359`), bringing the final comparison count to
154 wins.

Corner SLP on gap ranks 49–96 adds:

```text
n=587: 24.749891763444 -> 24.749495082532
n=694: 26.827240193605 -> 26.827195821962
```

The following corner-SLP band adds:

```text
n=593: 24.827169922514 -> 24.827169351444
n=921: 30.821411669358 -> 30.821411444366
n=981: 31.796032244408 -> 31.796032193678
```

Corner SLP on gap ranks 145–192 adds:

```text
n=542: 23.785578776598 -> 23.785578771494
n=740: 27.706646517490 -> 27.706484801457
n=806: 28.851387036332 -> 28.851386952536
n=864: 29.851920533231 -> 29.851920522479
n=915: 30.745116621029 -> 30.745116012882
```

Corner SLP on gap ranks 193–240 adds:

```text
n=595: 24.861925377827 -> 24.861738389512
n=869: 29.912162992669 -> 29.912161657170
n=644: 25.847855736064 -> 25.847855697239
n=651: 25.948909294217 -> 25.948909259722
```

The result was merged per puzzle with the latest parallel portfolio, preserving
its concurrent `n=687` improvement.

The following corner-SLP band adds:

```text
n=497: 22.785578776370 -> 22.785578771363
n=696: 26.847855736494 -> 26.847855696639
n=816: 28.978163188659 -> 28.978163145807
n=925: 30.862335749762 -> 30.862335634477
```

The following corner-SLP band adds:

```text
n=502: 22.877638427742 -> 22.877638188979
n=638: 25.763086098967 -> 25.762714542621
n=873: 29.957161972499 -> 29.957161697504
n=934: 30.964145685768 -> 30.964144894529
```

The following corner-SLP band adds:

```text
n=505: 22.930439981526 -> 22.930439952725
n=689: 26.745616649926 -> 26.745616462507
n=756: 27.923500714416 -> 27.923218749667
```

The final corner-SLP band adds:

```text
n=814: 28.944432731185 -> 28.944432156029
```

The support-SLP expansion of gap ranks 49–144 adds ten strict improvements,
including `n=587` and `n=738`, and brings the final comparison count to 156
wins with 45 tiny regressions.

Support SLP on gap ranks 145–240 adds nine more strict improvements, including:

```text
n=595: 24.861738389512 -> 24.861695596669
n=740: 27.706484801457 -> 27.706455253054
```

Support SLP on gap ranks 241–336 adds ten further strict improvements,
including:

```text
n=638: 25.762714542621 -> 25.762130073140
n=595: 24.861695596669 -> 24.861685595857
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

The strict Decimal comparison finds 154 wins and 47 tiny regressions. The
regressions total `4.4585e-10` score over all 1,000 puzzles; the official net
improvement remains `0.0104575919354`.

No Kaggle submission is performed.
