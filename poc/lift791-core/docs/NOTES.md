# Notes

## Initial exact refill

Fixing all 791 tracked squares and regenerating the upright fill gives:

| target n | tested side | count | short |
| ---: | ---: | ---: | ---: |
| 908 | `30.634070751579078` | 907 | 1 |
| 971 | `31.636788713000000` | 968 | 3 |
| 972 | `31.638982145344254` | 968 | 4 |
| 973 | `31.666638209670600` | 968 | 5 |

All 16 axis-aligned lattice phases in `{0, 0.25, 0.5, 0.75}^2` give the same
counts at `n=908` and `n=971`, so a simple global grid shift does not close
the gap.

The one-square `n=908` deficit and three-square `n=971` deficit are small
enough to justify exact one- and two-square core-release searches.

## One-square release

Every one of the 791 core squares was removed in turn and the upright region
was regenerated exactly:

```text
n=908  tasks=791  best=907/908
n=971  tasks=791  best=968/971
```

No individual release changes the maximum count.

## Rigid transforms

All eight dihedral transforms were combined with valid quarter-unit core
translations while enforcing the container walls in the worker:

```text
n=908  transform tasks=392  best=907/908
n=971  transform tasks=968  best=968/971
```

The valid best is the original orientation at the same centered phase. An
earlier exploratory run reported 909 for `n=908`, but the core had been shifted
outside the box. `exact_fill` does not validate fixed squares against the
walls, so that result was discarded and a worker-side wall check was added.

## Motif replacement

Compact motifs from `n=908` and `n=969..973` were placed against the fixed
core. Removing conflicting seed uprights did not recover the count:

```text
n=908  fixed-core motif  best=907/908
n=908  conflict swap     best=905/908
n=971  fixed-core motif  best=968/971
n=971  conflict swap     best=965/971
```

The motif placements either add no net square or remove more upright contacts
than they replace.

## Nearby dual-pocket test

The `n=972` two-pocket template was replayed at its current side for all 64
dihedral pairs and 16 half-unit phase pairs. Even with candidate supports, the
best exact refill is `964/972`; the extracted pocket geometry alone loses the
exit lattice that the full incumbent carries.

No strict win was found.
