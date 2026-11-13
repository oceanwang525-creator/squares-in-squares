# Cross-pocket phase search

## Idea

The existing dual-pocket scans enumerate each pocket's dihedral transform but
usually tie the two pockets to the same half-unit channel phase. That misses
the relative phase between the two exits.

This POC composes two exact tilted pockets from different source packings
(initially the `908` and `971` templates), keeps each pocket's upright exit
supports rigidly attached, and enumerates:

- all 64 pairs of pocket dihedral transforms;
- independent left and right half-unit phases, 16 pairs in total;
- exact maximum independent set refill of the upright candidates.

No area estimate is treated as a packing.

## Result

Negative. The independent phases did not turn any tested template into a
complete target group.

| target | first pocket | second pocket | scan | best |
| ---: | --- | --- | --- | ---: |
| 971 | `971:0` (136) | `908:1` (101) | all 64 dihedrals, 16 phase pairs | 959 |
| 971 | `971:0` (136) | `908:0` (101) | all 64 dihedrals, 16 phase pairs | 959 |
| 971 | `908:0` (101) | `908:0` (101) | all 64 dihedrals, 16 phase pairs | 955 |
| 971 | `973:0` (144) | `971:0` (136) | all 64 dihedrals, 16 phase pairs | 964 |
| 971 | `972:0` (140) | `971:0` (136) | all 64 dihedrals, 16 phase pairs | 964 |
| 791 | `908:0` (101) | `908:0` (101) | all 64 dihedrals, 16 phase pairs | 780 |
| 972 | `973:0` (144) | `973:1` (144) | all 64 dihedrals, 16 phase pairs | 965 |
| 972 | `972:0` (140) | `972:1` (135) | all 64 dihedrals, 16 phase pairs | 965 |

The `972` scan was repeated for the top 64 `973 x 973` poses with the default
corner lattices plus all 16 upright-lattice quarter phases. It remained
`965/972`, so the seven-square gap is not an omitted upright translation.

No `final.csv` is included because no complete target group was found, and
therefore there is nothing to run through the full-union official gate.

## Files

- `poc/scan.py` - independent cross-pocket phase scan and exact upright refill.
- `docs/NOTES.md` - commands, geometry checks and measured negatives.
- `data/` - compact scan summaries and best negative probe coordinates.
