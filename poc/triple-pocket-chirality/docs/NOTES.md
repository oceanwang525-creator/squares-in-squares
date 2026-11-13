# Notes

This POC is a direct follow-up to the four-half negative result. It tests
whether one preserved full pocket is enough to keep the exit lattice coherent
while the split pocket supplies the missing phase freedom.

Every candidate is checked with pairwise SAT before exact upright MIS. No area
estimate is treated as a packing.

## Measured result

At the `n=971` incumbent side:

| source | full + split roster | tasks | best |
| ---: | --- | ---: | ---: |
| 908 | `101 + 50 + 51` | 4 | 947 |
| 971 | `136 + 68 + 68` | 4 | 947 |
| 971 | `136 + 68 + 68` | 100 | 952 |

The three-pocket variants remain below the corrected dual-pocket best of 965
and far below 971. Preserving one full pocket is better than splitting both,
but still loses more contact than the extra phase freedom recovers.
