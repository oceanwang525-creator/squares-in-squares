# Measured results

The incumbent was the tracked `merged.csv` from `b57e78b`. Its SHA-256 was `1f9c0fd3b7a88e5a0f6c44b567ff2ef354872812c6b0ad8e37807bd1a64e965c`.

The pilot screened seven groups, found a qualifying interlock in six, and tested 24 cross-packing transplants. No transplant passed the side prefilter. The incumbent-seeded controls also returned larger sides after overlap repair. Each side below is a numerical measurement after centre dilation, before official certification.

| n | Trials | Incumbent L | Control L | Best transplant L | Donor n |
|---:|---:|---:|---:|---:|---:|
| 908 | 4 | 30.634070751579 | 30.636239258165 | 30.636212091370 | 969 |
| 910 | 4 | 30.679149464636 | 30.682435383506 | 30.701052750134 | 971 |
| 911 | 0 | 30.695622170949 | 30.699476586340 | — | — |
| 969 | 4 | 31.633982501359 | 31.635986640608 | 31.635935786713 | 908 |
| 971 | 4 | 31.664318325205 | 31.666302436859 | 31.666274182737 | 909 |
| 972 | 4 | 31.683990214555 | 31.686957084425 | 31.687007551930 | 909 |
| 973 | 4 | 31.696609268859 | 31.699734686521 | 31.699727660205 | 971 |

The best 971 transplant came from 909 with 16 squares. Relative to the recipient's assigned patch, the seed displaced a centre by `0.252` and changed an angle by `0.457` rad. The whole-packing repair left `8.72e-5` penetration before dilation. At 908 and 969 the best transplants also retained residual penetration near `1e-4` after the whole-packing stage.

The search uses wall-anchored corner lattices, detected with a `0.055` centre tolerance and 93% occupancy threshold. The pilot found no qualifying occupied interlock for 911 under those detection rules. Only the two lowest-penalty donor choices per patch size were repaired, with 350 local and 420 full iterations. The results bound this pilot and its search settings.

## Fixed-angle repair follow-up

Re-running the four transplants and the control at each of 908, 969 and 971 with the existing `alln_fixed_lp` repair returned an LP projection for two of the twelve transplants and none of the three controls. The solver was allowed up to three rounds with a four-second limit per solve. Neither projected state beat the best dilation-repaired state at its `n`. The best result for each group remained as in the table. This run also yielded no sides below the incumbent, so no candidate advanced to official literal or pandas certification.

## Implementation checks

A finite-difference check on an overlapping four-square example gave Jacobian errors of `5.10e-8` with a partially fixed mask and `5.87e-8` with all squares movable. All eight dihedral transforms preserved measured side in the same example.
