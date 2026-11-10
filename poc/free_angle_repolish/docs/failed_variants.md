# Failed and inconclusive variants

## n=631 and n=932: no improving move

Neither target found any single-square angle change in the ladder that lowered the LP side. They stopped after pass 1. The final candidates are the baseline LP solution emitted with the safety dilation, and they certify microscopically worse than the incumbent:

- n=631: `-2.0e-16` L, `-8.11e-18` score.
- n=932: `-3.8e-15` L, `-1.24e-16` score.

This is a negative for free angles on these layouts at this step size, not proof that no coordinated angle change could help.

## n=293: strict win, but numerical dust

No angle move was accepted. The candidate is the same LP solution re-emitted, and it certifies 7.9e-12 lower on both transports. That is a strict win by the letter of the rule, but it is numerical noise from re-solving and dilation, not a free-angle gain. It should not be counted as evidence for the hypothesis.

## 490 and 535: pass 2 stalls

Both targets accepted one +0.005 rad move in pass 1 and no move in pass 2, so the 3-pass budget was not exhausted. A finer or larger ladder around the accepted move was not tried because the brief restricts this POC to the small ladder and forbids a parameter sweep.

## Not attempted

- Global or basin-hopping angle optimization; explicitly out of scope.
- Pairwise or block angle moves, which could escape a coordinate-wise local minimum.
- A ladder beyond +/-0.02 rad.
- The remaining 189 repaired targets.
- Whole-portfolio merge and 1,000-puzzle certification; the lead owns that step.

## Interpretation

The only real gains are about 1e-6 score each, from one square rotating by 0.005 rad. The measured total, 2.18e-6, is far below the 0.0268 target gap and below the 0.000724 weighted rounding ceiling. The result supports the pessimistic reading: on these repaired layouts free angles recover a little rounding slack but do not beat the rounded gallery layout in any material way.
