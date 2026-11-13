# Notes

The experiment fixes only the tilted contact components of `n=971`; upright
squares are regenerated exactly. Compact motifs are extracted from the current
`n=969..973` rows and must be SAT-separated from the fixed pockets before the
residual MIS run.

The initial side target is `31.6666382096706`, the published frontier value.
Any candidate must subsequently be checked against the current incumbent and
both official transports.

## Runs

First run:

```text
fixed tilted core      272
motifs                 100
grid step              0.25
placements per motif   1
best                   962/973
```

The closer `n=973` frontier side does not compensate for replacing the tracked
upright exit structure. This transplant is far short and is not a win.
