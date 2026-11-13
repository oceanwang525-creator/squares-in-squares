# Notes

The experiment uses the 202 tilted squares from `n=908` as an exact rigid
core, then inserts one compact motif and regenerates every upright square.
Motifs are drawn from the current nearby rows.

## Runs

The first run fixed the 202 tilted squares from `n=908` at the published
frontier side `30.6609125615235`. It tested 100 compact motifs from
`n=908..912` on a `0.25` translation grid, with two placements per motif and
an exact upright MIS refill (`reach=8`, `milp_limit=3`, eight workers).

```text
core                202
motifs              100
placement tasks     100
best                893/911
elapsed             258.6 s
```

All ten highest-count tasks returned 893 squares, so this family is 18 squares
short. The exact `n=908` tilted core does not preserve enough of the upright
exit lattice to transplant into `n=911` at its frontier side.
