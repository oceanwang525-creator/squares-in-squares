# Replica-exchange parallel tempering (negative)

## Idea

All previous failure modes were descent methods or independent stochastic
chains. This POC runs a **true parallel-tempering ladder**: `R=8` replicas at
geometric temperatures `2e-6 .. 5e-3`, each doing Metropolis moves at a fixed
box side `L`, with adjacent-replica state exchanges using

```
accept = exp((1/T_i - 1/T_j) * (E_j - E_i)).
```

Moves: single translation+rotation, square swap, fault-line row shift, and a
**void-targeted relocation** (rejection-sample a free unit-square site, then
move a random square into it). Energy is squared SAT penetration plus `5x`
squared wall violation. One clean scaled replica seeds the ladder; the others
are diversified with `sigma=0.002`.

## Measured result: negative

n=911 (frontier side 30.6609126), 8 independent ladders x 8 replicas x 40
rounds x 10,000 moves:

| compression | initial E | best E reached | certified |
|---|---:|---:|---|
| 1e-4 | 2.09e-8 | 2.09e-8 | no |
| 1e-5 | 1.96e-10 | 1.96e-10 | no |

No replica reaches the `E < 1e-16` feasibility threshold; the best state is
always the scaled incumbent and exchange/void moves never improve it. Even a
`1e-5` compression is unrepairable, so the contact graph is jammed far below
the LP/descent scale. 0 wins, `merged.csv` unchanged.

Scripts: `poc/pt_mc.py`, `poc/hs_sa.py`. Raw runs in `docs/`.
