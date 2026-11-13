# Notes

## Method

The runner imports `poc/claude-midn/poc/free_slp.py`, replaces its deterministic
`refine` function, and leaves the original exact repair, official scoring,
group writer and incumbent gate in place.

At iteration `k`, the angle-objective perturbation is

```text
noise_k = noise0 * max(0.02, 1 - k / (0.8 * iterations))
```

The seed is derived from `n` and the command-line campaign seed. A hit is
accepted only when the original `work` function produces a complete group with
both official transports strictly below the input group.

## Runs

Two `n=911` smoke runs were completed at 100 iterations:

```text
trust   noise    status
0.03    1e-6     no-gain
0.02    1e-3     no-gain
```

The randomized directions did not open a feasible basin at `n=911`. The
implementation is retained with the negative result; no final rows were
produced.
