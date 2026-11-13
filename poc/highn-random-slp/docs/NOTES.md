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
implementation is retained with that negative smoke-test result.

## Hard-target campaign

The second campaign targeted 20 difficult rows with trust `0.05`, `7`
workers, 100 iterations, and a seeded noise schedule. It produced seven strict
wins:

| n | old `L` | new `L` | change |
| ---: | ---: | ---: | ---: |
| 687 | 26.723271497397 | 26.723271497153 | `-2.446e-10` |
| 793 | 28.677242856066 | 28.677242856062 | `-4.437e-12` |
| 908 | 30.634070751579 | 30.634070746256 | `-5.323e-9` |
| 910 | 30.679149464636 | 30.679149455927 | `-8.709e-9` |
| 969 | 31.633982501359 | 31.633982501052 | `-3.076e-10` |
| 970 | 31.638982145344 | 31.638981988972 | `-1.564e-7` |
| 971 | 31.664318325152 | 31.664318321172 | `-3.980e-9` |

The exact report also contains 13 no-gain rows. Every winning group passed
both literal and pandas official checks. The total internal gain was
`5.633266e-9`.

These rows are not independent team gains: the deterministic free-SLP POC has
a lower certified value at all seven `n`. They are retained because they show
that randomized directions can escape a single SLP basis, while confirming
that the resulting topology is still the same local contact graph.
