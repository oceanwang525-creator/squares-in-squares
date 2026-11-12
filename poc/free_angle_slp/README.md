# Free-angle SLP over the whole portfolio (negative)

**Idea.** The fixed-angle LP (`poc/alln_fixed_lp`) only moves centres. Here the
LP also linearises each square's rotation: variables are all `x`, `y`, `θ` and
the container side `L`, with one active separating axis per nearby pair and the
extents/radii expanded to first order in `θ`. A trust region on the angles is
halved until the LP is feasible, the pair set is rebuilt after each accepted
step, and up to six rounds are run. Any step below the incumbent is gated by the
unchanged exact certifier (literal and pandas transport).

This is the strongest purely local test left: it can move and rotate every
square simultaneously, unlike the earlier one-square coordinate descent in
`poc/free_angle_repolish`.

## Measured result: negative, no win

| scope | groups | wins |
|---|---:|---:|
| top ten leader gaps (971, 911, 910, 972, 793, 634, 791, 908, 969, 803) | 10 | 0 |
| all | 1000 | 0 |

Every group returns an SLP side equal to the incumbent to the printed 9
decimals; none is strictly below, so nothing was emitted and `merged.csv` is
untouched. The full 1000-group run took about four minutes.

The top-gap probe in 9 seconds:

```text
n=971 no-gain 31.664318325 vs 31.664318325
n=911 no-gain 30.695622171 vs 30.695622171
n=910 no-gain 30.679149465 vs 30.679149465
n=972 no-gain 31.683990215 vs 31.683990215
n=793 no-gain 28.677242856 vs 28.677242856
n=634 no-gain 25.707106781 vs 25.707106781
n=791 no-gain 28.634070753 vs 28.634070753
n=908 no-gain 30.634070758 vs 30.634070758
n=969 no-gain 31.633982502 vs 31.633982502
n=803 no-gain 28.819189735 vs 28.819189735
```

## Interpretation

The tracked packings are local optima in the full `(x, y, θ)` space, not just at
fixed angles. Combined with `poc/alln_fixed_lp` (no translational slack) and
`poc/leader_poc_k2p4` (the n=328 incumbent is a strict free-angle local
optimum), the remaining gap to the leaders is **structural**: it needs
different contact topologies or different constructions, not more polishing.

## Reproduce

```sh
. .venv/bin/activate
python poc/free_angle_slp/poc/free_angle_slp.py \
  --incumbent poc/pell_corrugation/data/border-scan/merged.csv \
  --report poc/pell_corrugation/data/border-scan/merged-literal-official.json \
  --n-list "$(seq -s, 1 1000)" \
  --output /tmp/slp_all.json --wins /tmp/slp_wins.json --jobs 32
```

Requires `numpy`, `scipy` (HiGHS) and the shared official metric. `docs/scan.json`
holds the per-`n` SLP side and status from the full run.

## Parallel execution (32 threads)

One process per `n` (`--jobs`, default 32), BLAS/OpenMP thread counts pinned to
1, largest `n` first, `imap_unordered`, deterministic sorted merge. The full
1000-group run drops from about four minutes serial to 12.7 s at 32 jobs
(~19x).
