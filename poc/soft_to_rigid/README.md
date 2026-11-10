# Soft-to-rigid (hardening) on squares

Method (yoheinakajima/soft-to-rigid, "hardening"): start each piece as the inscribed disk, deform it toward the target shape while the container is pushed inward, then switch to rigid pieces. Their study finds it is a *different* search, not a better default (rigid starts lowest on 155/188 instances vs 142; hardening wins mostly on tiny squares-in-square, n <= 26).

## Implementation (CPU, numpy/scipy, `poc/`)
- `s2r.py`: superellipses |x/.5|^p+|y/.5|^p<=1, p 2 -> 48 over 30-40 stages while the box shrinks; penalty = boundary samples (K=40) of each piece inside the other's gauge + box overflow, analytic gradient (checked vs finite differences, `test_grad.py`), L-BFGS per stage. Then rigid squares: exact SAT-depth penalty with analytic gradient, raise L until feasible, shrink by continuation, dilate 1+1e-8, certify with the repo's official-metric `evaluate`.
- `run.py` / `sweep.py`: random starts and perturbed-incumbent starts (sigma 0.05 / 0.15 with target L 0.03 below incumbent / 0.30). `incumbent.py` loads merged.csv.

## Measured (official-metric certified L; `docs/results.txt`, `data/sweep_*.json`)
| n | incumbent L | ceil(sqrt n) | best random | best perturbed |
|---|---|---|---|---|
| 51 | 7.690356 | 8 | 8.000004 (grid fallback) | 7.690629 |
| 68 | 8.798795 | 9 | 9.006724 | 8.799225 |
| 86 | 9.820535 | 10 | 10.002489 | 9.820848 |

154 runs, ~25 CPU min. No n beat the incumbent. Estimated score gain: 0.

## Honest negatives
- Random starts never beat the grid fallback by any real margin (they land at or just above ceil(sqrt n)); at these sizes the incumbent structures (Pell/diagonal bands) are far from anything a generic soft relaxation finds.
- Perturbed-incumbent starts relax back to the incumbent basin, slightly worse (+3e-4 to +1e-3), consistent with the earlier basin-hopping/ruin-recreate failures; targeting L 0.03 lower did not help; sigma 0.3 destroys the structure.
- Not run for high n (600-1000): cost grows ~n^2 per evaluation and n<=86 already shows no signal; hardening's published edge is small n only, so relevance to the high-n gaps is nil on this evidence. Only 3 n tested, single p schedule, no tuning.
- No CSV produced; merged.csv untouched. `data/best_*.json` are certified but worse than incumbent rows.
