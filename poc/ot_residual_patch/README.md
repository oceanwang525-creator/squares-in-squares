# Optimal-transport residual patch

## Idea

The successful H7 transport search produced very small residual penetrations
on `n=328` and `n=367`, but its minimum global dilation exceeded the side
budget. This probe tests whether those residuals are local end defects that
can be removed by changing only the highest-stress contact neighborhoods.

The candidate keeps all `n` squares. It:

1. reproduces the best H7 donor map and fixed-side relaxation;
2. builds the overlap graph and ranks the deepest contact pairs;
3. freezes the rest of the packing and re-relaxes only Euclidean patches
   around several separated worst-stress sites;
4. repeats local patches while the maximum violation decreases;
5. applies the unchanged literal and pandas certification only to a complete
   candidate below the incumbent.

No global isotropic dilation is used.

## Quick probe

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
PYTHONPATH=poc/baseline/poc/src $PY \
  poc/ot_residual_patch/poc/probe.py \
  --targets 328,365,367 --out poc/ot_residual_patch/docs/results.json
```

## Result

**Quick probe: negative.** The apparent near-feasible rows are not local
end-defect cases. Their tiny penetrations are spread through one large connected
contact component, and patching separated worst-stress neighborhoods barely
changes the residual.

| n | initial max overlap | initial wall | overlap pairs | largest component | final max overlap | result |
|---:|---:|---:|---:|---:|---:|---|
| 328 | `9.5120e-6` | `7.0301e-6` | 767 | 307 | `9.5115e-6` | no win |
| 365 | `2.0453e-5` | `9.6484e-6` | 726 | 343 | `2.0451e-5` | no win |
| 367 | `1.6342e-6` | `1.0800e-6` | 593 | 337 | `1.6174e-6` | no win |

For each target the probe reproduced H7's best stored seed, then tested 24
separated patches at radii `1.25`, `2.0`, and `3.2`. The largest relative
improvement was on 367 (about 1%), while both wall and pair count remained
essentially unchanged. No candidate reached the `1e-10` local feasibility
gate, so the fixed-angle LP and official certifiers were not entered.

This closes the residual-patch interpretation of H7's 328/365/367 dilation
failures. A future repair would need a genuine global topology change or a
much larger coordinated displacement network, not a one/two-square exchange.

Evidence:

- `docs/results-328.json`
- `docs/results-365.json`
- `docs/results-367.json`

Reproduce the three-row probe with:

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
PYTHONPATH=poc/baseline/poc/src $PY \
  poc/ot_residual_patch/poc/probe.py \
  --targets 328,365,367 --iterations 300 \
  --out poc/ot_residual_patch/docs/results.json
```
