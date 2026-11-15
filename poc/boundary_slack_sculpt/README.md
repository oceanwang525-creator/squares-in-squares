# Boundary slack sculpt

## Idea

A jammed packing can be locally rigid while still carrying slack along a wall.
This probe treats each wall as a one-dimensional profile of edge gaps and
builds coherent displacement waves over its attached contact chain. Squares
within one contact hop of the chain are allowed to rotate and follow the wave.
The result is re-jammed at a slightly smaller side, then passed through a
fixed-angle LP tighten.

This differs from uniform dilation and random pocket jitter. The initial motion
is a shared wall profile, and every event changes which rectangle edges own a
wall contact.

## Quick probe

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
PYTHONPATH=poc/baseline/poc/src $PY \
  poc/boundary_slack_sculpt/poc/probe.py \
  --targets 803,908,969 --side-delta 0.0001 \
  --out poc/boundary_slack_sculpt/docs/results.json
```

The probe reports the source target gap, best residual penetration, side after
local relaxation, LP-tightened side, and wall-ownership changes. A complete
strict candidate is written only after literal-decimal and actual pandas
`read_csv`/`to_csv` transport both pass the official metric.

No public notebook/dataset coordinates are used and no Kaggle submission is
made.

## Results

**Quick probe: negative.** The probe tested 144 coherent wall-wave
configurations for each target. Each configuration allowed the attached chain
and one contact hop to move, with inward amplitude and sinusoidal tangential
phase; when a candidate reached `1e-5` residual, a full-variable re-jam was
also run.

| n | wall contacts L/R/B/T | target gap | best residual | best local side | LP side | result |
|---:|---|---:|---:|---:|---:|---|
| 803 | 17/19/10/6 | `1e-4` | `1.21e-8` | `28.817265876824` | `28.817265879865` | no win |
| 803 | 17/19/10/6 | `1e-6` | `1.65e-8` | `28.817265876824` | `28.817265878311` | no win |
| 908 | 30/26/30/19 | `1e-4` | `1.67e-5` | `30.627635758530` | `30.627638057917` | no win |
| 969 | 29/31/28/29 | `1e-4` | `2.11e-5` | `31.627635758532` | `31.627638633534` | no win |

The 803 near miss is below `1e-5` residual, but full-variable re-jamming
returns the same incumbent box and the subsequent LP side is still above the
incumbent. At 908/969 the coherent bottom-wave can change wall ownership, but
the irreducible residual remains `1e-5` scale and the LP expands slightly.
There is no one-square or `1e-9` near-feasible win to justify scaling this
sampled family.

`final.csv` is header-only because no complete dual-transport winner was
produced.
