# Void-density triage

## Idea

The expensive families in this repository try to move an existing packing to a
smaller side. This POC asks the complementary question first: can the next one
or two squares be inserted into a related set at a side below a target row?

For a gap row `n`, it takes the canonical `n-1` packing, embeds it in a box at
`L(n) - 1e-7`, and searches for one additional rotated unit square. A hit is a
complete `n` packing at a strictly smaller side. If no one-square insertion
exists, the row is not a cheap repair target for this mechanism.

The search uses a spatial index, a fractional-position grid, and candidate
angles derived from the source's dominant orientations plus a fixed
rational-angle palette. A coarse hit is only a candidate; `final.csv` is
written after the unchanged literal and pandas numeric-transport certifiers
both accept the complete group.

## Run

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
PYTHONPATH=poc/baseline/poc/src $PY \
  poc/void_density_triage/poc/probe.py \
  --targets 803,328,791 --grid-step 0.05 --out \
  poc/void_density_triage/docs/results.json
```

## Result

**Quick audit: negative.** The probe first looked for one direct insertion into
the canonical `n-1` packing at `L_inc - 1e-7`. It then refined the position grid
from `0.05` to `0.02`. No coarse placement passed the exact SAT screen.

| n | source n | source L | target L | angles | grid points | result |
|---:|---:|---:|---:|---:|---:|---|
| 803 | 802 | 28.7988269336453 | 28.8172657768231 | 37 | 70,009,609 | no insertion |
| 328 | 327 | 18.5837821901520 | 18.6143688384910 | 33 | 24,650,605 | no insertion |
| 791 | 790 | 28.6087125784478 | 28.6276357585316 | 33 | 61,462,610 | no insertion |

For the largest target, the two-step route was also screened: even the first
of two added squares could not be placed into the canonical `n-2 = 801`
packing at the `n=803` target side. That fine-grid screen tested 64,260,206
angle/position combinations and found no direct placement.

This does not prove that an arbitrary rearrangement is impossible. It does
close the cheap hypothesis for these three targets: neither the direct
`n-1 + 1` route nor the first move of the `n-2 + 2` route creates even an
uncertified placement. No winning or complete candidate group was produced, so
`final.csv` remains header-only.

Reproduce both audits:

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
PYTHONPATH=poc/baseline/poc/src $PY \
  poc/void_density_triage/poc/probe.py --targets 803,328,791 \
  --grid-step 0.02 --out poc/void_density_triage/docs/results-fine.json

PYTHONPATH=poc/baseline/poc/src $PY \
  poc/void_density_triage/poc/probe.py --targets 803 \
  --source-offset 2 --grid-step 0.02 \
  --out poc/void_density_triage/docs/results-two-step-screen-fine.json
```
