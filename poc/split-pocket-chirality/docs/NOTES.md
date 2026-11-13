# Notes

## Method

The two `n=908` tilted contact components are extracted exactly. Each component
is projected onto its principal axis, sorted by projection, and split at the
median square count. Splitting a rigid pocket changes the topology: each half
keeps all internal contacts, but the four halves can choose corner symmetries
and phases independently.

Every assembled four-pocket set must be pairwise SAT-separated. The residual
axis-aligned squares are then selected by the same exact conflict-graph MIS
used by `fillfast.py`, not by an area estimate.

## Runs

The four-half smoke run used the current `n=971` side, all eight dihedral
transforms, half-unit phases, and exact upright refill:

```text
halves                 50, 51, 50, 51
separated tasks        4
best                   520/971
```

This is a strong negative. Splitting both interlocking pockets destroys too
much of the boundary contact that makes the source upright fill dense.
