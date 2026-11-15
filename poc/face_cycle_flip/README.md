# Face-cycle flip

## Idea

Build the active contact graph of a certified packing and look for chordless
short cycles whose square centres enclose an unusually large free polygon.
The face is a discrete ownership unit: try quarter/half turns and reflections
of its boundary squares about the face centroid, release the face plus one
contact ring, re-jam at a side slightly below the incumbent, and certify the
result.

This is not a directionless mechanism walk. The move is selected from a
chordless face of the planar contact graph and reverses the orientation of
that cycle before any continuous repair.

## Reproduce

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
$PY poc/face_cycle_flip/poc/probe.py \
  --targets 969,908,791 --seconds-per-target 240 --workers 3
```

The probe writes the contact/face diagnostics and same-`n` best penalties to
`docs/results.json`. A complete candidate is exported only when both literal
decimal and actual pandas `read_csv`/`to_csv` certification give a strict
reduction in side length.

## Results

**Quick probe: negative.** The current high-`n` contact graphs are almost
fully triangulated. Their elementary chordless faces are triangles: 2,362 at
`n=969`, 1,813 at `n=908`, and 185 at `n=791`.

The probe tried 672, 633, and 780 face/transform/release combinations
respectively. These included rotations by 45, 90, 135, and 180 degrees,
reflections at six face axes, isotropic 0.82/1.10 rescaling, and release rings
of zero, one, and two graph hops.

| n | faces | attempts | valid at `incumbent_L - 2e-5` | meaningful wins |
|---:|---:|---:|---:|---:|
| 969 | 2,362 | 672 | 22 | 0 |
| 908 | 1,813 | 633 | 22 | 0 |
| 791 | 185 | 780 | 35 | 0 |

Every valid third-stage candidate re-jammed to the incumbent side. The best
official difference observed was a rounding-scale `2.4e-15`, which fails the
`-1e-9` real-win threshold. Face flipping therefore has no actionable near
miss in this probe. `final.csv` is header-only.

The machine-readable face ranking, transform labels, release-ring sizes, and
per-stage penetration values are in `docs/results.json`.
