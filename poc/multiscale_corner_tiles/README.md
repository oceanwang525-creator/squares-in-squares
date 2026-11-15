# Multiscale corner tiles

## Idea

The existing corner and staircase constructions use one periodic angled layer.
This probe asks whether a boundary module made from two or three staggered
angled layers, with independent angle, phase, depth, and wall attachment, can
retain more squares than the current `328/365/367` packings.

The quick gate is deliberately bounded. For each target it enumerates
two/three-layer schedules along one wall, combines those candidates with
upright filler lattices, and solves exact non-overlap as a maximum independent
set. This tests whether the layered tile can supply a reusable capacity state
before implementing a full corner-composition DP.

## Quick probe

```sh
python3 poc/multiscale_corner_tiles/poc/probe.py \
  --targets 328,365,367 --delta 0.0001
```

The target side is read from the in-repository canonical packing and reduced
by `delta`. A win must contain exactly `n` squares, be strictly smaller in both
literal and actual pandas transports, and then is exported to `final.csv`.

## Results

**Quick gate: negative.** I tested five two/three-layer angle schedules on all
four walls and three upright-grid phases per target, for 60 exact
maximum-independent-set solves per `n` and 180 total. The probe side was
`incumbent_L - 1e-4`.

| n | incumbent L | best combined count | best tile-only count | upright count in same depth | short of n |
|---:|---:|---:|---:|---:|---:|
| 328 | 18.614368938491 | 324 | 44 | 72 | 4 |
| 365 | 19.612856387672 | 361 | 46 | 76 | 4 |
| 367 | 19.692388155425 | 361 | 46 | 76 | 6 |

In every case the exact set-packing solver selected only the ordinary
`floor(L) x floor(L)` upright grid. The angled boundary modules never displaced
one grid layer at a useful density. At the widest tested depth (`3.2`, or
`3.75` for three-layer schedules), the best tile-only independent set retained
44-46 squares while the corresponding upright boundary holds 72-76. The
multilayer geometry therefore loses at least 26 squares inside its own strip;
none of the candidates came close to the one-square repair threshold.

The useful falsification is that independently staggered layers do not fix
the lost sawtooth capacity inside a finite wall-depth band. A future tile needs
a proof-level seam transition that removes this per-layer deficit before a
larger corner DP is justified.

No complete winning group was produced. `final.csv` is header-only. The full
schedule-by-schedule measurements are in `docs/results.json`.
