# Continuous row-state transfer

## Idea

This probe tests a restricted but fully constructive version of continuous
row-state transfer. A row-state is a rotated unit-square lattice clipped to a
vertical ribbon. Its state is `(width, angle, x-phase, y-phase)`. A full target
box is formed by three disjoint ribbons:

```
[left rotated lattice][central upright lattice][right rotated lattice]
```

Unlike the earlier fixed-grid skyline search, the row states use exact relative
wall geometry and arbitrary phases. The central upright ribbon is the stable
bulk state. The two side states are selected from a continuous-angle catalog,
so the assembly can switch orientation between neighboring columns.

## Quick falsification

Targets: `n = 861, 747, 804`.

At a side `L` no larger than the incumbent, every candidate side width is
sampled on a 0.25-unit grid. Each width receives a catalog over angles from
-45 to +45 degrees and multiple phases. For each split
`W_left + W_center + W_right = L`, the central ribbon is an exact upright
lattice and the two side ribbons use their best catalog state. The test asks
whether the assembly contains at least `n` squares. This is a capacity test,
not a score based on a proxy.

Run:

```sh
python3 poc/continuous_rowstate_transfer/poc/probe.py \
  --targets 861,747,804 --out poc/continuous_rowstate_transfer/docs/probe.json
```

## Result

**Negative for the vertical row-state decomposition.**

At every tested side from `L_inc - 1e-9` through `L_inc`, the best independent
ribbon assembly contains only the plain upright `K^2` packing:

| n | K | incumbent L | best assembly count |
|---:|---:|---:|---:|
| 861 | 29 | 29.815764558940618 | 842 |
| 747 | 27 | 27.810663458507218 | 729 |
| 804 | 28 | 28.827155380556654 | 784 |

The only gain over the upright bulk is one square at `n=861`; the other two
targets stay exactly at `K^2`. The limiting geometry is that the fractional
widths available beside a full-height central ribbon are mostly too narrow for
a tilted lattice row once its rotated bounding boxes are clipped to the ribbon.
Changing phase and angle does not recover the remaining deficit.

This is a negative result for vertical ribbon transfer, not a proof against
every possible mixed-direction row state. A larger width/fine-grid search and a
horizontal partition are the natural continuations, but neither showed a lead
worth spending the quick-test budget on.

`final.csv` is header-only because no complete strict win was found.

