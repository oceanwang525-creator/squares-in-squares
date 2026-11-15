# Heterogeneous lattice set packing

## Idea

Generate unit-square candidates from several independently phased, independently
oriented local lattices, then select one non-overlapping subset with a binary
set-packing MILP.  Unlike a single tilted band or one rectangular pocket, the
selected geometry can contain multiple local orientations and phases in the
same container.

The probe starts from the current incumbent structure without reusing public
coordinates:

1. read the in-repository canonical packing for the target `n`;
2. collect its dominant non-axis orientations;
3. add upright candidates at a small palette of translational phases;
4. add rotated local-lattice blocks at all four corners, using angles near the
   incumbent modes and rational-slope angles;
5. add a safety gap to every generated lattice, build the pairwise conflict
   graph with exact square SAT, and solve a maximum independent set with the
   SciPy HiGHS MILP backend.

The side is probed only a small amount below the incumbent.  A probe is a
success only when the MILP reaches at least `n` squares and the complete group
then passes the unchanged official metric in both literal-decimal and actual
pandas transport.

## Quick probe

From the repository root:

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
$PY poc/heterogeneous_lattice_mip/poc/probe.py \
  --targets 803,791,908 --seconds 900 --seed 1
```

The runner writes `docs/results.json`, keeps all search details in that JSON,
and exports a complete dual-certified strict win to `final.csv` when one is
found.

## Requirements

- the in-repository canonical packing;
- SciPy with `scipy.optimize.milp` and its bundled HiGHS solver;
- the shared `squares` package under `poc/baseline/poc/src`;
- pandas 3.1.x for the actual CSV transport certification.

No public notebook or dataset coordinates are used.  No Kaggle submission is
performed.

## Results

**Quick probe: negative.**  The side was probed `0.0002` below the current
canonical side.  The heterogeneous block search sampled rotated local lattices
at all four corners and solved an upright-grid MIS against each selected block
set.  The global candidate union was also passed to the HiGHS set-packing
probe.

| n | target count | best heterogeneous-block count | best global-pool count | short |
|---:|---:|---:|---:|---:|
| 803 | 803 | 756 | 784 | 19 |
| 791 | 791 | 756 | did not improve | 35 |
| 908 | 908 | 872 | did not improve | 36 |

The 803 global-pool result is exactly the plain `28 x 28` upright grid.  The
MILP did not finish a useful higher-count basin within the short probe limits:
its first 8-second solve selected only 531 mutually non-overlapping points from
the 12,819-candidate union.  In the block probe, a full rotated block replaces
more wall-adjacent upright capacity than it adds, so the best one-block plus
grid layouts remain well below the target count.

This rules out the cheap hypothesis in its sampled form.  It does not prove
that no arbitrarily heterogeneous non-overlap configuration exists, but there
is no `1`-square near miss on which to justify a longer identical run.  No
group is exported: `final.csv` contains only the required header.
