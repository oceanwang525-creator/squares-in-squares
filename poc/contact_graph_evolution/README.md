# Contact-graph evolution

## Idea

Treat a certified packing as a labelled contact graph rather than a bag of
coordinates.  A graph rewrite changes which contacts define the local
constraint network; coordinates are then re-realized only after the rewrite.
The probe evolves single-edge swaps around low-connectivity contact vertices
and asks whether a rewritten topology can approach a smaller container.

The main distinction from the existing mechanism walk is the search unit:
mechwalk moves along a soft mode of one fixed graph, while this probe changes
the graph's edge set before producing a coordinate seed.

## Quick probe

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
$PY poc/contact_graph_evolution/poc/probe.py \
  --targets 803 791 908 \
  --rewrites 24 --generations 2 --workers 3
```

For each target:

1. build the near-contact graph at the canonical geometry;
2. choose a degree-3..8 pivot and rewire one incident contact to a nearby,
   currently non-contact square;
3. seed the pivot and its new neighbour at the new one-unit contact distance;
4. shrink the complete seed to `incumbent_L - 1e-4`;
5. relax the geometry and record exact maximum overlap/wall penetration;
6. retain useful rewritten graphs for a second variation generation.

A candidate is called a win only if it has exactly `n` squares, remains below
the incumbent, and passes both the literal-decimal official metric and the
actual pandas `read_csv`/`to_csv` transport.

## Results

**Quick probe: negative.**  The probe tested 48 rewritten graphs per target
after shrinking the container by `1e-4`.  The best relaxed graph at each target
still had broad, distributed penetration rather than one isolated repairable
contact:

| n | graph edges | pivots | best max overlap | best wall excess | overlap pairs | optimized side |
|---:|---:|---:|---:|---:|---:|---:|
| 803 | 795 | 208 | `5.835e-6` | `6.172e-7` | 768 | `28.817166484` |
| 791 | 965 | 224 | `5.615e-6` | `1.022e-6` | 886 | `28.627536784` |
| 908 | 2,627 | 875 | `6.760e-6` | `5.128e-7` | 2,561 | `30.627536363` |

The smallest single-rewrite penetration was `6.45e-6`, over 600 times larger
than the `1e-10` export gate.  More importantly, the best rewrites were
penetrating hundreds or thousands of pairs, not approaching feasibility at one
seam.  The local optimizer stopped after only three accepted iterations,
indicating that the mutated graph immediately sits in a dense conflicting
state.

No graph produced a candidate at or below its incumbent, and no complete group
was exported.  `final.csv` remains header-only.  This closes the sampled
single-edge genotypic family; a future graph search needs a rewrite operation
that preserves most active contacts rather than densifying an already packed
region.

Machine-readable graph sizes, rewrites, penetration records and the best
diagnostics are in `docs/results.json`.
