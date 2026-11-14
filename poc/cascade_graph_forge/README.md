# Cascade graph forge

## Idea

The H78 homocasc work proved that an L-border peel can move a better
free-form basin from one `n` into a lower-`K` child, and that a winning child
can become the donor for another edge.  This POC treats those transitions as a
graph over the complete portfolio instead of hand-selecting a handful of
pairs.

For a packing of `n_d` squares with side `L_d`, a one-unit L-border contains
`2*ceil(L_d)-1` squares.  In the high-`n` regime where
`ceil(L_d) = K_d+1`, its inverse edge is

```
n_child = n_d - (2*K_d + 1)
```

and the peeled object has side about `L_d-1`.  A graph edge is promising when
the projected child side is below the current certified child side.

The runner then uses the proven H78 stack:

1. generate clean, clamp, relocation and seam-surgery peel seeds;
2. run a C++ masked melt around the changed seam;
3. polish with joint `(x,y,theta)` SLP and fixed-angle LP;
4. run contact-Jacobian mechanism walking plus lpslack;
5. certify the final complete groups in literal CSV and actual pandas
   `read_csv`/`to_csv` transport.

## First graph edges

| edge | projected reason |
|---|---|
| `852 -> 795` | `852` improved substantially upstream; `852-2*29-1=793`, but nearby peel layers make `795` a high-value target |
| `860 -> 803` | `803-2*28-1=860`; `860` is the exact one-layer parent |
| `848 -> 791` | `791-2*28-1=848`; `848` is the exact one-layer parent |

The runner also supports arbitrary target/donor pairs so a winning child can
be fed back into the graph.

## Run

```sh
python3 poc/cascade_graph_forge/poc/cascade_forge.py \
  --target 795 --donor 852 --threads 8 --max-seeds 24
```

Repeat with `--target 803 --donor 860` and
`--target 791 --donor 848`.  Results are checkpointed under `docs/`; certified
groups are exported to `final.csv`.

## Result

**Negative.** Six high-value graph edges were run through the full stack. None
produced a feasible certified candidate, and every best polished side remained
strictly above its target incumbent:

| donor -> child | layers | jobs | feasible melt | best `dL` | wall time |
|---|---:|---:|---:|---:|---:|
| `937 -> 876` | 1 | 40 | 0/40 | `+2.440536544e-3` | 2401.7 s |
| `999 -> 936` | 1 | 40 | 0/40 | `+4.478651520e-3` | 2187.0 s |
| `762 -> 707` | 1 | 40 | 0/40 | `+4.804166115e-3` | 1237.1 s |
| `852 -> 795` | near-edge | 64 | 0/64 | `+1.590140667e-2` | 1390.0 s |
| `860 -> 803` | 1 | 68 | 0/68 | `+5.747455378e-2` | 444.0 s |
| `848 -> 791` | 1 | 96 | 0/96 | `+4.504239018e-2` | 585.3 s |

The two closest exact graph edges (`937 -> 876` and `999 -> 936`) both polished
to a side about one unit above the target and then converged to a stable
positive offset. Masked melts had minimum penetration around
`2.3e-8..5.2e-8` for the closest seeds, but the resulting basins always
settled above the incumbent after joint SLP, fixed-angle LP, mechwalk and
lpslack. This is a negative result for the selected one-layer peel topology,
not a proof against every possible L-border cascade.

`final.csv` is correctly header-only because no complete group was certified
as a strict win in both transports.

## Reproduce

```sh
python3 poc/cascade_graph_forge/poc/enumerate_edges.py

python3 poc/cascade_graph_forge/poc/cascade_forge.py \
  --target 876 --donor 937 --threads 8 --max-seeds 20
python3 poc/cascade_graph_forge/poc/cascade_forge.py \
  --target 936 --donor 999 --threads 8 --max-seeds 20
python3 poc/cascade_graph_forge/poc/cascade_forge.py \
  --target 707 --donor 762 --threads 8 --max-seeds 20
```

The enumeration script ranks all one-, two-, and three-layer exact L-border
edges in `data/edge_ranking.json`. Each run writes a checkpoint under `docs/`.

## Implementation note

`_atomic()` uses a PID-specific temporary filename. Multiple graph edges can
therefore rebuild `final.csv` concurrently without racing on a shared
`final.csv.tmp`.
