# Rigid-cluster crossover

## Idea

Two certified packings can be treated as contact graphs rather than coordinate
lists. Decompose each graph into geometrically meaningful rigid communities,
match large communities with compatible wall and seam interfaces, and exchange
one community with its complement. Repairing only the new seam could cross a
basin that neither parent can reach by polishing.

This is deliberately different from fixed donor crops, one-way motif
transplants, and directionless mechanism walks: the proposed move retains both
parents as complete packings and changes cluster ownership across a shared
interface.

## Quick falsification

The first gate asks whether the target and donor graphs even contain a large,
non-identical, replaceable cluster. It builds the contact graph at a 0.05 SAT
gap tolerance and reports connected components and their bounding boxes.

Targets: `803, 791, 908`, because they have the largest current leader gaps.
For each target, the donor pool contains:

- the adjacent canonical packings at `n-1` and `n+1`;
- same-`n` alternative groups already present in certified POC finals.

Run:

```sh
python3 poc/rigid_cluster_crossover/poc/screen.py \
  --output poc/rigid_cluster_crossover/docs/screen.json

python3 poc/rigid_cluster_crossover/poc/insertion_probe.py \
  --n 803 791 908 --step 0.1 \
  --output poc/rigid_cluster_crossover/docs/insertion_probe.json
```

## Result

**Negative at the structural gate.** The high-gap targets do not split into
comparable large rigid communities:

| target n | largest incumbent component | largest adjacent component | same-n alternative |
|---:|---:|---:|---|
| 803 | 37 | 10 | distinct rows, same 37-square bbox |
| 791 | 10 | 1 | 9-square component |
| 908 | 8 | 27 (`n=907`) | 6-square component |

The `n=803` packing has several large contact components, but `802`, `804`
do not contain a corresponding large component. The available alternate `803`
layout differs only at dust precision and has the same largest component
bounding box, so exchanging it cannot create a new basin.

For `791`, no component reaches 20 squares and the alternate is also small.
For `908`, `n=907` does contain large strips, but the target is fragmented:
its largest component is only 8 squares. There is no corresponding target
cluster to replace with that donor. Crossover therefore requires wholesale
parent replacement or small-patch exchange, both of which are outside this
hypothesis and already represented by failed donor/motif experiments.

A secondary completion probe placed the `n-1` parent in the target box under
all nine combinations of x/y slack shifts, then searched 0.1-spaced centers at
six orientations. It checked roughly 4.4-5.1 million placements per target.
No unit square fit, so the large `n=907` components cannot simply be completed
into an `n=908` win.

No candidate was close enough to justify invoking the masked-melt/SLP repair
stack. `final.csv` is header-only because no strict certified win was found.

## Method limits

- Connected contact components are a necessary screening proxy, not a formal
  rigidity proof.
- Larger contact tolerance can merge nearby components, but it does not create
  matching interfaces absent at the exact transport tolerance.
- The negative result applies to available same-`n`/adjacent parents. A future
  parent population with genuinely different large clusters would need a new
  probe.

