# Cross-`n` graph-motif alignment

## Idea

Packings at different `n` can contain the same local contact motif even when
their global coordinate clusters or bounding boxes do not match. This probe
labels contact-graph stars by orientation class, degree, and ownership of the
four walls, then looks for identical motifs in target rows and related donor
rows.

For each match it transplants the donor motif into the target contact at one of
the eight dihedral transforms, releases the motif and its one-hop contact ring,
and re-jams that neighborhood at a target side below the incumbent. The
complete `n`-square group is exported only after literal-decimal and actual
pandas transport certification both improve the side.

This is graph-edit continuation across `n`. It does not exchange contiguous
coordinate clusters, crop a donor, or run a directionless local relaxation.

## Quick target set

`803, 791, 908` have the largest current leader gaps. Donors are the canonical
rows at `n-2, n-1, n+1, n+2` and the adjacent `K` members of the same fixed
`D = n-K^2` family.

## Run

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
PYTHONPATH=poc/baseline/poc/src $PY \
  poc/crossn_graph_motif_alignment/poc/probe.py \
  --targets 803,791,908 \
  --seconds-per-target 240 \
  --workers 3
```

Outputs are written to `docs/results.json`. `final.csv` remains header-only
unless a complete strict winner is found.

## Results

**Quick probe: negative.** The labeled contact graphs contain many shared star
motifs, especially at `n=908`, but realizing the donor topology in the target
row did not create a strict-win basin.

| n | target star signatures | largest donor match count | matched motifs tried | re-jam attempts | best penetration |
|---:|---:|---:|---:|---:|---:|
| 803 | 161 | 5 | 22 | 208 | `3.846e-4` |
| 791 | 191 | 7 | 21 | 280 | `0` |
| 908 | 532 | 87 | 9 | 200 | `0` |

Zero overlap by itself was not sufficient. At 791 and 908 the best selected
motifs were effectively the same local symmetry as the target; re-jamming kept
the measured side at the incumbent (best apparent strict difference was only
`2.3e-15` at 791, below the required `1e-9`). The much larger motif supply at
908 therefore did not transfer density. At 803, the high-gap target has only a
small set of matching degree>=3 stars, and every transplant retained a
`3.846e-4` overlap after releasing its contact ring.

The probe closes this graph-edit implementation: shared star motifs mostly
encode repeated local lattice symmetry, not portable excess capacity. A larger
donor set would increase the number of matches, but the quick test produced no
one-square or `1e-9` near miss to justify scaling it.

`final.csv` is header-only because no complete group passed the strict dual
transport gate.
