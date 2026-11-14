# Cell-grain reassembly

This is an isolated proof-of-concept. It searches for a better finite mixed-angle
cell for a complete high-gap packing, using the existing rectangular SLP and
SAT relaxation machinery. It does not harvest coordinates from the public gap
table and does not submit anywhere.

## First targets

The target list starts at `846, 380, 684, 702, 801`.  The selected incumbents
are complete `K x K + D` cells:

- `846 = 29^2 + 5`, current side `29.583782190152117`
- `380 = 19^2 + 19`, current side `19.961350023811147`
- `684 = 26^2 + 8`, current side `26.642569705751771`

`702` and `801` are kept as neighboring cell structures.  The public gap file is
used only to rank side targets, never as coordinate input.

## Run

From the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python \
  poc/cellgrain_reassemble/run.py --n 846,380,684,702,801 --seeds 5 --jobs 4
```

`--final` writes a candidate to this folder's `final.csv` only after the
unchanged official metric certifies it strictly below the incumbent in both the
literal CSV and an actual pandas `read_csv`/`to_csv` round trip.

## Result: negative

No candidate met the required strict-win threshold (`dL <= -1e-9`), so
`final.csv` is header-only.

- Direct rectangular SLP on `846`, `380`, `702`, and `801` returned no exact
  strict improvement.
- Direct rectangular SLP on `684` returned a dual-valid side of
  `26.64256970575176972883644191592745803762773673673914`, only
  `1.070705805429661046603964098e-15` below the incumbent.  This passes geometry
  but fails the required `1e-9` significance threshold and was not exported.
- Five perturbed `684` searches (full, pocket, and tilted-grain moves) reached
  lower float sides but retained SAT penetration between `2.9e-4` and `2.4e-3`;
  the exact metric rejected them.
- A C++ masked-melt probe on `846` with eight seeds reached float sides near
  `29.5814`, but all retained roughly `4e-4` to `5e-4` SAT penetration.

## Diagnosis

The tested incumbents are already saturated as isolated finite cells under the
existing rectangular SLP.  Small perturbations escape the contact basin only by
creating overlap, and the tested annealing budgets do not repair that overlap
without returning above the incumbent side.  This does not disprove a larger
multi-grain seam construction, but it gives no sign of a feasible replacement
cell within the 15-minute gate.
