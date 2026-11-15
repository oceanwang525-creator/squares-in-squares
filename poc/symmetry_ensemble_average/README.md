# Symmetry ensemble average

## Idea

Several certified packings for the same `n` can contain nearly the same local
contact geometry in different global frames. This probe aligns those groups by
the full dihedral group of the square, uses a contact-weighted consensus of
their centers and square orientations, and then attempts a short fixed-side
repair. The goal is to average away frame-specific end defects without
discarding the dense shared contact structure.

The ensemble contains the canonical group and every complete same-`n` group
found in `poc/*/final.csv`. Each alternative is first mapped by a Hungarian
assignment after its best dihedral transform. The consensus blends the
canonical geometry with the aligned weighted mean over several interpolation
strengths.

This is not a single-parent SLP, a donor crop, or a rigid-cluster exchange.

## Reproduce

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
PYTHONPATH=poc/baseline/poc/src $PY \
  poc/symmetry_ensemble_average/poc/probe.py \
  --targets 803,328,908 --repair-iterations 2400 \
  --out poc/symmetry_ensemble_average/docs/results.json
```

Every win must be a complete `n`-square group, strictly shorter than the
canonical side, and pass both the literal-decimal official certifier and the
actual pandas `read_csv`/`to_csv` transport.

## Results

**Quick probe: negative.** The aligned alternatives mostly differ in frame
and micro-perturbation rather than offering a compatible alternate basin.
Contact-weighted averaging therefore creates dense overlaps instead of
removing a coherent end defect.

The probe used at most the three best matching same-`n` alternatives, four
interpolation strengths, and side reductions from `1e-8` through `1e-4`. Each
consensus start received up to 1,200 L-BFGS iterations.

| `n` | aligned alternatives | best max violation | overlap pairs | result |
|---:|---:|---:|---:|---|
| 803 | 3 (scores 2.58e-4, 2.58e-4, 5.10e-4) | 6.23e-2 | 1,557 | expanded |
| 328 | 0 | 5.91e-3 | 715 | expanded |
| 908 | 3 (scores 1.04e-2..1.19e-2) | 2.86e-2 | 2,170 | expanded |

For 803, the closest three groups are effectively copies of the same packing:
their best Hungarian alignments have normalized mismatch below `5.2e-4`.
Averaging them still leaves over 1,500 overlaps. The 908 alternatives have
mismatch around `1e-2`, and their consensus has over 2,000 overlaps. Neither
row is near the `1e-9` repair scale of the successful narrow transport probe.

No complete candidate was below the incumbent, so `final.csv` is header-only.
The result closes this dihedral consensus operation for the current portfolio;
it does not rule out a structurally different multi-basin ensemble.
