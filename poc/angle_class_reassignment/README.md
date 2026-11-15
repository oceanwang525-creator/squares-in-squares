# Angle-class reassignment after centre transport

## Idea

H7/H16 transport square centers into a new basin, while H18 showed that putting
an angle term directly inside Sinkhorn creates incoherent wall-side
assignments. This probe separates the two operations:

1. use centre-only/multi-donor Sinkhorn maps to move the geometry;
2. cluster the transported folded angles into classes;
3. reassign class labels either pointwise with spatial smoothing or by a
   contiguous spatial Voronoi partition;
4. replace angles by the circular mean of each class;
5. repair, minimally dilate, tighten with fixed-angle LP, and certify both
   official transports.

The packing always remains complete at exactly `n`.

## Quick probe

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
PYTHONPATH=poc/baseline/poc/src $PY \
  poc/angle_class_reassignment/poc/probe.py \
  --targets 803,328,791 --seconds-per-target 180 \
  --repair-iterations 12000 --top-starts 48
```

The effective incumbent includes the existing H16 `final.csv`, so a candidate
must beat the best available same-`n` group, not only the canonical row.

## Results

**Quick probe: negative.** The probe screened 36,288 starts per target
after centre transport, across 24 Sinkhorn maps, two/three/four angle classes,
pointwise and contiguous-region label modes, and partial/full class
replacement. It then spent 8,000 iterations on each of the best 24 starts.

| n | incumbent side | best repaired violation | best safe-dilated side | result |
|---:|---:|---:|---:|---|
| 803 | `28.8172658768` | `8.58e-4` | `28.8416762601` | expanded |
| 328 | `18.6143689385` | `4.36e-4` | `18.6212859819` | expanded |
| 791 | `28.6265239483` | `5.70e-5` | `28.6281544306` | expanded |

The angle-class rebuild consistently creates a connected overlap network
along class interfaces. The closest n=791 case used a two-region partition,
but the safe dilation still adds `1.63e-3` in side length. None reached a
realized or certifiable candidate.

This closes class reassignment after a global centre map. Keeping angle
classes coherent would require a contact-level reassignment that preserves
the actual interface graph; quantizing or spatially thresholding transported
angles does not.

Evidence:

- `docs/smoke.json`
- `docs/results.json`

Re-run the quick probe:

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
PYTHONPATH=poc/baseline/poc/src $PY \
  poc/angle_class_reassignment/probe.py \
  --targets 803,328,791 --seconds-per-target 120 \
  --repair-iterations 8000 --top-starts 24
```
