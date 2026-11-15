# Oriented mass transport

## Idea

The verified optimal-transport probe at `n=791` used Sinkhorn coupling of
centres and a post-hoc circular mean of square angles. This probe couples
centre and folded orientation jointly instead. A unit square's orientation
has period `pi/2`, so the orientation coordinate is
`(cos(4 theta), sin(4 theta))`, not an unwrapped angle.

Each target square and donor square is embedded in this joint space. Sinkhorn
transport uses a metric that charges both centre movement and orientation
change. The barycentric donor angle is then smoothed within a small folded
orientation class before coordinate realization. This preserves coherent
angle patches instead of mixing upright and tilted mass through an
orientation-blind centre map.

The candidate remains a complete `n`-square group. It is exported only when
the unchanged official metric accepts both the literal CSV and an actual
pandas `read_csv` / `to_csv` round trip at a strictly smaller side.

## Run

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
PYTHONPATH=poc/baseline/poc/src $PY \
  poc/oriented_mass_transport/poc/probe.py \
  --targets 803,328,908 --seconds-per-target 120 \
  --repair-iterations 5000 --top-starts 24 --workers 3 \
  --out poc/oriented_mass_transport/docs/results.json
```

The lambda sweep includes a centre-only control. A positive result must beat
that control as well as the canonical incumbent.

No public notebook or dataset coordinates are used. No Kaggle submission is
made.

## Results

**Quick probe: negative.** The joint-angle probe screened 90 transport maps
and 1,800 starts per target at sides down to `1e-4` below the incumbent. Each
best repaired start used the centre-only control (`lambda_angle = 0`); adding
the folded orientation term consistently increased residual penetration.

| n | best repaired violation | best safe dilation side | incumbent side | result |
|---:|---:|---:|---:|---|
| 803 | `8.59e-4` | `28.8417740102` | `28.8172658768` | expanded |
| 328 | `3.40e-5` | `18.6149966060` | `18.6143689385` | expanded |
| 908 | `4.13e-4` | `30.6401903494` | `30.6276357585` | expanded |

At `n=328`, the first orientation-weighted start (`lambda_angle=0.12`) reached
`2.01e-4` violation and a safe dilation above `18.6175`, clearly worse than
the `3.40e-5` centre-only start. At 803 and 908 no orientation-weighted start
entered the best twelve.

The result is decisive for the cheap version of this hypothesis: H7's angular
structure in the n=791 success comes from the existing post-transport mean
and downstream repair, not from transporting a folded orientation label as
another Cartesian coordinate. Joint angle mass creates incoherent wall-side
assignments faster than patch smoothing can remove them.

`final.csv` is header-only because no candidate was safely inside the
incumbent side.
