# Optimal-transport top-gap fanout

## Idea

H7 and H16 established that a narrow band of optimal-transport candidates can
cross from an already jammed row into a distinct feasible basin. The proven
path is a center-transport barycenter over one or several nearby donors,
followed by a short fixed-side repair, the minimum safe dilation, fixed-angle
LP, and both official transport checks.

This POC applies that path to the top remaining leader gaps without carrying
over any public geometry. It is deliberately a broad screening pass: most
rows are rejected by residual penetration, while only low-residual starts are
repaired and certified.

Targets in the first wave:

`365,456,228,982,262,965,448,545,862,921,911,853,738,746,981,582,901,330,593,803`.

## Run

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
PYTHONPATH=poc/baseline/poc/src $PY \
  poc/ot_gap_fanout/poc/probe.py \
  --targets 365,456,228,982,262,965,448,545,862,921,911,853,738,746,981,582,901,330,593,803 \
  --workers 5 --seconds-per-target 75 --repair-iterations 5000 --top-starts 18
```

Per-target checkpoints are written under `docs/`; wins are merged into
`final.csv` only after both the literal-decimal and actual pandas
`read_csv`/`to_csv` official certification modes pass.

## Results

**Quick scan: negative.** Twenty top-gap targets were screened with the full
multi-donor transport map, short fixed-side repair, safe dilation, fixed-angle
LP, and both official transport checks. The 9.3-minute wave found no certified
strict winner.

| n | best residual penetration | minimum safe-dilation excess over incumbent |
|---:|---:|---:|
| 330 | `4.85e-7` | `7.16e-6` |
| 456 | `9.51e-7` | `1.73e-5` |
| 365 | `1.13e-6` | `2.19e-5` |
| 262 | `1.62e-6` | `2.24e-5` |
| 448 | `5.02e-6` | `9.86e-5` |

The remaining fifteen targets were farther out, from `2.52e-5` to `1.36e-3`
in residual penetration.

### Focused refinement

The four nearest rows were rerun with 64 repaired starts, 25,000 L-BFGS
iterations, three transport temperatures, finer interpolation/step grids, and
repeated fixed-angle LP plus joint-angle SLP. None produced a complete group
below its incumbent.

The closest candidate was `n=330`:

| stage | side |
|---|---:|
| incumbent | `18.692388155425192` |
| initial fixed-side repair | `18.692395664910467` after minimum dilation |
| first fixed-angle LP | `18.692391992442480` |
| first joint SLP | `18.692388172826117` |
| second joint SLP | `18.692388164128836` |
| third joint SLP | `18.692388158061340` |
| fourth joint SLP | `18.692388156320995` |
| fifth joint SLP | `18.692388156260787` |

The final candidate remains `8.36e-10` above the incumbent, so it is not a
strict win and is not exported. The sequence converges by repeated SLP, but
the first fixed-angle LP does not recover the minimum-dilation cost.

For 456/365/262, deeper repair retained little or none of the initial residual
gain once the transport candidate was constrained to the smaller side. The
best 456 residual remained `9.51e-7`, 365 improved only to `1.03e-6`, and 262
improved to `4.81e-7`; their safe-dilation costs remained over budget.

This closes the current top-gap fanout implementation. `final.csv` is
header-only because no candidate passed both official transports at a
strictly smaller side.

Reproduce the focused nearest-row search:

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
PYTHONPATH=poc/baseline/poc/src $PY \
  poc/ot_gap_fanout/poc/refine.py --targets 330,456,365,262 \
  --seconds-per-target 420 --repair-iterations 15000 --top-starts 40
```

Per-target raw evidence is in `docs/result-<n>.json`,
`docs/refine-result-<n>.json`, and the first wave's captured child outputs in
`docs/results.json`.

### n=330 near-miss continuation

The focused result was close enough to justify a separate continuation. The
base case is reproduced exactly from donors `n-1` and `n+1`, weight schedule
`2:1`, Sinkhorn `epsilon=0.018`, target side reduction `1e-6`, and
interpolation `alpha=0.05`.

`poc/near330.py` adds zero-margin fixed-angle LP and repeated exact-Jacobian
joint SLP. `poc/near330_sweep.py` performs two deterministic parameter grids
around the best seed. The best tightened geometry reached
`18.692388155424336`, only `8.56e-13` below the incumbent and therefore
below the `dL <= -1e-9` win threshold.

The remaining `1e-9` cannot be obtained by a uniform contraction. At the
nearest target, the shrunk geometry has a `7.4e-11` to `1.6e-10` overlap;
the exact metric rejects it. Zero-margin LP, joint SLP, and a focused
nonlinear re-jam either return to the incumbent scale or expand the side.
The constrained fixed-side SLP also accepts no correcting step from the
contracted states.

Consequently this continuation has no exportable win. `final.csv` remains
header-only. Evidence:

- `docs/near330-best.json`
- `docs/near330-best-tight.json`
- `docs/near330-grid.json`

Reproduce:

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
PYTHONPATH=poc/baseline/poc/src $PY \
  poc/ot_gap_fanout/poc/near330.py \
  --iterations 30000 --alpha 0.05 --delta 7.5e-7 \
  --epsilon 0.0185 --schedule=-1:2.2,1:1

PYTHONPATH=poc/baseline/poc/src $PY \
  poc/ot_gap_fanout/poc/near330_sweep.py
```
