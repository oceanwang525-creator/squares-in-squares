# Optimal-transport basin transfer

## Idea

Related puzzle sizes often use nearly compatible groups of upright and tilted
square grains, but a rigid crop or a single-square deletion preserves the
wrong wall contacts. This probe treats every square as a unit of
center-orientation mass and computes a balanced Sinkhorn transport map between
the target group and nearby groups. A barycentric map of that transport moves
whole blocks together instead of applying independent coordinate mutations.

The candidate remains a complete `n`-square group. Cardinality differences
between nearby source rows are represented by each target square receiving the
barycenter of its transported donor mass. Several interpolation strengths and
target-side reductions are screened, then the best maps receive a short
fixed-side non-overlap solve.

This is a nonlocal continuation test. It is not donor cropping, single-parent
deletion, one-way motif transplant, or rigid-cluster exchange.

## Reproduce

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
$PY poc/optimal_transport_basin_transfer/poc/probe.py \
  --targets 803,791,908 --seconds-per-target 100 --workers 3 \
  --repair-iterations 5000 --top-starts 24 \
  --out poc/optimal_transport_basin_transfer/docs/results.json
```

The script reads the in-repository canonical packing, computes Sinkhorn maps
to adjacent rows, screens several interpolants, and records the best residual.
A complete candidate is exported only after the unchanged
official metric passes both literal-decimal and actual pandas
`read_csv`/`to_csv` transport, and only if both sides are strictly below the
current incumbent.

No public coordinates or Kaggle submission are used.

## Results

### Certified wins

`n=791` and `n=908` are strict wins in both official transports:

| n | incumbent | literal L | pandas L |
|---:|---:|---:|---:|
| 791 | `28.627635758531547` | `28.627533748315897057` | `28.627533748315897057` |
| 908 | `30.627635758528807` | `30.627535212069181892` | `30.627535212069181892` |

The 791 reduction is `dL = -1.020102156488e-4`, worth
`ds = -3.627065712784e-6`. The donor map came from `n=789`, with
`epsilon=0.025`, target delta `1e-4`, and interpolation `alpha=0.2`.
After the transport solve, the minimum isotropic repair required only a
`1.000000073914` dilation; fixed-angle LP then shortened the exact box to the
reported value.

The 908 extension used donor `n=912`, `epsilon=0.01`, target delta `1e-4`,
and interpolation `alpha=0.1`. Its best repaired residual was
`3.822e-8`; fixed-angle LP reduced the safe side to `30.62753521206918`.
The reduction is `dL=-1.00546459603e-4`, worth `ds=-3.329e-6`.

### Criterion and initial target set

The 160 screened starts for each target used four nearby donors, two Sinkhorn
regularizations, five very small side reductions, and four interpolation
strengths. The first three-target run produced:

| n | incumbent | best transport residual after L-BFGS | result |
|---:|---:|---:|---|
| 803 | `28.81726587682309` | `0.001074` max violation | no win |
| 791 | `28.627635758531547` | `5.316e-8` max violation | strict win after repair |
| 908 | `30.627635758528807` | `7.316e-4` max violation | no win |

The `n=791` map is unusually stable: several nearby donor/delta combinations
land within `1e-7` of feasibility, while 803 and 908 remain three to four
orders of magnitude farther away.

### Scale-up

Two additional six-target waves ran with the same settings and retained the
existing `final.csv` win:

- `790,792,804,848,861,969`: no certified win.
- `328,365,367,456,747,982`: no certified win.

The extended wave used 84 donor/regularization maps and 3,528 starts per
target across `367,328,803,908,969,861`. Only `n=908` produced a certified
win; the other five rows remained tie-only or above their incumbents. A still
longer follow-up on the winning 791 row is recorded by the multi-donor
`ot_multidonor_barycenter` POC.

The closest misses in the scale-up were `n=367` (`1.63e-6` initial max
violation, but its minimum safe dilation exceeded the incumbent) and
`n=328` (`9.51e-6`, also over budget after safe dilations). This is a narrow
one-row result, not a broad transport family.

### Verification

The final file contains two complete groups, 791 and 908. Their literal and
pandas official certification results are identical to all displayed digits
above. The current `final.csv` SHA256 is
`26276b3be52c2bb080d380825b8fc46cd8b7312cdcf47c76bf17432058304b43`.

### Limits

The Sinkhorn map alone is not feasible geometry. This probe succeeds only
when the transported topology is within a very small dilation of feasibility;
the fixed-angle LP is then used to recover exact non-overlap and a smaller
box. Broad donor maps collapse toward the incumbent or expand beyond it.
