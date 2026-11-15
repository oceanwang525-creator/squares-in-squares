# Multi-donor optimal-transport barycenter

## Idea

H7 established that a Sinkhorn map from one nearby packing can move the
`n=791` packing into a distinct feasible basin. This POC tests whether mixing
several donors can preserve the useful bulk of one row and the end treatment
of another.

For every target square, a balanced coupling is computed to each donor row.
The donor-to-target maps are then mixed with a small set of sparse weight
vectors, and the resulting centers and folded angles define the seed. Several
sides and interpolation strengths are screened before a fixed-side L-BFGS
repair. A candidate is exported only when a complete group is strictly shorter
than the incumbent in both literal-decimal and actual pandas transport.

This is not donor cropping or deletion. Every target square receives a
many-square barycentric map.

## Quick probe

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
PYTHONPATH=poc/baseline/poc/src $PY \
  poc/ot_multidonor_barycenter/poc/probe.py \
  --targets 803,367,328 --seconds-per-target 120 \
  --repair-iterations 5000 --top-starts 24 --workers 3 \
  --out poc/ot_multidonor_barycenter/docs/results.json
```

The probe never reads public coordinates. No Kaggle submission is made.

## Results

### Certified win

The multi-donor barycenter improves the already certified H7 `n=791` win:

| construction | literal L | pandas L |
|---|---:|---:|
| upstream incumbent | `28.627635758531549059...` | `28.627635758531549059...` |
| H7 single-donor win | `28.627533748315897057...` | `28.627533748315897057...` |
| H16 multi-donor barycenter | **`28.627434433035796153...`** | **`28.627434433035796153...`** |

Relative to upstream, `dL = -2.013255495746e-4` and the official score gain
is `-7.1596e-6`. Relative to H7 alone, H16 contributes another
`dL = -9.931531524e-5`, worth `-3.5326e-6` in score.

The winning schedule mixed donors `n+2` and `n+3` at weights `1:2` with
Sinkhorn `epsilon=0.025`, interpolation `alpha=0.2`, and a requested side
reduction of `2e-4`. The initial repaired state had maximum penetration
`7.46e-8`; fixed-angle LP then reduced its safe-dilated side from
`28.62743780199` to the certified `28.62743443304`.

### Quick-probe outcomes

- `803,367,328`: no certified win. `328` was the closest: its best candidate
  had `4.46e-6` residual penetration, but its minimum safe dilation still
  exceeded the target side by about `8.1e-5`.
- `791,804,848`: one new certified win at 791; 804 remained above its target
  side by `8.7e-3`, and 848 by `3.3e-3`.
- Focused `328,367` reruns with 40 starts and 20,000 repair iterations found
  no further win. Their best safe-dilated sides remained above the incumbent.

The final file contains exactly one complete 791-square group. An independent
coordinator-side load and official-metric check returned identical literal and
pandas sides above.

### Limits

The multi-donor barycenter is still a seed generator, not a feasibility
proof. Its useful regime is narrow: candidates with a residual near `1e-7`
can be closed by isotropic dilation and fixed-angle LP, while residuals at
`1e-6` or above consume too much side budget. The result also shows that
combining verified basins can strictly improve a prior win, but it does not
yet extend to the other top-gap targets.
