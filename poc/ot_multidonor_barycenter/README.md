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
| first H16 barycenter | `28.627434433035796153...` | `28.627434433035796153...` |
| extended H16 winner | **`28.626523948339571162...`** | **`28.626523948339571162...`** |

Relative to upstream, `dL = -1.111810191976e-3` and the official score gain
is `-3.953141948372e-5`. Relative to H7 alone, the extended H16 result
contributes another `dL = -1.009798917634e-3`, worth `-3.590759997784e-5`
in score.

The final winning schedule mixed donors `n+3` and `n+4` at weights `1:0.85`
with Sinkhorn `epsilon=0.0125`, interpolation `alpha=0.04`, and a requested
side reduction of `2e-4`. Its repaired residual was `6.22e-8`; fixed-angle LP
reduced the safe-dilated side to the certified value.

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

## Extended follow-up

The successful row was scaled in four focused waves rather than stopped after
the first win:

| wave | screened starts | repaired starts | best certified side |
|---|---:|---:|---:|
| broad donor/weight sweep | 19,656 | 80 | `28.627361454798546571...` |
| higher repair / larger deletion | 2,592 | 64 | `28.627035026608082978...` |
| winner-local extrapolation | 1,050 | 72 | **`28.626523948339571162...`** |
| next extended tie round | 1,050 | 53 | no improvement |

The first three waves each found a strict improvement. The fourth reproduced
the same reduced basin and produced no lower certified side, so the focused
791 extension reached an empirical stop.

A six-row neighborhood wave over `789,790,792,793,804,848` screened 211,680
starts and completed 269 repair runs at `300` seconds and 12,000 iterations
per target. It produced no certified win. Together with the 10-minute final
791 tie round, this is the extended no-win evidence required before stopping
this family.

The final `final.csv` SHA256 is
`c827c04baab6b81c42aca716ea2f37b7b9d66e69331c24ce6818fd7216490b08`.
Its only group contains exactly 791 squares and certifies identically in
literal and actual pandas transport.

Reproduce the final winning wave with:

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
PYTHONPATH=poc/baseline/poc/src $PY \
  poc/ot_multidonor_barycenter/poc/probe.py \
  --targets 791 --seconds-per-target 300 \
  --repair-iterations 30000 --top-starts 72 --workers 1 \
  --donor-offsets 3,4 \
  --epsilons 0.01,0.0125,0.014,0.016,0.018 \
  --alphas 0.025,0.032,0.04,0.05,0.06 \
  --deltas 3e-4,4e-4,5e-4,6e-4,8e-4,1e-3,1.3e-3 \
  --weight-ratios 1:0.6,1:0.7,1:0.8,1:0.85,1:0.9,1:1 \
  --no-triples \
  --out poc/ot_multidonor_barycenter/docs/results-extend-791-wave3.json
```

### Limits

The multi-donor barycenter is still a seed generator, not a feasibility
proof. Its useful regime is narrow: candidates with a residual near `1e-7`
can be closed by isotropic dilation and fixed-angle LP, while residuals at
`1e-6` or above consume too much side budget. The result also shows that
combining verified basins can strictly improve a prior win, but it does not
yet extend to the other top-gap targets.
