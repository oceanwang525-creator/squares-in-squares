# Contact-mechanism walk

## Idea

A locally optimal square packing is jammed for a very specific reason: every
nearby contact removes one translational degree of freedom. Previous searches
perturb coordinates, rotate blocks, or solve an LP on a fixed contact graph.
Those moves generally violate many contacts at once and relax back into the
same basin.

This POC instead moves along a first-order **mechanism** of the contact graph.
For every active square-square separating axis and wall contact, the motion
must have zero relative normal velocity. The right singular vectors of that
contact Jacobian with the smallest singular values are coordinated
`(dx, dy, dtheta)` motions that preserve the existing jam to first order.
Whole-layout rigid motions are projected out before walking each mode. A finite
step can change which contacts are active while retaining the dense structure,
after which a soft-core re-jam and joint `(x, y, theta, L)` SLP can settle in a
new contact topology.

## Method

1. Load one incumbent `n` group from the current merged portfolio.
2. Build a sparse contact Jacobian from near-active separating axes and wall
   contacts, including rotation derivatives of both square extents and the
   selected separating axis.
3. Extract the smallest right singular modes with sparse SVD.
4. Walk both signs of each mode over configurable amplitudes and re-jam each
   seed at the incumbent side with the compiled SAT L-BFGS engine.
5. Tighten jointly in `(x, y, theta)` with generating SLP.
6. Certify the result with both the official literal check and pandas
   `read_csv`/`to_csv` transport.

`poc/mechanism_walk.py` supports target files, configurable walk amplitudes,
and a minimum mechanism variation. `poc/export_final.py` compares the private
portfolio with the latest merge artifact and emits only complete groups that
are strictly better in both certification modes.

## Result

Against `merged-22.15743398692.csv`, the current final leaves 12 complete
groups. The official full gate scores the merge at `22.157433986920000` and
the candidate at `22.157430702854302`, both with `1000/1000` accepted in the
literal and pandas transports. The exact aggregate improvement is
`dscore=-0.000003284065699`. Concurrent upstream wins superseded several
earlier mechanism results; those groups were removed rather than carried as
snapshots.

Largest surviving wins:

| n | merge L | final L | dL |
|---:|---:|---:|---:|
| 546 | 23.839879448679131 | 23.839850580426873 | -2.886825226e-05 |
| 538 | 23.705915748949025 | 23.705902523574277 | -1.322537475e-05 |
| 416 | 20.874007458659960 | 20.873999394231944 | -8.064428016e-06 |
| 419 | 20.934136845493981 | 20.934129634819833 | -7.210674148e-06 |
| 800 | 28.775374198878872 | 28.775369620969101 | -4.577909770e-06 |

All exported groups:

| n | merge L | final L | dL |
|---:|---:|---:|---:|
| 127 | 11.801754813640645 | 11.801754205547709 | -6.080929361e-07 |
| 379 | 19.929313984349374 | 19.929313326048337 | -6.583010362e-07 |
| 413 | 20.816986590428140 | 20.816984597803071 | -1.992625067e-06 |
| 416 | 20.874007458659960 | 20.873999394231944 | -8.064428016e-06 |
| 419 | 20.934136845493981 | 20.934129634819833 | -7.210674148e-06 |
| 421 | 20.973694368556853 | 20.973690701748247 | -3.666808606e-06 |
| 507 | 22.960235978241307 | 22.960234570297324 | -1.407943984e-06 |
| 538 | 23.705915748949025 | 23.705902523574277 | -1.322537475e-05 |
| 546 | 23.839879448679131 | 23.839850580426873 | -2.886825226e-05 |
| 649 | 25.911771162144994 | 25.911769787089387 | -1.375055607e-06 |
| 703 | 26.940185788938305 | 26.940183458822116 | -2.330116191e-06 |
| 800 | 28.775374198878872 | 28.775369620969101 | -4.577909770e-06 |

The detailed machine-generated before/after report is in
`docs/final_export.json`.

## Negative results

- A 117-target neighboring-`n` scan around successful groups produced zero
  certified wins.
- Deep retries missed for `n=634` and `n=852`.
- High-mode, wide-amplitude retries missed for `n=791`, `908`, `911`, `969`,
  `972`, and `973`.
- A broad untested `700..1000` scan was interrupted after mostly zero or
  low-amplitude modes; it should not be repeated unchanged.
- The attached `submission-kaggle-safe.csv` was rejected as a donor: its
  official score was `23.279632614051454` and it contained zero groups
  strictly better than the then-current private portfolio.
