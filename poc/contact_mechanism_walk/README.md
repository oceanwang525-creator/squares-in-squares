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

Against `merged-22.15798556090.csv`, the current final leaves 20 complete
groups. The official full gate scores the merge at `22.157985560897423` and
the candidate at `22.157842218704012`, both with `1000/1000` accepted in the
literal and pandas transports. The exact aggregate improvement is
`dscore=-0.000143342193411`. Concurrent upstream wins superseded several
earlier mechanism results; those groups were removed rather than carried as
snapshots.

Largest surviving wins:

| n | merge L | final L | dL |
|---:|---:|---:|---:|
| 464 | 21.983258973386352 | 21.982347885575468 | -9.110878108832e-04 |
| 546 | 23.840418393704002 | 23.839879448679131 | -5.389450248731e-04 |
| 914 | 30.731743778947809 | 30.731219928218067 | -5.238507297425e-04 |
| 421 | 20.974089413444670 | 20.973694368556853 | -3.950448878172e-04 |
| 419 | 20.934328138732688 | 20.934136845493981 | -1.912932387068e-04 |

All exported groups:

| n | merge L | final L | dL |
|---:|---:|---:|---:|
| 127 | 11.801765635216046 | 11.801754813640645 | -1.082157540177e-05 |
| 379 | 19.929490009302835 | 19.929313984349374 | -1.760249534607e-04 |
| 413 | 20.817055467704847 | 20.816986590428140 | -6.887727670683e-05 |
| 416 | 20.874183681027365 | 20.874007458659960 | -1.762223674032e-04 |
| 419 | 20.934328138732688 | 20.934136845493981 | -1.912932387068e-04 |
| 421 | 20.974089413444670 | 20.973694368556853 | -3.950448878172e-04 |
| 455 | 21.818238253567714 | 21.818178713270857 | -5.954029685623e-05 |
| 464 | 21.983258973386352 | 21.982347885575468 | -9.110878108832e-04 |
| 507 | 22.960284457622258 | 22.960235978241307 | -4.847938095126e-05 |
| 538 | 23.705917541320218 | 23.705915748949025 | -1.792371193052e-06 |
| 546 | 23.840418393704002 | 23.839879448679131 | -5.389450248731e-04 |
| 598 | 24.902521130029744 | 24.902517565903821 | -3.564125919528e-06 |
| 646 | 25.868826171030857 | 25.868726075740724 | -1.000952901329e-04 |
| 649 | 25.911782311432241 | 25.911771162144994 | -1.114928724655e-05 |
| 652 | 25.960802857353574 | 25.960795524493935 | -7.332859638283e-06 |
| 695 | 26.831790880142655 | 26.831786338182223 | -4.541960435634e-06 |
| 703 | 26.940185998202882 | 26.940185788938305 | -2.092645752136e-07 |
| 800 | 28.775374244599160 | 28.775374198878872 | -4.572029129089e-08 |
| 852 | 29.699131372754820 | 29.699131029032021 | -3.437228002909e-07 |
| 914 | 30.731743778947809 | 30.731219928218067 | -5.238507297425e-04 |

The detailed machine-generated before/after report is in
`docs/final_export.json`.

## Negative results

- A 117-target neighboring-`n` scan around successful groups produced zero
  certified wins.
- Deep retries missed for `n=634` and `n=852`.
- A broad untested `700..1000` scan was interrupted after mostly zero or
  low-amplitude modes; it should not be repeated unchanged.
- The attached `submission-kaggle-safe.csv` was rejected as a donor: its
  official score was `23.279632614051454` and it contained zero groups
  strictly better than the then-current private portfolio.
