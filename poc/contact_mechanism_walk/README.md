# Contact-mechanism walk

## Idea

A locally optimal square packing is jammed for a very specific reason: every
nearby contact removes one translational degree of freedom. Previous searches
perturb coordinates, rotate blocks, or solve an LP on a fixed contact graph.
Those moves generally violate many contacts at once and relax back into the
same basin.

This POC instead walks along a first-order **mechanism** of the contact graph.
The right singular vectors of the contact Jacobian with the smallest singular
values are coordinated `(dx, dy, dtheta)` motions that preserve the existing
jam to first order. Whole-layout rigid motions are projected out, then finite
steps explore new contact topologies before a soft-core re-jam and joint
`(x, y, theta, L)` SLP.

## Method

1. Load one incumbent `n` group.
2. Linearize active square-square separating axes and wall contacts, including
   separating-axis rotation derivatives.
3. Extract the smallest right singular modes and walk both signs at several
   amplitudes.
4. Re-jam with the compiled SAT L-BFGS engine and tighten jointly with
   generating SLP.
5. Export only strict per-`n` wins and certify both the literal and pandas
   transports.

`poc/mechanism_walk.py` supports target files, configurable walk amplitudes,
and a minimum mechanism variation. `poc/export_final.py` compares the private
portfolio with the latest merge artifact and emits only complete groups that
are strictly better in both certification modes.

## Result

Against `merged-22.15726293235.csv`, the current final leaves the recursive
`n=800` improvement. The complete candidate scores `22.15726292930414`, with
`1000/1000` groups accepted in both literal and pandas modes:

| n | merge L | final L | dL |
|---:|---:|---:|---:|
| 800 | 28.775369213891515 | 28.775369127667270 | -8.622424431e-08 |

The aggregate reduction is `-3.045860e-9`. The previous 12-group mechanism
result and the first recursive `n=800` result are included in the merge
artifact; this iteration supersedes its `n=800` row with a deeper recursive
walk.

The detailed machine-generated before/after report is in
`docs/final_export.json`.

## Negative results

- A 117-target neighboring-`n` scan around successful groups produced zero
  certified wins.
- Deep retries missed for `n=634` and `n=852`.
- High-mode, wide-amplitude retries missed for `n=791`, `908`, `911`, `969`,
  `972`, and `973`.
- Recursive amplitude retries missed for `861`, `546`, `419`, `416`, `421`,
  `703`, `649`, `507`, and `538`; the closest was `419` at `+2.455e-9`.
- A further fine/wide retry missed for `367`, `468`, `605`, and `612`; the
  nearest was `367` at `+9.20e-8`.
- A broad untested `700..1000` scan was interrupted after mostly zero or
  low-amplitude modes; it should not be repeated unchanged.
- The attached `submission-kaggle-safe.csv` was rejected as a donor: its
  official score was `23.279632614051454` and it contained zero groups
  strictly better than the then-current private portfolio.
