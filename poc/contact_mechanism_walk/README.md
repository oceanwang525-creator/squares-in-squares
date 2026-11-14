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

Against `merged-22.15406754142.csv`, the current final leaves five strict
improvements from iterated mechanism passes over the merged structural seeds.
The complete score is `22.154065984536224` in both literal and pandas modes,
with all `1000/1000` groups accepted and no rejects or missing groups:

| n | merge L | final L | dL |
|---:|---:|---:|---:|
| 687 | 26.717920906484127 | 26.717919008812192 | -1.897671934e-06 |
| 736 | 27.634367487849832 | 27.634365368612369 | -2.119237463e-06 |
| 793 | 28.659309154049116 | 28.659281475945289 | -2.767810383e-05 |
| 851 | 29.666036666640738 | 29.666034291320278 | -2.375320463e-06 |
| 910 | 30.656437491823008 | 30.656427173119178 | -1.031870383e-05 |

The earlier recursive `n=800` result is now part of the upstream merge
artifact and is no longer exported. The `n=909` group was removed after the
new upstream merge improved that record. The aggregate certified reduction is
`-0.00000155688175292266026893196104488673560330283596292085054021504651867`.

A sixth-pass beam run on the exported records added
`-9.852243421950985e-8` at `n=687` and `-3.870653229518633e-7` at `n=736`.

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
- A targeted tilted/upright interface melt for polished `n=794` tested 55
  mixed and one-sided masks; its best certified result still missed by
  `+6.617691448e-7`.
- A second mechanism pass from the newly merged `n=972` seed missed by
  `+3.424031143e-6`.
- Third and fourth passes from the new `n=909` and `n=910` records missed,
  while a third pass from `n=851` added `-1.001444740e-6`.
- A further `n=800` pass missed by `+3.854627353e-7`.
- Stage-three searches missed for `n=803` (`+1.942898518e-5`), `n=685`
  (`+7.775807283e-6`), and `n=913` (`+6.850939862e-6`).
- A first search from the recovered `n=842` boundary packing came within
  `+2.822802410e-7`, but did not certify a strict win.
- The remaining `k^2+1` and nearby middle-band sweep missed for `962`, `901`,
  `785`, `730`, `677`, and `626`; the closest of these was `730` at
  `+1.311409144e-7`.
- Further middle-band searches missed for `742`, `691`, `758`, `690`, and
  `644`; the closest was `690` at `+8.986538134e-7`.
- A six-round mechanism pass from the updated `n=793` record missed by
  `+4.439093485672131e-5`.
- A broad untested `700..1000` scan was interrupted after mostly zero or
  low-amplitude modes; it should not be repeated unchanged.
- The attached `submission-kaggle-safe.csv` was rejected as a donor: its
  official score was `23.279632614051454` and it contained zero groups
  strictly better than the then-current private portfolio.
