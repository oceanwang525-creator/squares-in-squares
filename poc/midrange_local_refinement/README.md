# Local refinement for 100 to 133 squares

The incumbent is `poc/pell_corrugation/data/border-scan/merged.csv` from upstream
`2c1e6a6`, SHA-256
`d08670d4e517548f72ac10dc61fba5274028efacc9b305bffaf1cebc4f49557f`.
The target rows are identical to those in `cd17a65`, the snapshot used to
start the run.

| Method | Trials | Strict improvements |
| --- | ---: | ---: |
| Joint center and angle linearization on distinct seeds | 39 | 0 |
| Joint linearization with randomized separating-axis choices | 158 | 0 |
| Perturb a neighborhood or tilted squares, then minimize squared overlap and repair | 335 | 0 |
| Add one square to the preceding layout, then relax and repair | 297 | 0 |

The insertion pass included the first grid sizes in each block, 111 and 133.
Their best squared overlap penalties were 0.0034611 and 0.0018311 respectively
at a target box side 0.00001 below the incumbent. The penalty includes wall
violations. It is a search diagnostic, with feasibility decided by the official
evaluator. See [full counts and measured penalties](data/results.json).

Trial selection uses Float64 geometry and side thresholds of 1e-9 or 1e-8.
For any candidate that passes that filter, the scripts require strictly smaller
official side lengths for the literal rows and the pandas CSV round trip.
No result met the candidate filter, so this experiment produced no replacement
rows. The time budgets and local perturbations do not establish optimality.

## Reproduce

Use Python 3.12, NumPy, SciPy, Numba, pandas 3.1.0 and the baseline package in
`poc/baseline/poc`. Put the `sqk` toolkit from chaydavs/squares-in-squares
revision `ddb32bed3f3f1d34c2ab0067684504a4677bc509` on `PYTHONPATH` for
packing geometry, fixed-angle LP repair and export. Run from the repository root,
setting `seed` to the incumbent CSV path.

```sh
python poc/midrange_local_refinement/poc/polish.py "$seed"
python poc/midrange_local_refinement/poc/branches.py "$seed"
python poc/midrange_local_refinement/poc/patch.py "$seed"
python poc/midrange_local_refinement/poc/insert.py "$seed"
```

`polish.py` also accepts any number of additional seed CSV paths after the
incumbent. The measured pass included local seed files
`approved-optimized-n001-n100.csv`, `final_registry.csv`, `final_atlas.csv`,
`integration_candidate.csv`, `integration_transport.csv`, `slp_next.csv`,
`slp_checkpoint_next.csv` and `optnext_slp.csv`. The other passes read only
the incumbent. The randomized passes use per-size seeds and wall clock
budgets, so trial counts vary by machine.
