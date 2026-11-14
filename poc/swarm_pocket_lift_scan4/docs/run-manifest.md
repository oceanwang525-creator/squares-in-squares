# Run manifest

## Baseline

| item | value |
|---|---|
| source | `upstream/merge:merged-22.15248431101.csv` |
| commit | `2714efd981190e83db228f1b0a4f300de4e4122a` |
| rows | 500500 |
| sha256 | `5c9bad10cc9a74147d4423ab2b134915156da1f7814474ca224d8c69ffd32db4` |
| official score | `22.15248431100757` |
| accepted / missing / rejected | `1000 / 0 / 0` |

## PR #72 donor overlay

Read-only source worktree: `/private/tmp/h128-masked-melt-pr.Osldqv` at
`7fd616632523bd9acbd4b5d115df20692bc03380`.

| file | groups | sha256 | overlay |
|---|---|---|---|
| `swarm_masked_melt_bootstrap/final.csv` | 634,695,861,926 | `bc4aff7d1775cef36529011c7f9d024ae6c4e69818a7bed90a58a041dc9a2a17` | n=926 from H128 |
| `swarm_masked_melt_gen3/final.csv` | 634,695,861 | `7491c1fc687443d3541863ee5605fe69813522474f448fbd0f86262432d53e77` | n=634,695,861 from H131 |
| `swarm_masked_melt_breadth/final.csv` | 442,485,530,577,588 | `18875c270167e72f866aa94dff110a7190fc4c3a784b1b1e367020ca3d62a345` | n=442,485,530,577,588 from H132 |

The resulting donor CSV has sha256
`61b48d270d8f90b478e96882b63312854a19023fbe7645edefce992a78d2f3f1`.
It is ignored by Git and never replaces the official baseline.

## Environment

| item | value |
|---|---|
| Python | 3.13.18 |
| NumPy | 2.5.4 from `/tmp/h94-venv` |
| pandas | 3.1.0 |
| concurrency | 8 independent scan processes, each with `--procs 1` |
| available CPU | up to 10 logical cores |

Every scan process sets `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`BLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`,
`VECLIB_MAXIMUM_THREADS=1`, `VECLIB_NUM_THREADS=1`, and
`KMP_NUM_THREADS=1`.
