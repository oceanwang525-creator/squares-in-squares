# Sheared-Row L-Pocket Push

Goal: Turn the zero-sawtooth sheared-row L-pocket lead into certified wins against the re-verified `merged.csv` incumbent (`~22.173715`) toward `22.0999`.

## Variant Log

| Variant | Target $n$ | Mechanism Added | Best Cap | Deficit | Best $L$ | Incumbent $L$ | Certified Win? | Artifact |
|---|---|---|---:|---:|---:|---:|:---:|---|
| `v0_poc1` | 634 | Rigid sheared L-pocket (`sw_upright`, fixed `th`, coarse grid) | 631 | 3 | `25.705107` (probe) | `25.707107` | No (lead) | [`poc1_sheared_pockets.json`](../hydrogen_brainstorm/docs/poc1_sheared_pockets.json) |
| `v1_continuous_angle` | 634 | Continuous pocket `th` (`dth in [-3,+3] deg`), `relax_tight` (`w=1e6`), `0.125` void fill + soft-insert | 631 | 3 | `25.705107` (probe) | `25.707107` | No | [`v1_continuous_angle.json`](docs/v1_continuous_angle.json) |
