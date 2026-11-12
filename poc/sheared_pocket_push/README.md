# Sheared-Row L-Pocket Push

Goal: Turn the zero-sawtooth sheared-row L-pocket lead into certified wins against the re-verified `merged.csv` incumbent (`22.173715450068453`) toward `22.0999`.

## Incumbent Baseline Verification

Verified on `poc/pell_corrugation/data/border-scan/merged.csv` via `poc/certify_merged.py` ([`incumbent_baseline.json`](docs/incumbent_baseline.json)):
- Official literal check: `score=22.173715450068453`, `accepted=1000`, `missing=0`, `rejected=0` (`506.6s`)
- Actual pandas `read_csv`/`to_csv` roundtrip: `score=22.173715450068453`, `accepted=1000`, `missing=0`, `rejected=0` (`252.0s`)

## Variant Log

| Variant | Target $n$ | Mechanism Added | Best Cap | Deficit | Best $L$ | Incumbent $L$ | Certified Win? | Artifact |
|---|---|---|---:|---:|---:|---:|:---:|---|
| `v0_poc1` | 634 | Rigid sheared L-pocket (`sw_upright`, fixed `th`, coarse grid) | 631 | 3 | `25.705107` (probe) | `25.707107` | No (lead) | [`poc1_sheared_pockets.json`](../hydrogen_brainstorm/docs/poc1_sheared_pockets.json) |
| `v1_continuous_angle` | 634 | Continuous pocket `th` (`dth in [-3,+3] deg`), `relax_tight` (`w=1e6`), `0.125` void fill + soft-insert | 631 | 3 | `25.705107` (probe) | `25.707107` | No | [`v1_continuous_angle.json`](docs/v1_continuous_angle.json) |
| `v2_end_taper` | 634 | Flush upright end-taper caps (`splice` & `reserve`), `u_cols in {0,1}`, `y_phase`, fan angles `0.35th/0.65th` | 629 | 5 | `25.705107` (probe) | `25.707107` | No | [`v2_end_taper.json`](docs/v2_end_taper.json) |
| `v3_miter_corner` | 634 | Pure & partial mitered SW corner (`diag_clip` & `45°` spine, `sw_u in {0,3,4,5}`) | 628 | 6 | `25.705107` (probe) | `25.707107` | No | [`v3_miter_corner.json`](docs/v3_miter_corner.json) |
| `v4_arm_phases` | 634 | Independent row widths `(w,u_cols)`, phases `y_phase`, x-pads, & partial corner taper per arm | 631 | 3 | `25.705107` (probe) | `25.707107` | No | [`v4_arm_phases.json`](docs/v4_arm_phases.json) |
| `v5_asymmetric_arms` | 634 | Asymmetric 2-arm (`Kx!=Ky`, `P1!=P2`) + 4-arm pinwheel fractional strip splits | 631 | 3 | `25.705107` (probe) | `25.707107` | No | [`v5_asymmetric_arms.json`](docs/v5_asymmetric_arms.json) |
| `v6_slp_squeeze` | 634 | Expanded-`L` sequential insertion (634 sq) + `squeeze_L` + joint `(dx,dy,dth,L)` SLP polish | 634 (`L=25.7599`) | 3 (at probe) | `25.759919` | `25.707107` | No | [`v6_slp_squeeze.json`](docs/v6_slp_squeeze.json) |
