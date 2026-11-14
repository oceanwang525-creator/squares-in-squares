# Hypothesis H75 (`lpslack`): Alternating Soft-Angle Dual-Contact Release SLP + Unbounded Fixed-Angle LP + Uniform Separating-Axis Gap Shrink

## Geometric Mechanism
1. **Why fixed-angle LP alone stalls**: Once `(x, y)` hits a 3-point contact lock at fixed square orientations `deg`, no further translation `(dx, dy)` can reduce `L` without tiny rotational relaxation `dtheta` on the bottleneck contacts.
2. **Alternating loop** (`poc/lpslack.py`):
   - **Selective Soft-Angle SLP**: Jointly optimizes `(dx, dy, dtheta, L)` with co-rotating separating axes (`u(theta + dtheta)`) and exact 4-sign branch representation (`±cos(d) ± sin(d)`) so angle kinks at `0°/90°` and parallel-square contacts are modeled without false zero-gradient slips. Unlocked sets progress across `dual_non90` (top-dual active non-0/90° squares), `non90`, and `all` with decaying angular trust regions `|dtheta| <= 5e-4 .. 5e-5`.
   - **Unbounded Fixed-Angle LP**: Immediately follows each SLP step with 8–12 rounds of no-trust-region fixed-angle `(dx, dy, L)` LP (`HiGHS`), allowing squares to slide arbitrarily far into newly opened pockets and repairing any second-order `O(dtheta * dx)` contact pinch.
   - **Uniform Separating-Axis Centre-Scale Shrink**: Exact 80-digit `Decimal` uniform centre-scale shrink (`0.98 * safe_ratio`) closes residual LP margin while preserving `>= 5e-13` separating-axis gap for `float64` pandas transport.
   - **Fail-Closed Certification**: Every candidate is certified in both `literal`, `final.csv` `pandas` transport, and `core.kaggle_roundtrip` against the full `origin/main` portfolio (`merged.csv` + all `poc/*/final.csv`).

## Gate Results (PR #1 — 6/6 Strict Certified Wins)
Target gate set: `n = 911, 740, 861` (starting from PR #60 `poc/postmerge_gap_refresh/final.csv`) and `n = 796, 853, 909` (skipped by PR #60 during rebase). All 6 groups strictly beat the full `origin/main` (`7283555`) portfolio in `literal`, `final.csv` `pandas` transport, and `core.kaggle_roundtrip`, reducing the competition score by **`7.812671e-7`** (`docs/verify_summary.json`):

| `n` | Incumbent `L` | New `L` (worst of literal/pandas/roundtrip) | `dL` | `ds = dL / sqrt(n)` |
|---:|---:|---:|---:|---:|
| 740 | `27.705900900583` | `27.705900674795` | `-2.257885e-7` | `-8.300152e-9` |
| 796 | `28.705888632492` | `28.705882380543` | `-6.251949e-6` | `-2.215945e-7` |
| 853 | `29.705885903026` | `29.705882480488` | `-3.422538e-6` | `-1.171855e-7` |
| 861 | `29.817035630831` | `29.817033793521` | `-1.837310e-6` | `-6.261536e-8` |
| 909 | `30.638834384281` | `30.638828600428` | `-5.783852e-6` | `-1.918383e-7` |
| 911 | `30.676989122549` | `30.676983697695` | `-5.424855e-6` | `-1.797335e-7` |

