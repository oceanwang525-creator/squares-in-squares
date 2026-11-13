# Uniform shrink (dust, 18 strict wins)

## Idea

Some imported packings were dilated for safety, so every pair of neighbouring squares keeps a small positive gap. If the smallest separating-axis gap g_min over all pairs is positive, scaling every centre toward the bounding-box centre by s = 1 − δ (angles unchanged) is overlap-free while δ·|d| < gap for each pair, and it lowers L by about δ·(L − 1).

## Method

`poc/shrink_pass.py`:

1. For each n, compute the SAT gap of every pair with centre distance below 1.5 (numba) and r = min(gap / |d|).
2. If the minimum gap is at least 1e-12, set δ = f·r (f = 0.5 on the first pass, 0.8 on later passes; `SHRINK_FACTOR`).
3. Rewrite the centres exactly in decimal: x' = cx + s(x − cx), 25 decimals.
4. Certify each n with the official per-puzzle evaluator, literal and after a pandas 3.1.0 round trip, and keep it only if both L are strictly lower.

Three passes were run (rounds 2–4 of `poc/public_refresh_nov12`). Rounds 2 and 3 were merged in PR #5; this folder holds what is left of round 4 after the PR4/PR5 union.

## Result

On top of upstream `b57e78b` (`border-scan/merged.csv`, re-scored here: **22.173715450068453**, 1000/1000 literal and pandas):

| Transport | Score | Accepted | Missing | Rejected |
|---|---|---:|---:|---:|
| Literal | 22.17371545003316 | 1000 | 0 | 0 |
| After `pd.read_csv(...).to_csv(index=False)` | 22.17371545003316 | 1000 | 0 | 0 |

**18 strict wins** in both transports, **0 regressions**, score gain 3.53e-11. Improved n: 420, 421, 459, 463, 464, 507, 508, 510, 550, 602, 644, 697, 703, 744, 761, 874, 971, 988. Both L values per n: `docs/wins.md`. The regression check compares against my own re-score of upstream under the same `squares score` transport.

`data/improved_rows.csv` holds only the rows of those 18 n (same columns as `merged.csv`; the ids come from the full file); swap them per n. `docs/verification.txt` is the whole-file run output.

## Limits

Dust by construction. 946 of 999 non-trivial packings have a touching pair (gap < 1e-13), so they cannot shrink uniformly; the remaining ones lose about 80% of their clearance per pass, so a fourth pass would yield about 1e-11.
