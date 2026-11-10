# Gap reachability audit

How much of the remaining `0.0267634584158375` score gap (current
`22.216763458415837` vs target `22.19`) is reachable, where, and under which
benchmark? This directory holds the data, the audit scripts, and the full
tables.

## Headline

| quantity | value | note |
|---|---:|---|
| current total | 22.2167634584158375109 | Decimal sum of the L strings |
| target | 22.19 | |
| gap | 0.0267634584158375109 | |
| public-table ceiling | **0** | every public per-n record is already absorbed |
| local-median envelope (positive) | **0.9379573464** | smoothing benchmark, not achievable |
| local envelope, n >= 100 | 0.1312115654 | |
| local envelope, n >= 600 | 0.0287717434 | ~107% of the gap |
| plausible local-repair range | **~0.008 - 0.016** | top-10 to top-30 high-n gaps |
| gallery re-polish residual | 0.0007237880 | 2.7% of the gap |

The public route contributes **nothing**: `merged-literal-official.json` is at
or below `baseline-official.json` for all 1000 n (195 strictly better, 805
tied, 0 worse). The remaining gap must come from packings that are better than
both tables, not from copying a public incumbent.

The local-median route is large but not credible as a target: 86% of its
positive mass sits at `n < 100`, where packings are small, often proved
optimal, and the steep, nonstationary `r(n)` curve makes a 10-neighbour median
a poor reference rather than a lower bound. Restricting to `n >= 600` leaves
0.0288, but capturing it would require closing essentially every high-n local
gap.

**Bottom line:** with targeted local repair, roughly 0.008-0.016 (30-60% of the
goal) is a defensible working range; the full 0.0268 requires closing the top
~100 high-n local gaps or a genuinely new construction family. No number here
is a certified gain.

## Definitions

For non-square `n`, with `k = floor(sqrt(n))`, `D = n - k*k`:

- `r(n) = L(n)/sqrt(n) - 1`
- `local_baseline(n) = median{ r(m) : m in [n-5, n+5], m != n, m not square }`
- `local_gap(n) = r(n) - local_baseline(n)`
- `public_gap(n) = max(0, r(n) - r_public(n))`
- `reachable_score(n) = max(0, local_gap(n))` (a benchmark, not a certificate)

Squares contribute exactly 0 and are excluded from the median. The total is the
sum over `n = 1..1000`; all totals are computed with `Decimal`, never floats.

The brief wrote `k = ceil(sqrt(n))`. That convention makes `D = n - k*k <= 0`
for every non-square, so "`D` near `k` (`D in [k-5, k]`)" and "`k*k+1` boundary"
would both be empty. This audit therefore uses the floor convention, which is
also the one used by `poc/research/docs/ARGON_BRIEF.md` and
`high_n_gap_ranking.md`, and reports `k_ceil`/`D_ceil` in the per-n table for
completeness.

"Public" here means `baseline-official.json`, the public23 merge. It is the
best public per-n table available in the repository, not an external leaderboard
snapshot.

## Regimes

Priority order (first match wins, so the classes partition all 1000 n):

1. **perfect square** (31) - `r = 0`.
2. **corrugation window** (28) - `|n - c| <= 3` for `c in {739, 794, 851, 910}`.
3. **k^2+1 boundary** (31) - `D = 1`.
4. **D near k** (165) - `k-5 <= D <= k`.
5. **heptagonal plateau** (395) - `n` lies in a run of at least 7 consecutive n
   with identical L (to 1e-9), i.e. the flat grid-ceiling shelves.
6. **general interior** (350) - everything else.

The repository does not define "heptagonal plateau" anywhere. The operational
definition above is this audit's reading: the label is applied to the long flat
`L` shelves (the 7-run at n = 30..36 is the smallest member of the family).
Changing the run threshold moves n between "heptagonal plateau" and "general
interior" but does not change any total, because both classes use the same
`local_gap` formula.

If the intended reading was instead the `k + 1/sqrt(2) ~= k + 0.707107` shelf
discussed in `high_n_gap_ranking.md` (the "707" shelves at n = 852, 911, ...),
that cohort is 29 n with a positive local gap of 0.126406. It is already a
subset of the classes above, so the audit's totals are unchanged; only the
regime label would move.

## What the numbers say

### Local gap is dominated by small n

| cohort | top 10 | top 20 | top 50 | top 100 | all positive |
|---|---:|---:|---:|---:|---:|
| all n | 0.614779 | 0.751658 | 0.838314 | 0.884020 | 0.937957 |
| n >= 100 | 0.025908 | 0.041112 | 0.069819 | 0.097274 | 0.131212 |
| n >= 600 | 0.007973 | 0.012644 | 0.020200 | 0.027140 | 0.028772 |

The overall top 100 captures 94% of the positive envelope; the top 30 are all
below `n = 150`, and by rank 100 (`n = 738`) the gap is already down to
`6.4e-4`. Those gaps are mostly curve curvature at small n plus proved-optimal
packings, so they are not scalable targets.

### Per-regime split

| regime | count | signed sum | positive sum | public sum |
|---|---:|---:|---:|---:|
| D near k | 165 | 0.448301 | 0.462129 | 0 |
| k^2+1 boundary | 31 | 0.371699 | 0.378657 | 0 |
| general interior | 350 | -0.530287 | 0.086718 | 0 |
| corrugation window | 28 | 0.004351 | 0.005341 | 0 |
| heptagonal plateau | 395 | -1.173913 | 0.005112 | 0 |
| perfect square | 31 | -1.389280 | 0.000000 | 0 |

`D near k` and `k^2+1` account for 90% of the positive local envelope, but most
of that is again at small n. The corrugation windows and the long plateaus are
already tight; they are not where a 0.027 gap lives.

### Gallery repair is not a path to 0.027

The 194 repaired gallery packings already bank 0.037264 of gain (their
`gain_score` sum) and currently contribute 3.590192 to the total. Their
`box_growth/sqrt(n)` residual, i.e. what a perfect re-polish back to the
original gallery `L` could recover, is **0.000723788**, about 2.7% of the
remaining gap. Re-polishing them cannot close 0.027; at best it removes
rounding from the repair.

## Ranked targets (top 30, n >= 100)

Ranked by positive local gap. Small-n targets (n < 100) are excluded because
they are dominated by proved-optimal or rigid packings and are not actionable
for new constructions; the unrestricted top 30 are all `n < 150` and are kept
in `docs/local_gap_positive_top.csv` for reference. "Public" says whether
`baseline-official` is tied or worse than the merged table at that n.

| # | n | k | D | regime | local gap | public | mechanism |
|---:|---:|---:|---:|---|---:|---|---|
| 1 | 105 | 10 | 5 | D near k | 0.003611 | tied | vacancy deletion + multi-angle cluster |
| 2 | 103 | 10 | 3 | general interior | 0.002939 | tied | off-center thick band, split angles, corner gadget |
| 3 | 127 | 11 | 6 | D near k | 0.002918 | tied | vacancy deletion + multi-angle cluster |
| 4 | 147 | 12 | 3 | general interior | 0.002913 | tied | off-center thick band, split angles, corner gadget |
| 5 | 151 | 12 | 7 | D near k | 0.002653 | tied | vacancy deletion + multi-angle cluster |
| 6 | 150 | 12 | 6 | general interior | 0.002320 | tied | off-center thick band, split angles, corner gadget |
| 7 | 175 | 13 | 6 | general interior | 0.002245 | public worse | off-center thick band, split angles, corner gadget |
| 8 | 106 | 10 | 6 | D near k | 0.002179 | tied | vacancy deletion + multi-angle cluster |
| 9 | 110 | 10 | 10 | D near k | 0.002172 | tied | vacancy deletion + multi-angle cluster |
| 10 | 125 | 11 | 4 | general interior | 0.001956 | tied | off-center thick band, split angles, corner gadget |
| 11 | 176 | 13 | 7 | general interior | 0.001822 | tied | off-center thick band, split angles, corner gadget |
| 12 | 290 | 17 | 1 | k^2+1 boundary | 0.001771 | tied | rational-slope (atan 1/2) corner pocket |
| 13 | 124 | 11 | 3 | general interior | 0.001657 | tied | off-center thick band, split angles, corner gadget |
| 14 | 325 | 18 | 1 | k^2+1 boundary | 0.001578 | tied | rational-slope (atan 1/2) corner pocket |
| 15 | 102 | 10 | 2 | general interior | 0.001547 | tied | off-center thick band, split angles, corner gadget |
| 16 | 362 | 19 | 1 | k^2+1 boundary | 0.001415 | tied | rational-slope (atan 1/2) corner pocket |
| 17 | 367 | 19 | 6 | general interior | 0.001374 | tied | off-center thick band, split angles, corner gadget |
| 18 | 491 | 22 | 7 | general interior | 0.001350 | tied | off-center thick band, split angles, corner gadget |
| 19 | 232 | 15 | 7 | general interior | 0.001346 | public worse | off-center thick band, split angles, corner gadget |
| 20 | 203 | 14 | 7 | general interior | 0.001345 | tied | off-center thick band, split angles, corner gadget |
| 21 | 295 | 17 | 6 | general interior | 0.001321 | tied | off-center thick band, split angles, corner gadget |
| 22 | 401 | 20 | 1 | k^2+1 boundary | 0.001276 | tied | rational-slope (atan 1/2) corner pocket |
| 23 | 264 | 16 | 8 | general interior | 0.001202 | tied | off-center thick band, split angles, corner gadget |
| 24 | 442 | 21 | 1 | k^2+1 boundary | 0.001157 | tied | rational-slope (atan 1/2) corner pocket |
| 25 | 234 | 15 | 9 | general interior | 0.001155 | tied | multi-angle cluster with independent sub-bands |
| 26 | 201 | 14 | 5 | general interior | 0.001147 | tied | off-center thick band, split angles, corner gadget |
| 27 | 485 | 22 | 1 | k^2+1 boundary | 0.001053 | tied | rational-slope (atan 1/2) corner pocket |
| 28 | 626 | 25 | 1 | k^2+1 boundary | 0.001038 | tied | rational-slope (atan 1/2) corner pocket |
| 29 | 183 | 13 | 14 | general interior | 0.001035 | tied | multi-block composite / defect-vacancy relaxation |
| 30 | 132 | 11 | 11 | D near k | 0.000997 | tied | vacancy deletion + multi-angle cluster |

Sum of the top 30: **0.052495** (196% of the gap), but these are median-relative
gaps, not certified saves. Seven of the 30 are `k^2+1` candidates, where the
known 45-degree / rational-slope pockets are the natural first attempt.

## Reachable-range estimate

- **Public ceiling: 0.** Already exhausted; there is no incumbent left to copy.
- **Local-median ceiling: 0.9380** overall, but 0.8067 of it is `n < 100`; the
  high-n envelope (`n >= 600`) is 0.0288.
- **Actionable band:** top-10 high-n gaps = 0.0080, top-30 = 0.0157, top-100 =
  0.0271. So a realistic local-repair campaign might recover 0.008-0.016, and
  reaching 0.0268 requires nearly the whole high-n envelope or a new family.
- **Re-polish:** 0.0007, negligible at this scale.

None of this is a proof of infeasibility, and none of it is a certified gain.
The audit's claim is narrower: the public table offers nothing, the local median
overstates reachability by ~35x, and the plausible local-repair range is below
the goal.

## Files

- `poc/analyze_gap.py` - rebuilds the per-n table, local/public gaps, rankings,
  regime split, top-k closures, and gallery residual; writes `docs/`.
- `poc/verify_score.py` - independent Decimal checks (score, literal vs pandas,
  public-gap zero, gallery residual). Currently 9/9 pass.
- `docs/per_n_table.csv` - all 1000 rows: `n, k, D, k_ceil, D_ceil, is_square,
  regime, plateau_run, L, r, local_baseline, local_gap, abs_local_gap,
  positive_local_gap, L_public, r_public, public_gap, public_status`.
- `docs/local_gap_abs_top.csv` - top 100 by absolute local gap.
- `docs/local_gap_positive_top.csv` - top 100 by positive local gap.
- `docs/public_gap_top.csv` - top by public gap (all zero; kept for the audit).
- `docs/targets_top30.csv` - actionable top 30 with mechanism text.
- `docs/regime_breakdown.csv` - per-regime counts and sums.
- `docs/summary.json` - all headline numbers in machine-readable form.
- `docs/full_tables.md` - the same tables rendered as Markdown.

Run from the repository root:

```sh
python3 poc/gap_reachability/poc/analyze_gap.py
python3 poc/gap_reachability/poc/verify_score.py
```

## Caveats

- `merged-literal-official.json` and `merged-pandas-official.json` differ in 196
  L strings at rounding level; their totals agree to `1.5e-16`. This audit uses
  the literal table, which is the one named in the brief.
- Summing the per-n `score` floats gives `22.21676345841584`, ~3e-15 above the
  L-string total; the L-string sum is the authoritative one.
- `public_gap` compares against our own public23 merge, not an external
  leaderboard snapshot. A new external record would not appear here.
- The local median excludes squares and self; near `n = 1` and `n = 1000` the
  window is truncated, which slightly changes the small-n baseline.
- Perfect squares appear in the per-n and absolute-gap tables for completeness.
  They contribute 0 to the score, are excluded from every baseline, and are
  excluded from the positive envelope and the target list.
- "Reachable score" is `max(0, local_gap)`. It is an upper bound on the
  median-relative improvement, not on what a construction can actually achieve.
