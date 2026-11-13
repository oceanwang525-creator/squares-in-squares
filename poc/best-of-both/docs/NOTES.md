# Notes

## What PR #5 adds

PR #5 computes multiple public-refresh rounds on top of `merged.csv`. Its 41
reported wins include `n=370, 598, 996`, which are already better in the local
dual-pocket union. Exact comparison leaves 37 apparent groups; 13 fail the
two-transport strict gate against the current group, leaving 24 accepted
replacements.

The useful rows are copied into `data/pr5-winning-rows.csv`. They are data only;
the script does not import or execute code from the PR branch.

## Merge gate

For each candidate group the script certifies:

1. the exact decimal strings as read from the CSV;
2. the same group after `pandas.read_csv` / `to_csv` transport.

The candidate replaces the local group only if both certified `L` values are
strictly lower. The assembled full file is then rescored from scratch on all
1,000 puzzles. The final gate rejects any per-`n` regression against either the
upstream literal report or the upstream pandas report.

## Negative context

The PR #5 whole-file score is worse than the local dual-pocket union. Its
utility here comes only from per-`n` union. The complementary gain is below
`1e-10` score, so this POC is not evidence that a new construction was found.

## Provenance

- Local rows: `poc/dual-pocket`, already merged in PR #4.
- Candidate rows: open PR #5 from `public_refresh_nov12` round 2.
- Incumbent: `poc/pell_corrugation/data/border-scan/merged.csv`.

Original authors retain credit. No source code from the PR is executed.
