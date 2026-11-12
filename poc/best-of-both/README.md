# Best-of-both certified union

This POC combines two independently certified results:

- the merged `poc/dual-pocket/data/submission.csv` contribution, which keeps
  28 strict per-`n` wins over the team incumbent;
- the winning rows from open PR #5, `public_refresh_nov12` round 2.

The merge is done per puzzle. A PR #5 group is accepted only when both its
literal and pandas-roundtrip `L` are strictly below the current dual-pocket
group. The full resulting 1,000-row file is then checked against both official
transport reports.

## Measured result

PR #5 round 3 contains 41 strict wins over the team incumbent. `n=370, 598,
996` and other rows are already better in the local dual-pocket union, while
additional candidates fail the two-transport strict gate against the current
group. **24** groups are accepted. The final full file scores:

- literal: `22.173715449468716`
- pandas round trip: `22.173715449468716`
- accepted: `1000/1000`
- strict wins over the team incumbent: `52`
- strict regressions against the current incumbent: `0`

The PR #5 complement is dust-scale on top of the local union. Most of its value
is already covered by `n=370, 598, 996` and other local wins.

## Reproduce

The repository does not commit the complete 500,500-row output. It commits the
41 candidate puzzle groups and a script that rebuilds the full file from the
current local union.

```sh
python poc/best-of-both/poc/merge_best.py \
  --incumbent poc/dual-pocket/data/submission.csv \
  --candidate-rows poc/best-of-both/data/pr5-winning-rows.csv \
  --output /tmp/best-of-both.csv
```

The script writes `data/changed-rows.csv`, `data/summary.json` and both complete
official reports under `docs/`.

No Kaggle submission is performed.
