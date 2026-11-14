# Public notebook union

Read the ten best-ranked public **notebooks** returned by Kaggle for Squares in
Squares, ordered by `scoreAscending`. This is Kaggle's notebook ranking, not a
claim that every notebook's latest output matches its leaderboard submission.
Download each notebook's latest CSV outputs named `submission*.csv`,
`final*.csv` or `merged*.csv`, including nested output paths. Missing outputs,
private/unavailable notebooks and other schemas are recorded and skipped.
No notebook code is downloaded or executed. This tool has no submission API.

## Run

From the repository root, install the existing geometry dependencies and the
maintained Kaggle client in an isolated environment:

```sh
python -m pip install kaggle pandas numpy mpmath
# Use KAGGLE_API_TOKEN, or a private ~/.kaggle/kaggle.json, as supported by Kaggle.
# Never add credentials or downloaded API logs to this repository.
python poc/external/merge.py --work /tmp/squares-external-UNIQUE
```

`--work` must be a fresh directory outside the checkout. Default output is
`poc/external/final.csv`. The canonical baseline is read-only. Downloads,
per-group results, the fully certified union candidate and `audit.json` stay
in the work directory. Re-running requires a new work directory and fetches
the latest outputs again. Useful options are `--workers 8`,
`--group-seconds 90`, `--io-seconds 180` and `--limit 10` (maximum ten).

Both literal CSV and the actual whole-file pandas round-trip are parsed.
Incomplete/malformed groups are dropped independently. Exact parsed groups
are deduplicated; all admitted groups pass the official geometry certificate
in both modes. Independent worker processes are killed at the per-pair
deadline, rather than abandoning a still-running future. Invalid or timed-out
groups are omitted. Candidates replace an incumbent only on a strict win in
both modes. The actual public export is certified again after angle conversion,
then the full union must accept 1000/1000 in each mode with zero per-n
regressions. Output is replaced atomically only after these checks.

Treat the downloaded CSVs as untrusted data. The output SDK checks path
traversal; this script additionally constrains output paths, CSV sizes,
parsing time and worker concurrency. This is not an OS security sandbox.
Run without access to unrelated secrets. API/network and parsing errors are
recorded by exception type without persisting signed download URLs.

The audit contains notebook URLs, latest-run metadata, output SHA256s,
dropped groups and certified scores. A per-file full score is recorded only
when all 1000 groups passed both certificates, never extrapolated from a
partial set. `final.csv` contains only the certified improvements from this run.
The normal repository union workflow certifies it again when published.
There is deliberately no automatic Kaggle submission.

## Tests

```sh
python -m unittest discover -s poc/external -p 'test_*.py' -v
```

## Recorded run

Results and source details will be added after the first completed run.
