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
python -m pip install -e poc/baseline/poc kaggle numpy
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
are deduplicated. A cheap 65-digit bounding-side calculation ranks the groups
per n without running any geometry certificate. Only the lowest-L competitive
group for each n is certified first, prioritizing the largest potential wins
over main. Failure, timeout or lack of a strict dual-mode gain advances that n
to its next-ranked candidate. Unselected groups are not certified. Independent worker processes are killed at the per-pair
deadline, rather than abandoning a still-running future. Invalid or timed-out
groups trigger the next-ranked fallback for that n. Candidates replace an incumbent only on a strict win in
both modes. The actual public export is certified again after angle conversion,
then the full union must accept 1000/1000 in each mode with zero per-n
regressions. Output is replaced atomically only after these checks.

Treat the downloaded CSVs as untrusted data. The output SDK checks path
traversal; this script additionally constrains output paths, CSV sizes,
parsing time and worker concurrency. This is not an OS security sandbox.
Run without access to unrelated secrets. API/network and parsing errors are
recorded by exception type without persisting signed download URLs.

The audit contains notebook URLs, latest-run metadata, output SHA256s,
dropped groups and certified scores. Unselected groups are explicitly not certified. A per-file full score is
recorded only when all 1000 groups passed both certificates, never extrapolated
from a partial set. `final.csv` contains only the certified improvements from this run.
The normal repository union workflow certifies it again when published.
There is deliberately no automatic Kaggle submission.

An interrupted local run can be reused with --reuse-work /path/to/old-work.
Its download index pins the original notebook/rank mapping, and completed
certificates are reused only for matching exact parsed input keys. Only reuse
your own trusted work directories, not files supplied by third parties.

## Tests

```sh
python -m unittest discover -s poc/external -p 'test_*.py' -v
```

## Recorded run

November 14, 2026. Ten notebooks were queried; eight CSVs were downloaded
across seven notebooks (one CSV had a different schema and was skipped).
The merge-first run selected 58 per-n improvements over canonical
81fd9608405086b03d0d4751364c973263e11950. The full union passed literal and
actual pandas certification, 1000/1000 in each mode, with zero regressions.
The score moved from 22.146268736973266 to 22.146167653538928.
These are local certified union scores, not new Kaggle submissions.

Winning sources were [navneetprabhat's compilation](https://www.kaggle.com/code/navneetprabhat/squares-in-squares-compilation-by-instinct),
[penguin069's boxed51 union](https://www.kaggle.com/code/penguin069/squares-boxed51-certified-current-union),
and [alexchilton's starter kit](https://www.kaggle.com/code/alexchilton/squares-101-the-starter-kit-12-100-downloads).
See recorded-run.json for all ten source URLs, file hashes, per-n attribution,
and the exact baseline/candidate/final hashes. Unselected source groups were
not certified, so a full score for those notebooks is intentionally omitted.
The incremental final.csv SHA256 is
d5756fb8c9b4b64d03f53e0c8055c0c5fb06c5f7ed8d62409c82577ea2091756.
