# Public watch, Nov 12–13: periodic rescan of public outputs

## Idea

Public Kaggle notebooks now iterate on top of each other's unions, so each re-run can carry per-n wins. A watcher that re-lists notebooks and datasets about every 40 minutes, re-scans only outputs whose run time changed, and certifies every candidate n catches those wins within the hour instead of at the next manual harvest.

## Method

`poc/watch.sh` loops over:

1. `listk.py` – list the competition's notebooks through the public API (anonymous).
2. `merge_public.py` – stream every CSV output of notebooks whose run time changed; float prefilter per n; then the unchanged official per-puzzle evaluator on the literal strings and after an actual pandas round trip; keep a row only if both L are strictly below the best so far.
3. `listd.py` + `merge_datasets.py` – the same for public datasets, keyed by their last-updated time.

Public files are read as data only, never executed. Saved wins that are no longer below the incumbent are dropped on start, and downloads have a hard time limit.

`poc/make_finals.py` exports the surviving rows with the team converter (`poc/auto_union/poc/union.py: final_text`), and `poc/pr_gate.py` certifies them with the team gate functions (`load`, `check`, `baseline_checks`, `full_gate`) without writing `merged.csv`.

## Result

**85 strict per-n wins against main `252fe8b`, sum of per-n score gains 2.0725e-5.** All come from the public notebook `areebbb/sis-certified-union-micro-shrink-lb-22-17479`; its author keeps credit. The five largest (n = 639, 691, 745, 858, 918) are each 1.142e-4 shorter in L; three more are about 3.6e-8; the rest are a few 1e-9.

| | literal | pandas |
|---|---|---|
| Incumbent `merged.csv` at `252fe8b`, certified here (1000 groups) | 22.160524920480391 | 22.160524920480395 |
| Groups in `final.csv` certified from its exact bytes | 85 / 85 valid, 85 strict wins | 85 / 85 valid, 85 strict wins |

Per-n values: [docs/wins.md](docs/wins.md). The full 1000-group union gate (this final plus `poc/pocket_squeeze/final.csv`) is recorded in `docs/full_gate.json` once it has run; see the PR description for its status.

Two more public rows (n = 804 and 852) certified in the legacy radians format but become invalid after the degrees conversion of the final format, so they are left out.

### What the watcher saw overnight

Against the incumbent it started from (22.173715450, upstream `cd17a65`), the watcher's cumulative certified gain went from 2.0e-7 (40 n) at 20:06 MST to 2.1e-3 (76 n) at 23:15, when `jonathan1506/115-top-5-8-22-1757-fault-dp-fq-stretch` appeared, and to 1.32e-2 (146 n) at 07:06, when `malyshevdanil/squares-in-squares-living-atlas-of-tight-packings` re-ran ([docs/timeline.md](docs/timeline.md)). By the time of this PR the team had merged the same sources: 61 of those 146 n are matched or beaten on main, and the 85 above remain.

## Limits

This finds nothing the public does not already have. Its value is latency: wins are certified and ready within one cycle of a notebook re-running.
