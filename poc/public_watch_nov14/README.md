# Public watch, Nov 14

## Idea

Same as `poc/public_watch_nov13`: re-list public Kaggle notebook and dataset outputs, re-scan those whose run time changed, and certify every candidate n. This is a fresh cycle against the Nov 14 team file.

## Method

The watcher scripts are in `poc/public_watch_nov13/poc` (`listk.py`, `merge_public.py`, `listd.py`, `merge_datasets.py`); candidates are certified per n with the official evaluator (literal and actual pandas round trip). `poc/make_final_watch.py` exports the rows that still beat the incumbent with the team converter (`poc/auto_union/poc/union.py: final_text`). Public files are read as data only, never executed.

## Result

**85 strict per-n wins against main `772dfc2`, sum of per-n score gains 2.6716e-3.**

| Public notebook (authors keep credit) | n |
|---|---:|
| `malyshevdanil/squares-in-squares-living-atlas-of-tight-packings` | 53 |
| `penguin069/squares-boxed50-certified-recomposition` | 27 |
| `prvsiyan/a-square-deal-research-to-verified-packings` | 3 |
| `penguin069/squares-gpu-v1-current-replay-data` | 2 |

The largest are n = 791 (6.4e-3 shorter in L), 969 (6.4e-3), 453 (5.4e-3), 758 (4.6e-3), 459 (4.5e-3) and 908 (3.8e-3).

| | literal | pandas |
|---|---|---|
| Incumbent `merged.csv` at `772dfc2`, certified here (1000 groups) | 22.150145089030410 | 22.150145089030413 |
| Groups in `final.csv` certified from its exact bytes | 85 / 85 valid, 85 strict wins | 85 / 85 valid, 85 strict wins |
| Full 1000-group union of main with this final and `poc/row_ratchet_980/final.csv` | 22.147423335171034, 1000 accepted, 0 missing, 0 rejected | 22.147423335171037, 1000 accepted, 0 missing, 0 rejected |

All 86 improved n are strictly below the incumbent in both modes in the full gate, with no regressions ([docs/full_gate.json](docs/full_gate.json)). Per-n values: [docs/wins.md](docs/wins.md).

Two more public rows (n = 648 and 739) certify in the radians format but become invalid after the degrees conversion, so they are left out. n = 980 is left out because `poc/row_ratchet_980` has a shorter row.
