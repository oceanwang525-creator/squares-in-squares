# areebbb frontier-engine refresh (Nov 12 2026, 13:42 UTC)

The harvest watcher picked up a new kernel,
`areebbb/sis-frontier-targeted-certified-packing-engine`, whose current
`submission.csv` improves one n over the tracked file. `nilanjansaha123`'s
refreshed GPU-seam kernel carries the same row.

## Measured result

| check | score | accepted | missing | rejected |
|---|---:|---:|---:|---:|
| literal decimal strings | **22.173824282607068** | 1000 | 0 | 0 |
| actual pandas `read_csv` / `to_csv` | **22.17382428260707** | 1000 | 0 | 0 |

Previous incumbent: `22.173824282917252`.
Certified reduction: `3.101829810047406e-10`.

| n | score gain | old L | new certified L |
|---:|---:|---:|---:|
| 598 | `3.1018e-10` | `24.90316141262551762574947559668341603910535388845718` | `24.90316140504029105961564758866005185412997240625441` |

Merged file SHA256:
`94b50fe2c1e271f5f42b4890c2a2145871d5682bb2ee8755caf8499b0d825386`.

Method: harvest cycle → 32-process float prefilter → `merge_witnesses.py`
literal + actual-pandas gate → whole-file official literal and pandas
certification. No Kaggle submission was made.
