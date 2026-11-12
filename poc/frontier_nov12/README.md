# Frontier-engine dissection (Nov 12 2026)

`areebbb/sis-frontier-targeted-certified-packing-engine` publishes a per-n gap
table against a **newer frontier** than the Oct 29 snapshot. Its run log lists
64 public coordinate files (23 notebooks) and reports:

```
certified per-size best of all inputs: 22.17734693948
frontier (union of all teams' per-size bests): 22.050884
ours now: 22.177347   total gap to frontier: 0.126488
sizes where we beat the published frontier: [260, 507]
```

The frontier is therefore an all-team per-size envelope that is *not* the union
of the 64 public files; it includes private packings. The engine reaches
`22.177347`, which the current tracked file already beats (`22.173824`).

## New frontier targets (better than the Oct 29 table)

| n | frontier L | our L | gap |
|---:|---:|---:|---:|
| 973 | `31.6666382096706` | `31.69660926885852` | `0.0300` |
| 911 | `30.6609125615235` | `30.69562217094861` | `0.0347` |
| 910 | `30.6464577019472` | `30.67914946463598` | `0.0327` |
| 969 | `31.6055708985592` | `31.63398250210213` | `0.0284` |
| 791 | `28.6058845085404` | `28.63407075157900` | `0.0282` |
| 973/others | see `docs/per_n_gap.csv` | | |

## What was tested

All 23 notebooks (81 CSV outputs, 1.5 GB) were downloaded from the kernel
output API and scanned per-n against the current tracked file at a `1e-12`
prefilter. **0 candidate n** survived: the tracked union beats every public
coordinate file per-n.

Coordinates behind the `22.050884` frontier are not in any of the 64 files and
are not published by the engine. The remaining gap is therefore private
search quality, not a missing public source.

Files: `docs/per_n_gap.csv`, `docs/per_n.csv`, `docs/coordinate_sources.json`.
No CSV was written; `merged.csv` unchanged by this POC.
