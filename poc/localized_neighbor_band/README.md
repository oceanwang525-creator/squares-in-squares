# Localized neighbor-band transport

## Idea

The H16/H21 transport work shows that a Sinkhorn map can move a jammed row
into a useful nearby basin, and that varying donor weight by position can
improve a row further. This probe applies that mechanism to unresolved rows
adjacent to the strongest transport winners, rather than remapping only 791.

Each target starts from the best complete group currently available for that
`n` (canonical row or any certified `poc/*/final.csv`). Two donor maps are
combined with a smooth horizontal, vertical, diagonal, or quadrant weight
field. The resulting complete target is repaired, minimally dilated, tightened
with fixed-angle LP, and accepted only if both official transports are
strictly below the effective incumbent.

The quick gate targets `790,792,907` and includes the known winners as
controls. A second wave may cover `793,909,969,970,973` only if the first
probe gives a meaningful near miss or win.

## Run

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
PYTHONPATH=poc/baseline/poc/src $PY \
  poc/localized_neighbor_band/poc/probe.py \
  --targets 790,792,907 \
  --seconds-per-target 180 --workers 3 \
  --repair-iterations 10000 --top-starts 32 \
  --out poc/localized_neighbor_band/docs/results.json
```

No public notebook or dataset coordinates are used. `final.csv` contains only
complete groups that passed the strict literal-decimal and actual pandas
`read_csv`/`to_csv` certification gates.

## Results

**Quick probe: negative.** I tested the first three unresolved rows adjacent
to the strongest transport winners. Each row screened 27,000 localized-weight
starts, repaired the best 32 with 10,000 L-BFGS iterations, and applied the
minimum safe dilation plus fixed-angle LP gate where possible.

| n | incumbent L | best repaired violation | best safe L | result |
|---:|---:|---:|---:|---|
| 790 | `28.608712578447843` | `5.36e-5` | `28.610251725199510` | above |
| 792 | `28.637029401753864` | `4.63e-5` | `28.638239091526785` | above |
| 907 | `30.618366064843970` | `3.23e-5` | `30.619346492536312` | above |

The local field changes which wall/source donor supplies each region, but the
repair leaves distributed overlap at `1e-5` scale rather than an isolated
seam defect. Safe dilation consequently costs `1.0e-3..1.5e-3`, far beyond
the available side reduction. None of the targets reached the one-square or
`1e-9` near-feasibility gate that would justify extending to the remaining
neighbors.

`final.csv` is header-only because no complete candidate passed both official
transport modes at a strictly smaller side.
