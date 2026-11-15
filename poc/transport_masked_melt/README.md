# Transport-seeded masked melt

## Idea

H7 and H16 found that Sinkhorn maps can move a jammed packing into a useful
new basin, but the final repair still changed every square. This probe keeps
the transport result as a topology and freezes its bulk. It releases only the
contact rings around the largest residual penetrations, alternating several
well-separated patches.

This is a hybrid of the verified transport seed and the masked-pocket melt
family. It is not a fresh global melt, a single-parent SLP, or a donor crop.

## Quick probe

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
PYTHONPATH=poc/baseline/poc/src $PY \
  poc/transport_masked_melt/poc/probe.py \
  --targets 803,328,367,969,861,791,908 \
  --repair-iterations 5000 \
  --out poc/transport_masked_melt/docs/results.json
```

The script rebuilds the best recorded H16 transport start for non-winning
targets and uses the current complete groups as controls for 791/908. It
retains only local updates that reduce the exact SAT penalty. Any candidate
must then pass minimal safe dilation, fixed-angle LP, literal certification,
and actual pandas `read_csv`/`to_csv` transport at a strictly smaller side.

## Results

**Quick probe: negative.** The local objective reduced the transport residual
by roughly one order of magnitude, but the remaining penetrations were spread
over hundreds of contact pairs and the required safe dilation moved every
candidate back above the target side.

| n | seed | local max overlap | local max wall | best safe side | result |
|---:|---|---:|---:|---:|---|
| 803 | transport | `2.15e-7` | `2.43e-7` | `28.8172711085` | above |
| 328 | transport | `1.95e-7` | `1.69e-7` | `18.6143715488` | above |
| 367 | transport | `2.45e-7` | `2.48e-7` | `19.6923912489` | above |
| 969 | transport | `2.30e-7` | `3.01e-7` | `31.6276421882` | above |
| 861 | transport | `1.78e-7` | `2.40e-7` | `29.8157690378` | above |
| 791 | H16 champion | `8.77e-8` | `8.47e-8` | `28.6265254758` | above |
| 908 | H7 champion | `1.36e-7` | `1.38e-7` | `30.6275384641` | above |

The target side for non-winning rows came from the recorded best H16
transport attempt. Existing 791/908 controls were probed at
`incumbent - 1e-6`. In every case the masked updates reduced SAT energy but
left a connected residual network large enough that safe dilation erased the
available reduction.

This closes the sampled local-variable interpretation. The useful transport
basins require either a different global topology or a coordinated variable
set larger than these contact rings. No complete winning group was produced;
`final.csv` is header-only.
