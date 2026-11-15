# Localized donor weights

## Idea

The certified multi-donor transport win at `n=791` uses one global weight
between its donor rows. This probe lets that mixture vary by position. Donor A
can own one wall or contact region while donor B owns the opposite side,
without assigning one global ratio to every square.

The candidate remains a complete `n`-square group. Each target square first
receives a center and folded-angle barycenter from two Sinkhorn maps. A smooth
field over the square's normalized source position then reweights those maps
along the horizontal, vertical, or diagonal direction. The existing
fixed-side repair, minimum safe dilation, and fixed-angle LP are reused only
after the localized map is screened.

## Quick probe

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
PYTHONPATH=poc/baseline/poc/src $PY \
  poc/localized_donor_weights/poc/probe.py \
  --targets 791,803,328 --seconds-per-target 180 \
  --workers 3 --repair-iterations 12000 --top-starts 48
```

The probe treats the existing certified H16 `final.csv` row as the incumbent.
It exports only complete, strictly shorter groups accepted by both the
literal-decimal official metric and actual pandas `read_csv`/`to_csv`
transport. No public notebook or dataset coordinates are used.

## Results

### Certified win

The position-localized mixture improves the H16 winner at `n=791`:

| construction | L (literal = pandas) |
|---|---:|
| H16 incumbent | `28.626523948339571161...` |
| H21 localized weights | **`28.626213659843549455...`** |

The reduction is `dL=-3.102884960217e-4`, worth about `-1.103258882e-5`
in the official score. The winning state used donors `n+3` and `n+4`,
base weight `0.8`, field strength `2.5` along the `d+` diagonal,
Sinkhorn `epsilon=0.01`, target reduction `3e-4`, and interpolation
`alpha=0.02`. The repaired residual was `6.00e-8`; fixed-angle LP reduced
the safe-dilated side to the certified value.

The first implementation incorrectly compared only against canonical, so it
exported a 791 layout worse than the already certified H16 result. The
selection was corrected to load H16's complete 791 group as both a transport
source and an incumbent lower bound. A second issue that removed an equally
good POC row during reinitialization was also fixed; the winning run is
deterministic and was reproduced after the correction.

### Extended search

A second winning-source wave screened 45,000 starts and completed 80 long
repairs around the new geometry. It found no lower certified side: the best
safe-dilated candidate was `28.626261742...`, above the
`28.626213659...` incumbent. This closes the first extended tie-only round.

The final file contains exactly one complete 791-square group with SHA256
`0a563b5dae047d384ceba48c3d6566cf94bdcae80b7111883501d9d32c2eba3b`.
Both literal and actual pandas transport certify the displayed side.
