# Exact set-packing and gadget search for n=367 (negative)

Target: the n=367 frontier side `L=19.6708567970218` (our row `19.6923881554`).

## 1. Global set-packing MILP

Candidates: the 19x19 upright grid (361) plus a dense margin band
(orientations 15..75 deg, 0.3 grid in the top/right 3.7-wide strip), giving
**8,075 candidates** and **1,128,770 exact SAT conflict edges**. Solved

```
max sum x_i   s.t.  x_i + x_j <= 1 for every overlapping pair, x binary
```

with HiGHS (`scipy.optimize.milp`), 300 s time limit. Result: **305 selected**
(time limit reached), far below the required 367; no certificate.

## 2. Corner-gadget construction search

If n=367 were a lift of an `m x m` corner gadget, it would need `m^2+6` squares
at side `m+0.6708568`. Torch penalty search for m=8 (70 squares at side
8.6708568, density 0.931), 16 seeds x 3000 steps: **no feasible packing**
(no seed reached the penetration/wall tolerance).

So the frontier's n=367 is neither a small corner gadget nor reachable by the
global independent-set MILP at this candidate resolution. 0 wins,
`merged.csv` unchanged.

Scripts: `poc/setpack367.py`, `poc/gadget_search.py`; raw run in `docs/`.
