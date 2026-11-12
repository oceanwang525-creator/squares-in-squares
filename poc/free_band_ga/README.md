# Free-angle staircase-band population search (negative)

## Idea

The rational-band scans used discrete `p:q` slopes and never void-filled the
configurations. This POC does both:

1. **Void filling of every rational config** (`C>=880`, `L<30.72`): 20,408
   parameter sets, greedy box-constrained insertion at the deepest voids.
2. **Free-angle population search**: a GA over `(theta, T1, T2, w, gap,
   phase)` with an analytic BL/TR digital-staircase + tilted band count, run at
   the n=911 frontier side `L=30.6609126`, plus a 250k-sample fine scan.

## Measured result: negative

- Void filling: **0 wins**. Every rational config has voids of width `0.96`,
  just under one unit; no unit square fits (the earlier 2:1 donor is
  representative).
- Free-angle population search: best **898** squares with a band; fine scan
  reaches **900**, but that is the plain 30x30 grid (BL=899, TR=1, band=0).
- n=911 requires **911** squares at this side. The frontier packing is 11
  squares denser than any staircase-band configuration found.

So a from-scratch structured topology search at the frontier side cannot reach
n=911; the gap is not a parameter-tuning issue inside this family. `merged.csv`
unchanged.

Scripts: `poc/free_band_ga.py`. The void-fill scan is in the loop log.
