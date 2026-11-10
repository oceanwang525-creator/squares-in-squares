# multi_angle_band

Off-center diagonal band split into two segments with independent tilt angles and lattice phases, upright fill from four corners, chosen by a MILP max-independent-set over lattice candidates.

Result: negative. Valid, official-metric-certified packings (literal and pandas) but all worse than the incumbent:

| n | incumbent L | this L | delta |
| --- | --- | --- | --- |
| 626 | 25.52081528 | 25.67081993 | +0.1500 |
| 634 | 25.70710678 | 25.85711197 | +0.1500 |

At the incumbent L the pool places only k^2 (626, 677, 730, 785) or n-5 (634) squares, i.e. nothing better than the grid. Estimated score gain: 0. The top-5 excess gaps sum to about 4.5e-3 in ratio units, which is only a ceiling on the ratio that could be recovered, not a gain. Not worth building as lattice-rigid bands; rigid phases cost about 0.15 in L versus the Pell-17 sliding incumbents.

Files: poc/rank_targets.py (re-ranking, in memory), poc/mab.py + poc/search.py (generator/search), poc/certify.py (official check of one n), docs/notes.md (top 10 and raw counts), docs/*.log, docs/*.npy (saved packings). No CSV written; merged.csv untouched.
Run: `python poc/search.py <n> <L_inc> <seconds> <dL>` then `python poc/certify.py <n> <dL>`.
