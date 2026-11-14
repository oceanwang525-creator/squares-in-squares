## Summary

Reopens residual positive contact clearance in the latest `upstream/merge`
artifact, then runs the standard full-portfolio fixed-angle LP.

## Results

- Source: `upstream/merge` at `be377e1`, `merged-22.15330692498.csv`.
- Strict improved `n`: 81.
- Base certified score: `22.153306924975695`.
- Merged candidate score: `22.15330461280231`.
- Certified reduction: `2.3121733860875793e-6`.
- Literal certification: 1000/1000 accepted.
- Pandas transport certification: 1000/1000 accepted.

Improved groups:

`127, 232, 333, 372, 379, 401, 413, 416, 419, 421, 455, 459, 463, 464,
499, 500, 501, 507, 538, 540, 542, 546, 585, 587, 598, 632, 634, 639,
642, 646, 649, 652, 683, 686, 687, 692, 694, 695, 698, 700, 701, 703,
736, 739, 740, 741, 744, 789, 793, 796, 797, 798, 800, 803, 804, 811,
849, 850, 851, 852, 853, 854, 861, 864, 866, 875, 909, 910, 911, 913,
914, 915, 920, 922, 926, 936, 971, 972, 973, 978, 988`

## Validation

`poc/postmerge_gap_refresh/final.csv` contains only the 81 complete strict
groups. It has the required header `id,x,y,deg`, passed the literal final-CSV
structure check, and passed a fresh union-level certification in both modes.
Full numbers are under `poc/postmerge_gap_refresh/docs/`.
