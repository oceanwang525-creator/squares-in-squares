# Pell-17 border scan, 2026-11-07

CPU-only bounded scan of all attempted Pell-17 widths with core count <=1000, upright L borders, four coordinate-prefix deletion orders and four partial-border prefix orders. Screened 423357 candidates; polished only the shortlist. This is not exhaustive over arbitrary subsets. Failed core constructions are recorded in cores.json rather than silently treated as feasible.

Four improvements against the public23 merged-material-candidate baseline survive exact transport certification: n=739, 794, 851 and 910. The n=910 winner reuses the committed Argon synthetic construction; the other three come from width 13 cores and borders. Detailed side lengths and construction metadata are in data/border-scan/wins.json.

Official whole-portfolio score, both literal and actual pandas 3.1.0 CSV read/write: 22.254062083924335. Baseline, both transports: 22.25406709809781. Improvement: 0.000005014173474648942. Every certificate accepts 1000 groups with zero missing/rejected. Independent C++ checks give 22.254062083924328 on both merged transports, differing only in floating-point score aggregation.

Artifacts retain the merged literal CSV, four individual winners, official and C++ certificates, core caches, screening metadata and final report. The pandas round trip can be regenerated from merged.csv; the baseline input SHA256 is pinned in the script. No Kaggle submission was made.

Run from the repository root with the shared squares package and numpy, scipy, pandas 3.1.0 installed. Set BASE in poc/pell_corrugation/poc/scan_borders.py to the matching public baseline and build the shared C++ certifier first. Then run that script. It caches cores under data/border-scan. Targeted tests after completion: 4 passed; prior shared plus Pell suite: 69 passed.
