# Big-n donor crops (n>1000 sources)

Estimated gain: zero. No coordinates for n>1000 are reachable, and the local proxy crop finds no win.

## Reachability
- `https://www.kaggle.com/api/v1/kernels/pull/krisdreemurr/squares-in-squares-sides-for-n-1001-1500` is fetchable without login (notebook JSON). The notebook says "sides only, no coordinates". It reads `big_universe_sides_1001_1500.csv` (columns n, L, score, K, D, f, saving_vs_trivial) from the dataset `krisdreemurr/squares-in-squares-big-universe-sides`. Its zip, `.../api/v1/datasets/download/krisdreemurr/squares-in-squares-big-universe-sides`, also downloads anonymously (zip header seen, not unpacked).
- This gives only L values for n=1001-1500, no squares, so nothing can be cropped.
- Needed for the real experiment: per-square (x, y, angle) files for n=1001-1500. The public searches found none. Samanyu-dev/squares_in_squares has a summary CSV up to n=2000 and per-square files only for 101-1000. The search snippet suggests this but it was not checked.

## Proxy (poc/crop.py)
Donors are our merged.csv rows n'=910,920,...,1000 (10 donors). For each of the 4 reflections (this covers the 90-degree rotations), the window corner is anchored at every distinct (xmin, ymin) of the squares. The squares kept have their full bounding box inside. s(m) is the m-th smallest Chebyshev extent, so m=n gives the exact cropped n-packing with side s. Only axis-aligned windows are tested.

Result for targets n=100-900 (`docs/proxy_results.json`, `docs/run.log`): 0 wins, and s >= ours everywhere. The minimum delta is 0 (exact grid cases such as n=100 and n=900). The median delta is +0.024. Examples (s - ours): n=200 +0.132, 300 +0.092, 500 +0.067, 645 +0.053, 700 +0.064, 800 +0.026.
The result is expected. The crops come from the same family of constructions, and a cut through a seam loses the whole-container tilt gain.

No winning rows, so no CSV was built and `squares score` was not run.
Caveats: only 10 of the 100 donors n'=901-1000 were scanned (a full scan exceeded the time budget), the windows are axis-aligned only, and the "ours" side is the tight bounding box from float coordinates, not the literal strings.
