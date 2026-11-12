# Construction search (negative, with reusable tools)

Goal: find layouts whose excess c = L − k for n = k² + j beats our plateaus, as the gallery leaders do (for example j = 1: leader 0.526 at k = 20 → 0.502 at k = 29–31, ours 0.5355 then 0.5208). Result: **no new constructions**. The tools rediscover our existing records from scratch, which is evidence the method works but this family is exhausted.

## What our packings are

Every n = k² + 1 for k = 8…23 in `merged.csv` is the same 5×5 block at 45° (25 tilted squares) plus grid staircases, wrapped outward with grid rows; c stays at (5√2 − 6)/2 = 0.5355339. From k = 25 the Pell-17 core (17×17 at 45°, 289 tilted) gives 0.5208153.

The leader's n = 535 (Cats and boxes; we tie it) is structurally different: a grid with **rectangular holes holding blocks tilted at 24.5° and 54.5°**, placed in disjoint row and column bands, with a few slightly tilted filler squares along the block edges.

## Tools (`poc/`)

| File | What it does |
|---|---|
| `bandgen.py` | Numba builder: tilted band or block + greedy fill from four corner-anchored grids, exact SAT with a touch tolerance |
| `rectscan.py` | Sweep of 45° m×p blocks on the diagonal (offset grid) |
| `rect45.py` | Exact model: 45° rectangle + four triangular staircases, minimum L by LP (rectangle corners inside the box), explicit build and SAT check |
| `critscan.py` | 45° m×p blocks at critical positions (faces touching corner-lattice corners) + greedy fill |
| `blockgen.py` | Stage 1: r×s block at any angle and position, greedy fill; keeps configs that gain squares over k² at a loose c |
| `stage2.py` | Stage 2: shrink-and-relax (`relax.py`, `opt_n.py`) of each stage-1 seed to its jammed c* |
| `compare_blocks.py` | Maps (gain g, size k, c*) to every n = K² + g with K ≥ k (grid wrapping) and compares with our L |
| `insert_search.py` | Seeds holding n − 1 at L below ours, one square inserted in the largest void, relaxed with free angles |
| `tiny_pass.py` | Certified tiny wins: deletion from n + k and grid-border wraps, border placed 5e-14 beyond 80-digit extremes, squares spaced 1 + 2e-14 apart so they certify after the pandas round trip |

## Measurements

- **Validation:** stage 2 rediscovers the known records from loose seeds: 3×3 → c* 0.621330 (record 0.62132), 3×5 → 0.598631 (0.59862), 5×5 → 0.535538 (0.5355339), 17×17 at 44.5° → 0.520847 (Pell-17 0.5208153).
- **45° families:** for n = 401 no 45° block (diagonal sweep, exact LP model, critical positions) holds 401 below L = 20.5355; the best is 400. For n = 842 the best below ours is 840.
- **Non-45° single blocks** (272 stage-1 seeds at c = 0.92, mostly 24–28°): best jammed c* 0.6317 for gain 1, 0.7639 for gain 2, all worse than ours.
- **Large near-45° blocks** (51 seeds up to 24×24 at c = 0.70): best c* per gain matches existing families (gain 1: 0.5208, 0.5355; gain 2: 0.5711; gain 3 and 5: 0.6067); 0 n improved (`block_search_results.txt`).
- **Insert-one** (n = 401, L = 20.53): residual penalty about 1e-2 after relaxation; nowhere near feasible.

## Tiny certified wins

`tiny_pass.py` found 30 strict wins (3 deletions, 27 wraps), score gain 2.49e-10. The largest are n = 733 and 487 (3.0e-9 in L each). Small, but all pass the official metric literally and after pandas.

## Open direction

Leader-style layouts need multi-block arrangements with tilted fillers at the block boundary (n = 535). Single blocks squeezed from loose seeds jam well above the records; a search that places two or more blocks with explicitly modelled filler rows, or a larger global optimizer, is the next step.
