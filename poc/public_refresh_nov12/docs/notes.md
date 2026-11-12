# Notes

## Gap to the per-n leaders

The public gallery index (https://squares-in-squares-gallery.vercel.app/data/index.json, fetched Nov 12) gives each n's leader score through `status`/`gap`. Using leader score = gallery score − gap for `behind` rows and the gallery score for `tied` rows, the leader envelope is about **22.051**. After this refresh we are behind on about 430 n (table: `leader_gaps.csv`).

| n range | our total score gap to leaders |
|---|---:|
| 1–100 | 0.000 |
| 101–300 | 0.006 |
| 301–600 | 0.039 |
| 601–1000 | 0.083 |

"Cats and boxes" leads 340 n. In the excess c = L − k for n = k² + j, our values form plateaus (a core wrapped with grid borders keeps c fixed as k grows), while the leaders' c keeps falling with k, for example j = 1: 0.536 (k = 19) → 0.502 (k = 29–31), against our 0.536 → 0.521. Closing that needs new constructions whose excess shrinks with k, not polishing.

## Optimizer (negative)

`poc/relax.py`: energy = Σ SAT penetration² over neighbour pairs + Σ wall violation² of all corners, analytic gradient (finite-difference check, relative error 1.5e-8), L-BFGS-B, cell-list neighbours. Sanity: from random starts it finds n = 10 feasible at L = 3.72 (best side 3.71997) in 3 of 100 starts.

`poc/opt_n.py`: from the incumbent, shrink the box by δ ∈ {3e-3 … 1e-5}, apply no jiggle, a global jiggle or a local-region jiggle, and relax. On n = 269, 328, 333, 365, 367 and 379, about 300 relaxations in 150 s per n gave no feasible configuration at any δ: the incumbents are rigid. This matches `neighbour_deletion`, `freeze_relax_momentum` and the A100 runs in the Argon brief.

## Transport

All accepted packings keep the source's decimal strings. The full-file pandas round trip gives the same score as the literal file (22.176602112206332), so no n relies on literal-only digits.
