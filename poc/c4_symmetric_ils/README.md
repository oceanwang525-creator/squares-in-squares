# C4-symmetric iterated local search (negative)

## Idea

A public post (Kaggle discussion 756468, G. D. da Fonseca) reports a new n = 109 record from "an iterated local search over overlapping packings with symmetry built in": a four-fold pinwheel. Enforcing C4 symmetry divides the number of free squares by four, so a global search might reach structures that full searches miss. Targets would be n ≡ 0 or 1 (mod 4) where a leader is ahead (210 such n; for example 228, 328, 333, 365, 485, 577).

## Method

`poc/c4ils.py`: one representative (x, y, θ) per orbit of four (plus a centre square if n ≡ 1 mod 4), copies by 90° rotations about the box centre. Energy = squared SAT penetration + squared wall violation (`poc/rgad.py`, analytic gradient, finite-difference error 1e-9), L-BFGS relaxations, kicks (global jitter, re-seating 1–3 orbits, angle kicks), random restarts. `poc/c4run.py` runs it at a fixed side on 8 cores.

## Result (negative)

Validation target: n = 109. Our tracked row is 10.949289581; Göbel's classical 45° packing is 10.949747.

- Compress from a loose box (start 11.6): every worker stopped at L = 11.000000 (the grid).
- Fixed side 10.9498 (Göbel's value), 8 workers × 240 s, about 6,000 relaxations: minimum overlap energy 9.7e-4. No feasible packing.

The search cannot rediscover even the classical packing at this size, so it was not run on the leader-gap targets. The author of the post also reports that the method improved no other n.

No packing was produced; merged.csv untouched.
