# Literature notes

## SQUISH

The October 2026 SQUISH reports describe a composition pipeline rather than a
single numerical optimizer:

1. cut known packings into sub-packings;
2. narrow each piece inside its own rectangle;
3. place two pieces in opposite corners;
4. fill whole rows and columns;
5. repair the seam;
6. run nearby search and deep squeeze.

The direct lesson is that the discrete seed topology carries most of the gain.
The reported final squeeze moves the October 11 layouts by at most `7e-7`.

## Structure-guided search

Levy's October 2026 survey and the later structure-guided preprint emphasize
that best-known packings are strongly organized by a small number of angle
classes. The preprint reports that roughly half of certified tilted squares
are at exactly 45 degrees and that most neighbouring tilted squares share an
angle class within 0.5 degrees. This argues for preserving measured pockets and
searching their interfaces, not for restarting every square at a free angle.

## Adjacent-wall strips

Karimi's October 2026 two-strip preprint constructs an axis-parallel grid plus
one tilted strip on the bottom wall and another on the right wall. Its proven
near-integer regime has deficit `O(sqrt(K))`, while our high-`n` incumbents are
already `K^2 + 8..12`, so the theorem itself does not cover them. The useful
structural point is the controlled placement of two boundary components with
an axis-parallel interior.

The structural preprint on best-known packings independently describes an
L-shaped grid with two adjacent tilted bands and studies the transition in the
tilted fraction. This supports testing the relative chirality of two measured
pockets instead of treating each as an isolated rectangle crop.

## Exact residual packing

Huang and Korf's absolute-placement approach fixes one coordinate family before
branching on the other, then completes the instance to a perfect packing.
Balas, Dumitrescu and Toth reduce anchored square packing to maximum-weight
independent set. Both support the current decomposition: once the moving tilted
components are fixed, the upright remainder is an exact finite MIS problem.

## Sources

- `https://squaremath.org/papers/square-packing-methods-survey.html`
- `https://github.com/itsnaka/squish-certs`
- `https://doi.org/10.5281/zenodo.23633300`
- `https://doi.org/10.20944/preprints202611.0242.v1`
- `https://doi.org/10.20944/preprints202610.2669.v1`
- `https://doi.org/10.1613/jair.3735`
- `https://arxiv.org/abs/1603.00060`
