# High-n randomized free-angle SLP

## Idea

The high-n free-angle SLP is deterministic. Repeated trust-radius passes follow
the same contact graph and mostly replace each other's numerical dust. This POC
adds a seeded random linear objective on the angle variables inside each SLP
step. The perturbation selects different zero- or low-reduced-cost directions
while the hard separation and wall rows remain exact.

The noise decays geometrically over the iteration budget. Each candidate still
has to clear the unchanged literal and pandas official metric before it is
written.

## Files

- `poc/run.py` - parallel seeded runner around the collaborator SLP.
- `docs/NOTES.md` - experiment settings and measured results.

No `final.csv` is included unless a complete improved group is found.
