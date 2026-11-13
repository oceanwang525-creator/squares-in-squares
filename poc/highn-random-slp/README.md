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
- `data/hard-target-wins.csv` - complete groups for the seven strict wins.
- `data/hard-target-report.json` - exact literal and pandas changes.
- `final.csv` - the same seven strict groups in the repository final format.

The seven strict wins are all numerically dominated by the deterministic
free-SLP result already recorded in `poc/highn-free-slp/final.csv`; they do not
change the team union on their own.
