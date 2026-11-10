# k2p10_two_bands (n=635, 686): negative / inconclusive, no packing built, no CSV written

Targets: n=635 L=25.69874 (ours 25.7071067812), n=686 L=26.68672 (ours 26.7071067812). Measured L for either n: ours only (unchanged); no candidate reached n squares at a lower L, so `squares score` was not run and merged.csv is untouched.

## Method
`poc/sweep.py`: MILP max independent set (reuses `poc/multi_angle_band/poc/mab.py`) over upright squares on the 4 corner-anchored lattices plus 45deg squares whose centres sit on a 0.1 (t,d) grid inside one or two width-1 windows (random offsets c1,c2, separation 3..8).

## Evidence
- Calibration (n=635, L=25.7071078): one 45deg w=1 band recovers 635 squares (incumbent reproduced, 35 s).
- One band at the target L=25.69874: 625 squares (= 25^2, grid only). Lattice-candidate bands cannot reproduce the incumbent's sliding/tip structure once L drops below L45, so a count >= 635 needs >= 10 more squares than the fallback.
- Two w=1 bands at L=25.69874: 6 configs finished (c1,c2 = (1.82,8.76), (3.17,7.44), (-4.39,2.85), (2.66,6.8), (4.03,-2.78), (-5.97,-0.75)); every MILP hit its 30 s limit with no incumbent solution (count 0 recorded, not a packing result). Cause: ~8k candidates per config on a machine at load average 70-120, so each config took 20-30 wall minutes; I stopped the sweep. See `docs/sweep635_two.log`. n=686 and the K=24..31 scaling were not run.

## Conclusion
Nothing measured supports two bands or band+pocket beating 625/635-class rigid lattice results; no K-scaling formulas were derived (any count/L formula for K=24..31 would be unmeasured speculation). Prior negatives (multi_angle_band: lattice-rigid phases cost ~0.15 in L; leader_poc_k2p10: polished incumbent is first-order rigid) point the same way. Estimated gain: 0. Remaining open: a sliding (non-lattice) two-band model with a proper LP/relaxation stage, and a band+pocket (atan(1/2)) variant like n=411; both need an uncontended machine and a faster overlap-edge builder than the pure-Python SAT in mab.py.
