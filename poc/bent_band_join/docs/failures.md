# Failures and measured shortfalls

## 1. The `k + 0.79` claim is not reproduced

The brief predicts, for `n = k*k + D` with `D` near `k`, a bent band at about
`k + 0.79` versus a monolithic band at `k + 0.86..0.93`. Taking `n = 645`
(`k = 25`) the prediction is `L ≈ 25.79` against the incumbent `25.870318`,
i.e. `dL ≈ -0.080` and per-n score `+0.00315`.

Measured (certified, see `README.md`): `L = 26.192389`, `dL = +0.32207`,
per-n score `-0.01268`. For `751`, `925`, `987` the measured `dL` is
`+0.242`, `+0.572`, `+0.155`. All four are worse; total `+0.04524` score.

## 2. Where the capacity actually goes

The monolithic family is only 4–5 squares short of each target at the incumbent
side (max capacity at `L <= L_incumbent`: 640/746/921/982 vs 645/751/925/987).
So the whole contest is: add 4–5 squares at the same side length. The bent
builder tested here instead *loses* 35–45 squares at that side (606/710/881/942),
because its two small blocks replace efficient corner staircases with a
low-density corner fill.

## 3. Why an exact two-angle staircase seam does not exist

A square lattice `L(t) = Z u + Z v`, `u = (cos t, sin t)`, `v = (-sin t, cos t)`,
contains a direction `w` only if `w` is an integer combination of `u, v`. For
two rotations `t1 != t2` (mod 90 degrees) the intersection `L(t1) ∩ L(t2)` is
generically `{0}`. A seam staircase made of whole unit cells must lie in both
lattices, so no exact seam exists between two differently tilted blocks.

Consequences seen in the code and the earlier `poc/hinged_band` result:

* the two blocks either overlap by a small wedge (removed here, costing
  capacity) or leave a wedge that upright squares cannot fill at density 1;
* the freed wedge is exactly the `4–5` squares the claim needs, which is why the
  approximation loses more than it gains.

This is a structural argument, not a proof that no bent band can ever beat the
monolithic one; a construction that bends in *position* rather than *angle*
(e.g. a stepped staircase with a wider mid-channel) is not covered by it.

## 4. Things that were tried and their outcomes

* Exact monolithic 45-degree family, full bounded search for exact `n`:
  certifies at `L` above every incumbent (see README evidence table).
* Max capacity at the incumbent side: short by 4–5 squares; a 5th band row
  cannot be placed without breaking containment, so widening the band does not
  recover the gap.
* Bent two-block builder over five rational angles (`45`, `arctan(1/2)`,
  `arctan(2/3)`, `arctan(3/2)`, `30` degrees), widths 2–6, lengths 10–40
  (2401 configurations per target): best configurations collapse to
  `t1 = t2 = 45` with small blocks; capacity below both `n` and the monolithic
  max capacity.
* Generic numeric bent-band prototypes (two 45-degree-family blocks from
  opposite corners, four corner-anchored upright phases) reached only ~560–600
  at `L ≈ 25.86` for `n = 645`, confirming the loss above.

## 5. Not tried / open

* A stepped-staircase band (wider mid-channel, `a1/a2` changing with position)
  which the structural argument does not rule out.
* An exact seam built from a common *sublattice* of two Pythagorean-angle
  lattices (possible only for special angle pairs and special widths).
* Any annealing or continuous relaxation (out of scope: bounded discrete search
  only, CPU, ~30 min budget).

## 6. Data provenance and caveats

* Case page `https://squaremath.org/cases/303.html` was unreachable; geometry
  follows the written brief.
* The monolithic family may not be the same object as the brief's "monolithic
  tilted band"; the brief's `k + 0.86..0.93` is better than this family's
  exact-capacity sides for `D` near `k`, so the brief's baseline likely uses a
  different (probably multi-angle) structure.
* The max-capacity table is a bounded-search upper bound; the true family
  maximum could be lower, never higher, so it cannot rescue the claim.
