# Ruled-out variants inside the wildcard

These were tested or reasoned about inside this head; all are dominated for the
same boundary-waste reason unless noted.

1. **Pure rigid pinwheel tiling (`a+b` square).** When axis-aligned, four
   `a x b` blocks plus the central `(a-b)^2` square exactly tile `(a+b)^2`; that
   is the trivial grid, so its score is the grid score. Introducing tilt only
   enlarges the axis-aligned bounding square. No gain.

2. **A single common tilt (4-fold).** Measured: at every target the best
   capacity-matched member has `q = 0`. Tilted members are strictly worse
   (`L` higher by 0.3–0.5 at the tested sizes), e.g. `n=420` best 21.531 vs
   tilted 22.001.

3. **`side` family (`(dx,dy)=(d,0)`).** Scanned alongside `corner`; its minimum
   `L` for each capacity is no better than `corner`, and for the targets the
   winners are all `corner`. Discarded on measurement.

4. **Filling with a dense upright grid and letting the blocks sit in the
   corners.** This is what the best (untilted) member effectively is. It still
   loses to the plain grid because the four blocks occupy area that the grid
   would have used more tightly; the corner margins are wasted. Measured `L`
   exceeds `ceil(sqrt(n))` by `0.53 / 0.90 / 0.69` at `n = 420 / 650 / 930`.

5. **Larger blocks to lower relative seam waste.** Waste fraction falls as
   `a,b` grow (`~0.954` density at `L ~ 39`), but a larger block forces a
   larger `L`, so the capacity cannot be aimed at `n <= 1000`. At the scales
   where `n` lives the waste is ~10%, far above the ~4.6–5.7% area budget.

6. **Exact staircase seams.** Making the tilt rational (`tan theta = q/p`) lets
   the seam be described by a lattice staircase, but the sawtooth gap per step
   is irreducible unless the fill is itself tilted, which is just a rotated grid
   with no advantage. Not pursued past the argument because the bounding box
   already dominates measured results.

## Not covered

- Per-corner *different* angles (option b, corner molecule) and non-4-fold
  asymmetric interlocking were not scanned. The negative result here is specific
  to the 4-fold rigid pinwheel. An asymmetric molecule could in principle behave
  differently, but the boundary-waste budget above suggests it would need a
  construction that keeps waste under ~4.6% at `n=420` and ~2.8% at `n=930`.
