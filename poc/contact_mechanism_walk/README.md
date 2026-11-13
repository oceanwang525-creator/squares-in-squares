# Contact-mechanism walk

## Idea

A locally optimal square packing is jammed for a very specific reason: every
nearby contact removes one translational degree of freedom. Previous searches
perturb coordinates, rotate blocks, or solve an LP on a fixed contact graph.
Those moves generally violate many contacts at once and relax back into the
same basin.

This POC instead moves along a first-order **mechanism** of the contact graph.
For every active square-square separating axis and wall contact, the motion
must have zero relative normal velocity. The right singular vectors of that
contact Jacobian with the smallest singular values are coordinated
`(dx, dy, dtheta)` motions that preserve the existing jam to first order.
Whole-layout rigid motions are projected out before walking each mode. A finite
step can change which contacts are active while retaining the dense structure,
after which a soft-core re-jam and joint `(x, y, theta, L)` SLP can settle in a
new contact topology.

This is distinct from the logged random melts, axis-aligned fault shifts,
rigid-block translations, and near-tied separating-axis LP search. The new
ingredient is a deterministic whole-layout nullspace/soft-mode walk.

## Method

1. Load one incumbent `n` group from the tracked merged portfolio.
2. Build a sparse contact Jacobian from near-active separating axes and wall
   contacts, including rotation derivatives of both square extents and the
   selected separating axis.
3. Extract the smallest right singular modes with sparse SVD.
4. Walk both signs of each mode over several amplitudes and re-jam each seed at
   the incumbent side with the existing compiled SAT L-BFGS engine.
5. Tighten jointly in `(x, y, theta)` with generating SLP and certify only with
   the official literal and pandas transport checks.

## Initial targets

The pilot targets are `971`, `911`, and `367`, covering the top remaining
large-side gaps and the large `D=1` `45`-degree plateau. Scale-up is conditional
on a certified win or a small, structurally plausible residual gap.

## Result

Against the atlas-combined incumbent, `n=911` improves from
`L=30.689430606622263` to `L=30.686880320822803215...`, a certified
`dL=-2.550285799463e-3` after four successive mechanism walks. Both the
unchanged literal check and pandas `read_csv`/`to_csv` transport agree. The
`n=367`, `849`, `971`, and `973` pilots did not improve and are omitted.

`final.csv` contains only the complete winning `n=911` group.
