# Arm gadget: independent corner-arm repacking (negative)

## Idea

A large `K^2+D` packing can be cut at `x = w` and `y = a` into a bottom-left
upright block, a top-right upright block, and two arms of size
`w x (L-a)`. The arms are bounded by solid material, so shrinking `L` only
needs the top-right block to translate and the arms to repack in narrower
rectangles. This POC implements the arm LP with rectangular walls, a
free-angle feasibility SLP, and a provably safe Lipschitz SLP.

## Measured result: negative

Targets were the private leader sides for n = 973, 972, 911, 910, 908, 969 and
971. Every arm (and every whole packing) failed to become feasible:

- fixed-angle wall LP: infeasible at the target for all targets;
- free-angle feasibility SLP: infeasible for all targets;
- Lipschitz-safe SLP: infeasible for all targets;
- whole-packing Lipschitz SLP at the target side: infeasible for all seven;
- shrink continuation (0.05 steps, re-solving after each): n=973 stalls after
  0.06 of the required 1.0 arm compression.

## What the dissection found

The arms are not clean rectangles. Tilted squares poke through **holes in the
corner grid** at the interface: e.g. n=971 has seven protruding arm squares
(one by 0.336, six by 0.03-0.05) and the 24x24 top-right block has 574 squares
instead of 576. The interface is a corner-point zipper, so a flat-wall model
forbids the very contacts that make the packing dense. Rectangular clipping
survives 140-160 of the 167 arm squares, 6-9 short of the target counts.

## Artifacts

- `poc/arm_repack.py` - extraction, wall LP, SLP, Lipschitz SLP, assembly and
  whole-file certification.
- No CSV was written; `merged.csv` was unchanged by this POC.
