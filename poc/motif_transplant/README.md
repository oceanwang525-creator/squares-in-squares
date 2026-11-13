# Motif transplant from leader-tied packings (negative)

## Idea

We tie the private per-n leaders at about 77 n, including two "Cats and boxes" packings (n = 535 and n = 308). If a leader packing were "corner-anchored upright grid + local tilted gadgets", each gadget could be cut out with its jammed fillers and dropped into a fresh K + c box. A gadget that gains g squares at excess c then gives every n = K² + g at side K + c, and gadgets in disjoint row and column bands add their gains. That would let leader motifs spread to n where we are behind.

## Method

`poc/motif.py`: mark off-lattice squares (tilted, or upright but not aligned with any of the four corner-anchored lattices), cluster them, fix them in place, and greedily refill the rest of the box with the four corner lattices (exact SAT, touching allowed). Self-test: the refill must rebuild the source packing's count before any transplant is meaningful (`poc/self_test.py`).

## Result (negative)

| n | source | off-lattice squares | clusters | rebuilt |
|---:|---|---:|---|---:|
| 65 | 5×5 diamond (ours) | 25 | [25] | 65 / 65 |
| 308 | Cats and boxes (tied) | 110 | [110] | 285 / 308 |
| 535 | Cats and boxes (tied) | 467 | [467] | 529 / 535 |

Our diamond-core packings are exactly "grid + motif" and rebuild perfectly. The leader packings are not. In n = 535, 467 of 535 squares are off the corner lattices and form one connected cluster: whole upright blocks sit at their own offsets, shifted against each other, with tilted bands in the seams (the corner-block + arm picture of `poc/arm_gadget`). Their gain cannot be isolated as a local motif, so a transplant has nothing self-contained to move. No CSV was produced.

## What this rules out

Local cut-and-paste of leader clusters into corner-lattice grids. Any transfer of the leaders' structure has to move whole shifted upright blocks plus their seams, which is the block-split problem already logged as negative in Loops 22 and 28.

Output: `docs/self_test.txt`.
