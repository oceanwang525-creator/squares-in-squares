# Rectangle gadgets and the staircase composition rule (analysis; no packing built)

## Idea

The public fault-line DP builds most large-n packings by placing *gadgets* along an anti-diagonal staircase. A gadget (w, h, g, f) is a (w+f) × (h+f) pocket holding w·h + g unit squares. If our portfolio is not closed under composition of its own sub-gadgets, every missing combination is a free certified win; and if it is closed, a value map says which new gadgets would pay.

## Composition rule (derived, checked against the data)

In a K + f box the region below-left of the seam uses the grid anchored at the bottom-left corner and the region above-right uses the grid anchored at the top-right corner. Each pocket has its lower-left corner on the first grid and its upper-right corner on the second. Two consecutive pockets must be separated by one extra column **or** one extra row (not both), so m gadgets fit iff

    Σ w ≤ K,   Σ h ≤ K,   Σ (w + h + 1) ≤ 2K + 1,

and then n = K² + Σ g. Checks against tracked rows: three 8×8 diamonds (g = 1, f = 0.53553) need K ≥ 25, and n = 628 = 25² + 3 is at 25.535534 with 75 tilted squares; n = 908 is two (13, 16, +4) pockets with Σw = Σh = 29 at K = 30; two (2, 2, +1) pockets need K = 5 (n = 27 at 5.7071), not K = 4, consistent with s(18) = 4.8216.

**Correction:** the first version of this README stated the rule as Σ(w+1) ≤ K+1 and Σ(h+1) ≤ K+1. That is too strict (it would forbid n = 628). `poc/gaddp.py` uses the correct rule; `poc/gadget_dp.py` and `poc/gadget_value.py` still use the strict one.

## Measured

1. **Sub-gadget library** (`poc/gadlib.py`, `docs/gadget_types.txt`). For every tracked packing, the minimal clean-cut rectangle around each tilted cluster (and each pair of clusters): 892 gadgets, of which 337 are proper sub-rectangles of 95 distinct (w, h, g) types. Examples: (13, 16, +4) at f = 0.634071, (9, 15, +3) at 0.608713, (6, 10, +2) at 0.623065, (8, 11, +2) at 0.591378, (4, 5, +1) at 0.621320.
2. **Closure** (`poc/gaddp.py`). The corrected DP over all 892 gadgets and both orientations, K ≤ 31: **strictly better than the tracked table at 0 n.** Each cell is also its own gadget, so the DP can never be worse than the table; the informative result is that no combination of existing gadgets beats any tracked row. Composition of what we have is exhausted.
3. **Value map for new gadgets** (`poc/gadget_value.py`, `docs/value_map.md`). For each hypothetical (a, b, g), the largest f at which that one gadget would improve some n. It was computed with the strict rule, so it understates: any shape listed there pays under the correct rule too, and more shapes may. 176 realistic shapes (a ≥ 4) are listed, for example (4, 13, +3) at f ≤ 0.692 and (4, 10, +2) at f ≤ 0.650.
4. **Search validation** (`poc/rgad.py`, `docs/search_validation.md`). Random compress-and-relax in rectangular boxes reaches the records for n = 5, 7, 10, 13 (f = 0.70711) but not n = 17 (0.769 vs 0.6755) or n = 26. It is too weak for the 60–210-square gadgets that matter.

## Other findings from the same data

- The tracked rows use tilted blocks near 24–27° (slope ≈ 1/2), 18–19° (≈ 1/3), 33–35° and 41°, not only 45° (`poc/gad_extract.py` prints them). n = 908's pocket holds a 101-square block at 24.5°.
- Corrugation is not the missing mechanism: the corrugated Pell core at n = 739 (columns at ±0.173°) is only 2e-6 below the rigid band, and the better n = 738 row is a 24.8° block structure, not a 45° band.
- At K = 31 the frontier's wasted area L² − n is about 30 for every j = 1…12 (ours: 30.5–32.3). The frontier saves 1–2 unit squares of waste per packing; in gadget terms it has lower f for the same (w, h, g).

## Why it is negative for now

No new gadget was found. The useful targets are specific: a (13, 16, +4)-type pocket below f = 0.634 or a (9, 15, +3)-type below 0.6087 would lower many rows at once (j = 4, 8 at K ≥ 28; j = 3, 6 at K ≥ 25).

## Reuse

`gadlib.py` + `gaddp.py` re-check closure in about 10 seconds after any merge. If anyone finds a better gadget, add it to the library and the DP lists every n it improves.
