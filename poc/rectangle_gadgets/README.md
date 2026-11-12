# Rectangle gadgets and the staircase composition rule (analysis; no packing built)

## Idea

The public fault-line DP builds most large-n packings by placing *gadgets* along an anti-diagonal staircase. A gadget (a, b, g, f) is an (a+f) × (b+f) box holding a·b + g unit squares. Its library is extracted from existing square packings. Rectangles optimised as standalone problems have never been searched, and a thin, cheap gadget could let more gain fit into one K + f box. Example: two (4, 10, 2) gadgets at f ≈ 0.6 would give n = 18² + 4 at 18.6, which is the Cats-and-boxes template at n = 328 / 365 (L − K = 0.59954; ours 0.614).

## Composition rule (derived and checked)

In a K + f box, the region above-left of the staircase is anchored to the top-left corner and the region below-right to the bottom-right corner. The seam between them is f wide, and each gadget straddles it from its bottom-left to its top-right corner. Counting seam turns gives n = K² + Σ g_i, provided

    Σ (a_i + 1) ≤ K + 1   and   Σ (b_i + 1) ≤ K + 1,

so every extra gadget costs one extra row and column. Checks: three 8×8 diamonds (gain 1, f = 0.53553) need 27 ≤ K + 1, i.e. K = 26, which matches the public notebook's n = 679. Two 5-square gadgets (2×2, f = 0.70711) would need K = 5, not 4, which is consistent with the known record s(18) = 4.8216.

## Measured

1. **Our portfolio is closed under square-gadget composition.** `poc/gadget_dp.py` treats every current n₀ = k₀² + j₀ (L − k₀ < 1) as a gadget (555 gadgets) and runs the 2D knapsack for every K ≤ 31. It reproduces our table exactly: 555 / 555 predicted cells equal, 0 better. Composition of existing packings has nothing left to give.
2. **Value map for new gadgets** (`poc/gadget_value.py`, `docs/value_map.md`). For each (a, b, g), the largest f at which adding that one gadget improves any n, against the hard lower bound f(b, b, g). Padding b − a columns turns an (a, b, g) gadget into a (b, b, g) one, so f(a, b, g) ≥ f(b, b, g). 176 realistic shapes (a ≥ 4) would pay if they reach f between the bound and the threshold, for example (4, 13, 3) at f ≤ 0.692, (4, 17, 2) at f ≤ 0.591, and (4, 10, 2) at f ≤ 0.650 (n = 229, 260, 293, 328, 365, 489).
3. **Search validation** (`poc/rgad.py`, random compress-and-relax in rectangular boxes, `docs/search_validation.md`). It reaches the records for n = 5, 7, 10 and 13 (f = 0.70711) but not for n = 17 (0.769 vs 0.6755) or n = 26 (no fit). From random starts it is too weak for the 40–80-square gadgets the value map asks for.

## Why it is negative for now

All valuable shapes are thin strips (width a + f with a = 4–7) that must gain 2–3 squares. `poc/strip_cell` shows a strip narrower than a + 1 cannot beat a squares per unit length, so all gain must come from the two ends. I found no construction, and the search here cannot reach these sizes. Also, the leaders beat us at j = 1 (n = K² + 1), where no composition happens, so most of their lead is better *large* gadgets, not better composition.

## Reuse

`gadget_dp.py` predicts the whole table from any gadget library in seconds. If anyone finds a new gadget (from any optimiser), add it as (a, b, g, f) and the DP lists every n it improves and by how much.
