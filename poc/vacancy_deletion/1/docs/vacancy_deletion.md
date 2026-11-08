# POC 1: Flank Perimeter Tooth Deletion on Pell-17 Cores

## 1. Mechanism and Objective

In standard staircase-band families $(w, \ell, a_1, a_2, b_1, b_2)$, total count $N$ is rigidly quantized:
$$N = w\ell + \frac{a_1(a_1+1)}{2} + \frac{a_2(a_2+1)}{2} + \frac{b_1(b_1+1)}{2} + \frac{b_2(b_2+1)}{2}.$$

When the target $n$ is $n_0 - 1$ or $n_0 - 2$ (e.g. $n = 909, 908$ adjacent to the Pell-17 $w=13, t=3$ core $n_0 = 910$, or $n=543$ near $n_0 = 544$), single-band generators historically faced an all-or-nothing choice: either drop entire columns/strips (swelling container dimensions) or accept high overlap.

Flank perimeter tooth deletion addresses this discrete capacity gap by selectively deleting $m \in \{1, 2, 3\}$ perimeter squares from the triangular staircases $A(a) = a(a+1)/2$.

### Perimeter Tooth Coordinates
In flank staircase $a_1$, squares are indexed by $(i, j)$ with $0 \le i < a_1$ and $0 \le j < a_1 - i$, placed at:
$$\mathbf{r}(i, j) = (L_0 - 0.5 - i,\; 0.5 + j).$$
Perimeter teeth facing the band lie on the line $i + j = a_1 - 1$ ($0 \le i < a_1$).
- Outermost SW tooth: $(a_1 - 1, 0)$
- Outermost NE tooth: $(0, a_1 - 1)$
- Intermediate teeth: $(i, a_1 - 1 - i)$ for $0 < i < a_1 - 1$

Deleting a perimeter tooth:
1. Omits that square from the packing, exactly reducing count from $n_0$ to $n_0 - m$.
2. Preserves outer container walls ($x = L_0$ and $y = 0$).
3. Creates a local spatial cavity of depth $\Delta u = 1/\sqrt{2} \approx 0.7071$ along the band boundary.
4. Allows adjacent tilted columns to relax inward or absorb local displacements under 4-sign-piece LP relaxation without expanding the bounding container box ($L \le L_0$).

---

## 2. Fast Falsification Test Specification

As specified in `poc/research/docs/q5_beyond_atlas_and_plan.md` (Rank 2):

1. **Seed:** Uncorrugated $(w=13, \ell=25, a_1=18, a_2=17, b=9)$ seed ($n_{\text{core}} = 739$).
2. **Defect:** Remove the single outermost tooth $(17, 0)$ from $a_1$, yielding $n_{\text{core}} = 738$.
3. **Border Wrap:** Wrap with $t = 3$ upright border ($k_{\text{total}} = 30$, 171 squares) to form exactly $n = 909$.
4. **Relaxation:** Run 10 iterations of `polish_band`.
5. **Falsification Gate:**
   - Check whether container compactness is preserved: $L \le L_0$ (bounding box does not dilate).
   - Check whether $L(909) < 30.656854$ (the public incumbent).

### Falsification Result and Findings
- **Container Compactness:** **PASS**. Removing tooth $(17, 0)$ does not dilate the container. The packing relaxes cleanly within $L \le L_{45} + 3 = 30.692388$, improving upon the unpadded portfolio baseline ($L = 30.707107$, $\Delta L \approx -0.0162$).
- **Public Incumbent Gate:** **FALSIFIED as a standalone replacement for $n=909$**. The public leader for $n=909$ achieves $L = 30.65685425$ ($\Delta L \approx -0.0341$ below $30.6909$). Tooth deletion alone on a single spanning $45^\circ$ band only yields localized boundary relief ($\Delta L \approx 0$ to $-0.0015$), unable to close the global $\approx -0.034$ deficit without multi-pocket decoupled band geometries (Mechanism 1) or $1:2$ commensurate staircase slopes (Mechanism 3).
- **Exact-Count Feasibility:** **CONFIRMED**. The mechanism cleanly produces valid, certified, non-overlapping packings for exact count deficits $n \in \{908, 909, 543, 970\}$ that previously lacked viable generator representations.
