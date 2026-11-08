# Vacancy Deletion POC 2: Band-Column End Deletion and Vacancy Collapse

## 1. Executive Summary

Vacancy Deletion POC 2 tests whether deleting 1–2 squares directly from the diagonal band columns of the certified $n_0 = 739$ Pell-17 core ($w=13, l=25, a_1=18, a_2=17, b_1=9, b_2=9$) allows adjacent columns and flank staircases to collapse inward under 4-sign-piece epigraph LP relaxation.

**Result: Falsification Gate Failed ($\Delta L > 0$).**
- For target $n=909$ ($t=3$ border, 1 square deleted from core): polished $L \approx 30.690921$, incumbent $L = 30.65685425 \implies \Delta L \approx +0.034067$.
- For target $n=971$ ($t=4$ border, 1 square deleted from core + 233 border): polished $L \approx 31.690921$, incumbent $L = 31.67877988 \implies \Delta L \approx +0.012141$.
- In all tested configurations (outermost columns $c=0, c=24$ and central pinch columns $c=6, 11, 13, 18$), band-column end deletion produces local intra-column relaxation ($\Delta v \approx 0.25 - 0.50$) but **zero global container shrinkage** beyond the base corrugation limit of the intact columns.

---

## 2. Geometry and Construction

### 2.1 The Pell-17 Certified Core ($n_0 = 739$)
The intact core comprises:
- Flank staircase $a_1 = 18$ (SE): $a_1(a_1+1)/2 = 171$ squares
- Flank staircase $a_2 = 17$ (NW): $a_2(a_2+1)/2 = 153$ squares
- End staircase $b_1 = 9$ (SW): $b_1(b_1+1)/2 = 45$ squares
- End staircase $b_2 = 9$ (NE): $b_2(b_2+1)/2 = 45$ squares
- Diagonal band: $l = 25$ columns of $w = 13$ squares each ($25 \times 13 = 325$ squares)
- Total squares: $171 + 153 + 45 + 45 + 325 = 739$.

### 2.2 Band-Column Deletion Candidates
In each column $c \in \{0, \dots, 24\}$, squares are indexed by $s \in \{0, \dots, 12\}$:
- $s = 0$: negative $v$-end, abutting SE flank staircase $a_1$
- $s = 12$: positive $v$-end, abutting NW flank staircase $a_2$

Candidate deletion sets:
1. **Outermost SW Column ($c=0$):**
   - Candidate 1: Delete $s=0$ (SE end, $n_{\text{core}} = 738$)
   - Candidate 2: Delete $s=12$ (NW end, $n_{\text{core}} = 738$)
   - Candidate 3: Delete both $s=0$ and $s=12$ ($n_{\text{core}} = 737$)
2. **Outermost NE Column ($c=24$):**
   - Candidate 4: Delete $s=0$ (SE end, $n_{\text{core}} = 738$)
   - Candidate 5: Delete $s=12$ (NW end, $n_{\text{core}} = 738$)
3. **Central Pinch Columns ($c \in \{6, 11, 13, 18\}$):**
   - Minimal tooth offset $\max(|du_{\text{se}}|, |du_{\text{nw}}|) \approx 0.3588$ at $c=6, 18$.
   - Candidate 6: Delete $s=0$ (facing SE tooth cavity)
   - Candidate 7: Delete $s=12$ (facing NW tooth cavity)

### 2.3 Vacancy Collapse Seeding and LP Epigraph Polish
Upon deletion:
1. The remaining $w-1 = 12$ squares in column $c$ receive an initial translation vector into the vacant cavity ($\Delta v \approx \pm 0.25$).
2. The entire assembly (core + border) is relaxed using the 4-sign-piece epigraph LP polisher (`polish_band` from `poc/pell_corrugation`), which solves:
   $$\min L \quad \text{subject to } \mathbf{a}_{ij}^\top (\mathbf{x}_j - \mathbf{x}_i) \ge h_i + h_j, \quad 0 \le x_i \pm h_i \le L, \quad 0 \le y_i \pm h_i \le L$$
   with iterative trust-region updates on positions and angles.

---

## 3. Experimental Results

| Configuration | Target $n$ | Core Count | Border $t$ | Polished $L$ | Incumbent $L$ | Delta $\Delta L$ | Result |
|---|---|---|---|---|---|---|---|
| `outer_c0_s0_del1` | 909 | 738 | 3 | 30.690921 | 30.656854 | +0.034067 | Fail (held by intact band) |
| `outer_c0_s12_del1` | 909 | 738 | 3 | 30.690921 | 30.656854 | +0.034067 | Fail (held by intact band) |
| `outer_c24_s0_del1` | 909 | 738 | 3 | 30.690921 | 30.656854 | +0.034067 | Fail (held by intact band) |
| `outer_c24_s12_del1` | 909 | 738 | 3 | 30.690921 | 30.656854 | +0.034067 | Fail (held by intact band) |
| `outer_c0_both_ends_del2` | 908 | 737 | 3 | 30.690921 | 30.637030 | +0.053891 | Fail (held by intact band) |
| `pinch_min_s0_del1` | 909 | 738 | 3 | 30.690921 | 30.656854 | +0.034067 | Fail (held by intact band) |
| `outer_c0_s0_del1` | 971 | 738 | 4 (+1) | 31.690921 | 31.678780 | +0.012141 | Fail (held by intact band) |
| `pinch_min_s0_del1` | 971 | 738 | 4 (+1) | 31.690921 | 31.678780 | +0.012141 | Fail (held by intact band) |

---

## 4. Rigorous Geometric Blocker Analysis

Why does band-column end deletion fail to shrink the container?

### 4.1 Multi-Column Transverse Jamming
The container side length $L$ is simultaneously bounded by:
1. The vertical extent between horizontal walls $y=0$ and $y=L$, spanned by staircases $a_1$ and $a_2$.
2. The transverse clearance between flank staircases $a_1$ and $a_2$.
In the Pell-17 core, there are $l = 25$ columns running parallel to each other.
Deleting 1 square from column $c$ creates a void of dimension $1.0 \times 1.0$ *within column $c$ alone*.
However, the remaining $24$ intact columns still have length $w = 13$ squares. These 24 columns form an uninterrupted, rigid bridge spanning between staircase $a_1$ and staircase $a_2$.
Because the 24 intact columns cannot compress along $v$, flank staircases $a_1$ and $a_2$ cannot move closer together. Consequently, the bounding container $L$ remains completely pinned at $L = 27.690921 + t$.

### 4.2 Comparison with the Public $n=971$ Defect Relaxation
The public leader packing at $n=971$ ($L = 31.67877988$) was reverse-engineered as a defect relaxation of $n=972$. However, its underlying strip architecture is fundamentally different from a uniform 25-column band:
- In public $n=972$, the tilted band has only **13 strips with heterogeneous lengths**: 5 columns of length 13 and 8 columns of length 21 (total 233 tilted squares).
- Because the columns have staggered lengths and step-like terminations, removing an end square eliminates the *sole* contact point holding an entire sub-block against the flank staircase.
- In contrast, the Pell-17 core has **25 uniform columns of length 13**. Removing a single square leaves 24 other columns in full rigid contact.

### 4.3 Conclusion
Isolated band-column end deletion on a uniform-column corrugated core cannot achieve global container shrinkage. To reproduce the large shrinkage observed in public $n=971$, the vacancy must be introduced into a heterogeneous multi-block strip system where the deleted square is a critical kinematic pinning point for the entire staircase boundary.
