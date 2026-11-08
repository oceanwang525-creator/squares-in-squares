# Anatomical Dissection of Public Leading Packings at $n = 908, 971, 972$ and Large-Gap Neighbors

**Author:** Milo (Droppy Code / Hydra)  
**Date:** November 2026  
**Context:** Investigation into public leaderboard packings, structural deficits of single-band $45^\circ$ staircase families, and blueprint for multi-block non-$45^\circ$ generators.

---

## 1. Executive Summary & Diagnostic Ledger

Analysis of the official public records reveals that the remaining deficit (~88.25% of our gap versus the public per-$n$ table) is concentrated in the interval $(k^2 - n)/k \in [1, 2]$, where $k = \lceil\sqrt{n}\rceil$. In this regime, the baseline upright grid ($k \times k$) is underfilled by $k$ to $2k$ squares, necessitating tilted insertions.

Our current synthetic family—the single-band $45^\circ$ staircase band with Pell-17 pinch-column corrugation—assumes a single monolithic diagonal strip cutting entirely across the box from corner to corner. While this yields certified wins at $n = 910$ ($L = 30.69092107$), it stalls at $n = 908$ and $n = 971$.

| $n$ | Public Incumbent $L$ | Our Certified $L$ | $\Delta L$ (Ours - Public) | Public Topology & Mechanism | Our Topology | Primary Deficit Mechanism |
|---|---|---|---|---|---|---|
| **908** | `30.6370295596` | `30.6421356830` | `+0.0051061234` | Dual decoupled NW/SE pockets, $\theta \approx 24.1^\circ-26.6^\circ$, column profile $(5, 7, 8, 8, 9, 8, 8, 9, 9, 8, 8, 8, 5)$ | Single $w=20$ band at $45^\circ$ + 1 border | Single-band diagonal inflation; 45° tooth waste vs 2:1 rational slope |
| **910** | `30.6909601206` | `30.6909210742` | **`-0.0000390465`** | $w=13$ band at $45^\circ$ + 3-layer L-border | Pell-17 out-of-phase pinch-column corrugated band + 3 border | **Ours wins** via 4-sign-piece epigraph LP angle splitting |
| **971** | `31.6787798753` | `31.6909212021` | `+0.0121413268` | Asymmetric vacancy defect relaxation of 13-strip band ($\Delta L = -0.0136$ below $L_{45}+4$) | Pell-17 corrugated core ($n_0=739$) + 4 border | Single-band pinch corrugation gains only $-0.00147$; lacks cell vacancy relaxation |
| **972** | `31.6923881554` | `31.7071068426` | `+0.0147186872` | 739 upright (SW side-21 triangular staircase + NE block) + 233 tilted ($w=13$, cols: $13, 13, 21\times 8, 13, 13, 13$) | 1-strip fallback ($w=1, \ell=43$) | Missing exact $(w=13, t=4)$ seed emission ($L = 27.69238816 + 4$) |
| **543** | — | `23.8137085446` | `-0.1862914554` | $k=24$ baseline grid deficit | $w=16, \ell=16$ band at $45^\circ$ ($D=0$) | Synthetic baseline improvement over $24.0$ |
| **544** | — | `23.8284271704` | `-0.1715728296` | $k=24$ baseline grid deficit | $w=4, \ell=28$ band at $45^\circ$ | Synthetic baseline improvement over $24.0$ |
| **545** | — | `23.8994949824` | `-0.1005050176` | $k=24$ baseline grid deficit | $w=14, \ell=14$ band at $45^\circ$ + 3 border | Synthetic baseline improvement over $24.0$ |
| **591** | — | `24.8284271724` | `-0.1715728276` | $k=25$ baseline grid deficit | $w=4, \ell=28$ band at $45^\circ$ + 1 border | Synthetic baseline improvement over $25.0$ |
| **592** | — | `24.8994949844` | `-0.1005050156` | $k=25$ baseline grid deficit | $w=14, \ell=21$ band at $45^\circ$ | Synthetic baseline improvement over $25.0$ |
| **593** | — | `24.8994949844` | `-0.1005050156` | $k=25$ baseline grid deficit | $w=14, \ell=21$ band at $45^\circ$ | Synthetic baseline improvement over $25.0$ |

---

## 2. Anatomical Dissection of $n = 908$ ($L = 30.63702956$)

### 2.1 Global Composition and Quadrant Partition
In `baseline-targets.csv`, $n = 908$ is packed into a bounding box of exact side length:
$$L = 30.637029559612035$$
The 908 squares decompose into:
- **708 Upright Squares ($\theta = 0$):** Organized on the Cartesian half-integer grid ($x \approx i + 0.5, y \approx j + 0.5$).
- **200 Tilted Squares ($\theta \ne 0$):** Divided cleanly into two isolated 100-square blocks situated in the NW and SE quadrants.

```
ASCII Spatial Phase Distribution for n = 908 (31 x 31 Grid Cells)
Key: '#' = Upright (708), '+' = Tilted Block 1 (100, NW), '-' = Tilted Block 2 (100, SE)

Row 30: ###########..##################
Row 29: #######+..++.##################
Row 28: ####..+++++++..################
Row 27: ##.+++++++++++.################
Row 26: #.+++++++++.+++.###############  <-- NW Block: 100 tilted squares
Row 25: .+++++.++++++++.###############      embedded in upright cavity
Row 24: #+.++++++++++.++.##############
Row 23: #.+++++++.+++++.###############
Row 22: ##+++.++++++++.################
Row 21: ##.+++++++++.##################
Row 20: ####++++.######################
Row 19: ####.##########################
Row 18: #####################.#########  <-- Central diagonal upright corridor
Row 17: ####################.-..#######      remains 100% solid (no band cuts it!)
Row 16: ###################.----..#####
Row 15: ###################--.---.#####
Row 14: ##################.-------..###
Row 13: ##################-----.---.###
Row 12: ##################---------####  <-- SE Block: 100 tilted squares
Row 11: ##################--.-----.####      embedded in upright cavity
Row 10: #################.--------.####
Row 09: #################----------####
Row 08: #################-----.---#####
Row 07: #################--.-----.#####
Row 06: ################.--------.#####
Row 05: ##################-------######
Row 04: ##################.--.--.######
Row 03: ####################.--.#######
Row 02: ######################.#####...
Row 01: ##############################.
Row 00: ##############################.
        Col 0                       Col 30
```

### 2.2 Exact Transposition Symmetry Between Block 1 and Block 2
The two 100-square blocks are not independent heuristic clusters; they satisfy an **exact discrete transposition symmetry** with integer translation offsets:
$$\begin{pmatrix} x_2 \\ y_2 \\ \theta_2 \end{pmatrix} = \begin{pmatrix} y_1 - 3.0 \\ x_1 + 2.0 \\ -\theta_1 \end{pmatrix}$$

Checking all 100 corresponding square pairs between Block 1 (NW) and Block 2 (SE):
$$\max_{i \in [1, 100]} \left\| \begin{pmatrix} x_{2, i} \\ y_{2, i} \end{pmatrix} - \begin{pmatrix} y_{1, i} - 3.0 \\ x_{1, i} + 2.0 \end{pmatrix} \right\|_2 = 9.0164 \times 10^{-11}$$
This confirms that the two pockets are identical rigid clusters rotated by $90^\circ$ around the anti-diagonal and offset by $(-3, +2)$.

### 2.3 Coordinate Spans & Column Length Profiles
Both blocks possess identical internal lattice geometry:
- **Block 1 (NW):**
  - $X \in [1.415366, 15.222454]$ ($\text{span} = 13.807088$)
  - $Y \in [20.608623, 29.977350]$ ($\text{span} = 9.368727$)
  - Centroid: $(\bar{x}_1, \bar{y}_1) = (8.344499, 24.909707)$
- **Block 2 (SE):**
  - $X \in [17.608623, 26.977350]$ ($\text{span} = 9.368727$)
  - $Y \in [3.415366, 17.222454]$ ($\text{span} = 13.807088$)
  - Centroid: $(\bar{x}_2, \bar{y}_2) = (21.909707, 10.344499)$

Projecting Block 1 onto its longitudinal axis ($u = x\cos\bar{\theta} + y\sin\bar{\theta}$) decomposes the 100 squares into **exactly 13 parallel columns**:

$$\text{Column Counts (13 columns): } \mathbf{(5, 7, 8, 8, 9, 8, 8, 9, 9, 8, 8, 8, 5)}$$
$$\sum \text{Column Counts} = 5 + 7 + 8 + 8 + 9 + 8 + 8 + 9 + 9 + 8 + 8 + 8 + 5 = 100$$

| Col Index | Square Count | $u$ Mean | $v$ Span $[v_{\min}, v_{\max}]$ | Angle Span $[\theta_{\min}, \theta_{\max}]$ | Local Geometric Function |
|---|---|---|---|---|---|
| 0 | 5 | 11.828 | [18.40, 22.42] | [24.28°, 26.80°] | Outer tapered NW tip |
| 1 | 7 | 12.835 | [16.81, 22.90] | [24.10°, 26.80°] | Flank expansion |
| 2 | 8 | 13.839 | [16.35, 23.37] | [24.10°, 26.01°] | Full-width column |
| 3 | 8 | 14.848 | [15.94, 22.97] | [24.10°, 26.68°] | Full-width column |
| 4 | 9 | 15.853 | [15.49, 23.52] | [22.97°, 25.70°] | Crest column (maximum depth) |
| 5 | 8 | 16.859 | [16.02, 23.07] | [22.97°, 26.66°] | Interior body |
| 6 | 8 | 17.868 | [15.65, 22.65] | [24.10°, 24.10°] | Uniform central column |
| 7 | 9 | 18.886 | [15.19, 23.30] | [9.15°, 24.10°] | Junction crest (interface relax) |
| 8 | 9 | 19.891 | [14.73, 22.74] | [24.10°, 24.74°] | Secondary crest |
| 9 | 8 | 20.897 | [15.28, 22.29] | [24.10°, 25.85°] | Interior body |
| 10 | 8 | 21.906 | [14.88, 21.90] | [23.90°, 25.72°] | Flank taper |
| 11 | 8 | 22.924 | [15.38, 22.43] | [23.90°, 27.09°] | Outer boundary column |
| 12 | 5 | 23.899 | [15.83, 19.86] | [24.10°, 27.09°] | Outer tapered SE tip |

### 2.4 Angle Distribution & Why Non-$45^\circ$ Achieves $L = 30.637$
Of the 100 squares in Block 1:
- 58 squares have an **identical dominant angle**: $\theta = 0.420631\text{ rad} = \mathbf{24.1004^\circ}$.
- 8 squares are at $25.724^\circ$.
- 4 squares are at $26.798^\circ$.
- 3 squares are at $26.006^\circ$, 3 at $23.896^\circ$.
- Angles range between $9.15^\circ$ and $27.09^\circ$, centered around $\arctan(1/2) = 26.565^\circ$ and $\arctan(17/38) = 24.102^\circ$.

#### The Mechanical Advantages of $\theta \approx 24.1^\circ - 26.6^\circ$ Over $45^\circ$:
1. **Transverse Width Compression:**  
   At $\theta = 45^\circ$, the bounding span of a unit square along the coordinate axes is $\cos 45^\circ + \sin 45^\circ = \sqrt{2} \approx 1.41421$.  
   At $\theta = \arctan(1/2) \approx 26.565^\circ$, the bounding span is:
   $$\cos\theta + \sin\theta = \frac{2}{\sqrt{5}} + \frac{1}{\sqrt{5}} = \frac{3}{\sqrt{5}} \approx 1.34164$$
   This represents a **5.13% reduction in transverse space** per square compared to $45^\circ$.
2. **2:1 Rational Cartesian Staircase Matching:**  
   A $45^\circ$ band steps by $(1, 1)$, creating large triangular corner dead spaces against orthogonal boundaries. At slope $\tan\theta \approx 1/2$, the boundary interface steps by $2$ units horizontally for every $1$ unit vertically (or vice versa). Two upright squares ($2 \times 1$) mate with a tilted square of hypotenuse $\sqrt{1^2 + 0.5^2} = \sqrt{1.25}$, allowing periodic contact interlock without corner voids.
3. **Avoidance of Corner-to-Corner Diagonal Inflation:**  
   In any single-band $45^\circ$ construction, the band must span the full container diagonal. Its transverse thickness $w/\sqrt{2}$ forces the container side length to inflate by:
   $$L_{45} = \frac{a_1 + a_2}{2} + 1 + \frac{w}{\sqrt{2}}$$
   For $w = 20$, $L_{45} \approx 30.6421$; for $w = 8$, $L_{45} \approx 30.6569$.  
   By placing tilted squares in **two decoupled quadrant pockets** (NW and SE), the center diagonal corridor of the box remains filled with solid upright squares ($100\%$ packing density). Upright squares can run continuously from SW to NE along the main diagonal, completely decoupling container side length from the band width equation.

---

## 3. Anatomical Dissection of $n = 972$ ($L = 31.69238816$)

### 3.1 Exact Partition
In `baseline-targets.csv`, $n = 972$ has certified side length:
$$L = 31.692388155428386$$
Notice the exact algebraic decomposition:
$$L = 27.692388155428386 + 4 = L_{45}(w=13) + 4$$
The 972 squares partition into:
- **739 Upright Squares ($\theta = 0$):**
  - **231 Squares in the SW Triangular Staircase:** A complete triangular staircase of side $k_{\text{sw}} = 21$:
    $$\sum_{i=1}^{21} i = \frac{21 \times 22}{2} = 231 \text{ squares}$$
    Row counts from bottom to top: $(21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)$.
  - **508 Squares in the NE Upright Block:** A massive composite upright block incorporating the NE flank staircase and the 4-layer upright outer border ($t = 4$).
- **233 Tilted Squares ($\theta \approx 45^\circ$):** Forming a 13-strip diagonal band with variable column counts.

```
ASCII Spatial Distribution for n = 972 (32 x 32 Grid Cells)
Key: 'S' = SW Triangular Staircase (231), '/' = 13-Strip Tilted Band (233), 'N' = NE Upright Block (508)

NNNNNN.NNNNNNNNNNNNNNNNNNNNNNNNN
NNNNN./.NNNNNNNNNNNNNNNNNNNNNNNN  <-- NE Upright Block: 508 squares
NNNN././.NNNNNNNNNNNNNNNNNNNNNNN      (includes 4-layer outer border)
NNN./////.NNNNNNNNNNNNNNNNNNNNNN
NN.//.//./.NNNNNNNNNNNNNNNNNNNNN
N./////////.NNNNNNNNNNNNNNNNNNNN
.//.//.///./.NNNNNNNNNNNNNNNNNNN
N.///.//.//.NNNNNNNNNNNNNNNNNNNN
NN////////.NNNNNNNNNNNNNNNNNNNNN
NNN.//.//.NNNNNNNNNNNNNNNNNNNNNN
NNNN////.NNNNNNNNNNNNNNNNNNNNNNN
SNNNN./.NNNNNNNNNNNNNNNNNNNNNNNN  <-- 13-Strip Band: 233 squares
SSNNNN.NNNNNNNNNNNNNNN.NNNNNNNNN      Strip counts: (13, 13, 21x8, 13, 13, 13)
SSSNNNNNNNNNNNNNNNNNN./.NNNNNNNN
SSSSNNNNNNNNNNNNNNNN././.NNNNNNN
SSSSSNNNNNNNNNNNNNN./////.NNNNNN
SSSSSSNNNNNNNNNNNN.//.//./.NNNNN
SSSSSSSNNNNNNNNNN.////////..NNNN
SSSSSSSSNNNNNNNN.//.//./////.NNN
SSSSSSSSSNNNNNN././//.//.//./.NN
SSSSSSSSSSNNNN.///////////////.N
SSSSSSSSSSSNN.//.//.//.///.//./.
SSSSSSSSSSSSN//////////////////.
SSSSSSSSSSSSSN.//.///.//.//.//.N
SSSSSSSSSSSSSSN//////////////.NN
SSSSSSSSSSSSSSSN.//.//.///./.NNN
SSSSSSSSSSSSSSSSN.///.//.//.NNNN
SSSSSSSSSSSSSSSSSN////////.NNNNN
SSSSSSSSSSSSSSSSSSN.//.//.NNNNNN
SSSSSSSSSSSSSSSSSSSN////.NNNNNNN
SSSSSSSSSSSSSSSSSSSSN./.NNNNNNNN
SSSSSSSSSSSSSSSSSSSSSN.NNNNNNNNN
^ SW Staircase of side 21 (231 squares)
```

### 3.2 Tilted Band Structure & Strip Profile
Projecting the 233 tilted squares along $u = (x + y)/\sqrt{2}$ yields **exactly 13 strips**:
$$\text{Strip Lengths (13 strips): } \mathbf{(13, 13, 21, 21, 21, 21, 21, 21, 21, 21, 13, 13, 13)}$$
$$\text{Total Tilted Squares} = 5 \times 13 + 8 \times 21 = 65 + 168 = \mathbf{233}$$

### 3.3 Proof of Exact $L_{45} + 4$ Derivation
In the Pell-17 family core (see `poc/pell_corrugation/docs/pell_corrugation.md`), setting $w = 13$, $b = 9$, $(a_1, a_2) = (18, 17)$ yields:
$$L_{45} = \frac{18 + 17}{2} + 1 + \frac{13}{\sqrt{2}} = 17.5 + 1 + 9.1923881554 = 27.6923881554$$
Wrapping this core with a $t = 4$ upright border adds $+4$ to both coordinate axes:
$$L = L_{45} + 4 = 31.692388155428386$$
This matches the public incumbent for $n = 972$ down to $10^{-15}$ precision! The public leader for $n = 972$ is an exact algebraic $(w = 13, t = 4)$ staircase band. Our baseline failed to beat it only because our generator had emitted an uncorrugated single-strip fallback ($w = 1, \ell = 43, L = 31.7071$) rather than the $w = 13, t = 4$ seed.

---

## 4. Anatomical Dissection of $n = 971$ ($L = 31.67877988$) and Deficit Dynamics

### 4.1 The Compression Gap at $n = 971$
- **Public Incumbent:** $L(971) = \mathbf{31.67877987534328}$
- **Our Pell-17 Corrugated Candidate:** $L(971) = 31.69092120213406$
- **Uncorrugated $L_{45} + 4$ (from $n = 972$):** $L(972) = 31.69238815542839$

Comparing these lengths:
$$\Delta L_{\text{public}}(972 \to 971) = 31.67877988 - 31.69238816 = \mathbf{-0.01360828}$$
$$\Delta L_{\text{our\_Pell17}}(972 \to 971) = 31.69092120 - 31.69238816 = \mathbf{-0.00146695}$$

### 4.2 Why Pell-17 Pinch-Column Corrugation Undershoots the Public Mark
Our Pell-17 corrugation algorithm relies strictly on **equal-count transverse tooth corrugation**:
1. All 25 columns in the band remain present.
2. Odd $a_1 + a_2 = 35$ produces out-of-phase staircase teeth.
3. Pinch columns are tilted by $\delta_c \approx \pm 0.002$ rad to penetrate teeth gaps.
4. Maximum shrinkage is strictly limited by the second-order SAT expansion of adjacent columns:
   $$\Delta L_{\text{corr}} \approx -\frac{|\delta|}{2} \approx -0.001467$$

In contrast, the public leader for $n = 971$ achieves a shrinkage of **$-0.013608$**, which is **$9.28\times$ larger**!

### 4.3 The Vacancy Defect Relaxation Mechanism
The public packing at $n = 971$ is obtained by **deleting 1 square from $n = 972$**:
1. In $n = 972$, the tilted band has 5 columns of length 13 and 8 columns of length 21 (total 233 squares).
2. Removing 1 square from an end column (e.g. reducing a column from 13 to 12, or 21 to 20) introduces an **internal vacancy**.
3. This vacancy eliminates the longitudinal jamming condition along that column.
4. The remaining squares in that column can translate longitudinally, allowing the flank staircase teeth to collapse inward by:
   $$\delta u \approx 1 - \frac{1}{\sqrt{2}} \approx 0.2929$$
5. Because the boundary is no longer locked by full-rank rigid contacts, the LP polisher can contract the bounding container by $\Delta L = -0.0136$, reaching $L = 31.67877988$.

---

## 5. Large-Gap Neighbors: $n \in [543, 545]$ and $n \in [591, 593]$

### 5.1 The Interval Geometry of $(k^2 - n)/k \in [1, 2]$
The deficit intervals identified in `ARGON_BRIEF.md` follow a universal pattern governed by the square roots of $n$:

| Target Cluster | $k = \lceil\sqrt{n}\rceil$ | Baseline Grid $k \times k$ | Excess Capacity $(k^2 - n)$ | Deficit Ratio $(k^2 - n)/k$ | Public Table Weighted Gap |
|---|---|---|---|---|---|
| **$908 - 911$** | $31$ | $961$ | $50 - 53$ | $1.61 - 1.71$ | $0.0047890$ |
| **$971 - 973$** | $32$ | $1024$ | $51 - 53$ | $1.59 - 1.66$ | $0.0039596$ |
| **$591 - 593$** | $25$ | $625$ | $32 - 34$ | $1.28 - 1.36$ | $0.0035958$ |
| **$543 - 545$** | $24$ | $576$ | $31 - 33$ | $1.29 - 1.38$ | $0.0030262$ |

In every case, the target $n$ requires packing roughly $1.3k$ to $1.7k$ fewer squares than a complete grid of side $k$, but substantially more than a grid of side $k-1$ (which holds $(k-1)^2 \approx k^2 - 2k + 1$).

### 5.2 Deficit Analysis for $n \in [543, 545]$ ($k = 24$, Base Grid 576)
- **Grid Baseline:** $L = 24.0$ (holds up to 576 squares).
- **Required Vacancy:** $576 - 543 = 33$ squares.
- **Our Synthetic Solutions:**
  - $n = 543$: $w = 16, \ell = 16, D = 0, a = (12, 11), b = (11, 12) \implies L = 23.81370854$ (gain $-0.18629$).
  - $n = 544$: $w = 4, \ell = 28, D = 17 \implies L = 23.82842717$ (gain $-0.17157$).
  - $n = 545$: $w = 14, \ell = 14$ with $t = 3$ border $\implies L = 23.89949498$ (gain $-0.10051$).
- **Structural Pitfall:**  
  At $w = 16$, the band is almost square ($16 \times 16$), forcing extreme corner truncation. When transitioning to $n = 545$, the generator switches topology entirely to $w = 14$ plus border, leaving an unoptimized jump of $+0.086$ in $L$.

### 5.3 Deficit Analysis for $n \in [591, 593]$ ($k = 25$, Base Grid 625)
- **Grid Baseline:** $L = 25.0$ (holds up to 625 squares).
- **Required Vacancy:** $625 - 591 = 34$ squares.
- **Our Synthetic Solutions:**
  - $n = 591$: $w = 4, \ell = 28$ with $t = 1$ border $\implies L = 24.82842717$ (gain $-0.17157$).
  - $n = 592$: $w = 14, \ell = 21, D = 0 \implies L = 24.89949498$ (gain $-0.10051$).
  - $n = 593$: Duplicate of $592$ at $L = 24.89949498$.
- **Structural Pitfall:**  
  $n = 592$ and $n = 593$ emit identical configurations because the single-band discrete capacity jump ($\Delta \text{capacity} \ge w$) cannot absorb single-square increments smoothly without widening the bounding box.

---

## 6. Synthesis: The Four Missing Mechanisms

| Mechanism | Description | Demonstrated In | Impact on $L$ | Why Single-Band $45^\circ$ Fails |
|---|---|---|---|---|
| **1. Multi-Pocket Decoupled Partitions** | Divide tilted squares into 2 or more localized quadrant pockets (e.g. 100 NW + 100 SE) instead of 1 spanning band. | $n = 908$ | Prevents diagonal dilation ($L \approx 30.637$ vs $30.642$) | Single band forces $L = \frac{a_1+a_2}{2} + 1 + \frac{w}{\sqrt{2}}$, swelling container diagonally. |
| **2. Rational-Slope Lattice Frames ($\theta \approx \arctan(1/2)$)** | Orient tilted clusters at $\theta \approx 24.1^\circ - 26.6^\circ$ matching 2:1 integer Cartesian steps. | $n = 908$ | Reduces transverse square width by $5.13\%$ ($1.3416$ vs $1.4142$); eliminates boundary tooth voids. | $45^\circ$ band has $\cos 45^\circ + \sin 45^\circ = \sqrt{2}$, maximizing transverse extrusion. |
| **3. Asymmetric Corner / End Truncation** | Taper column lengths at band ends (e.g. $(13, 13, 21\times 8, 13, 13, 13)$) to mate with asymmetric upright staircases. | $n = 972$ | Matches exact $L_{45} + t$ border with zero wasted perimeter cells. | Rigid uniform-length bands leave large triangular voids at corners. |
| **4. Single-Cell Defect Vacancy Relaxation** | Remove 1 cell from a jammed boundary column, allowing adjacent columns and teeth to relax inward via LP. | $n = 971$ | Yields $\Delta L \approx -0.0136$ (nearly $10\times$ deeper than pinch-column corrugation). | Pinch corrugation preserves full count; jammed contacts halt first-order LP contraction. |

---

## 7. Concrete Generator Blueprint for Phase 4

To systematically beat the public leaderboard table in the $(k^2 - n)/k \in [1, 2]$ gap intervals, the next CPU generator must incorporate:

1. **Dual-Pocket $\arctan(1/2)$ Seed Generator:**
   - Parameterized by pocket size $m$ (e.g. $m = 100$), slope angle $\theta \in [\arctan(4/9), \arctan(1/2)]$, column count $c = 13$, and relative translation offset $(\Delta x, \Delta y) = (-3, +2)$.
   - Embedded into a $(k-1) \times (k-1)$ upright grid by removing $2 \times (\text{pocket footprint})$ upright squares.
2. **Defect-Driven LP Shrink Polish:**
   - Take the certified $n$ configuration (such as $n = 972$ at $L = 31.69238816$).
   - Systematically delete 1 square from each candidate column of the band (testing all 13 columns).
   - Solve the 4-sign-piece epigraph LP with an active box-shrink objective $\min L$.
   - This directly generates candidate $n-1$ packings (such as $n = 971$ at $L \le 31.67878$).
3. **Exact $(w=13, t=4)$ Seed Emission for $n = 972$:**
   - Emit the exact $(w=13, t=4, \ell=(13, 13, 21\times 8, 13, 13, 13))$ configuration to capture $L = 31.69238816$.
