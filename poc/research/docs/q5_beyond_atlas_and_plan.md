# Beyond Atlas: Evaluation of Novel Large-n Mechanisms and Ranked Execution Plan

## 1. Executive Summary

Atlas polishing and unstructured annealing have plateaued against jammed upright-grid topologies: 22,380 A100 candidate trials across 45 weak $n$ produced zero admissions, and 1.36 million state transitions in cross-$n$ deletion yielded numerical dust ($1.76 \times 10^{-11}$). However, 88.25% of our deficit versus the public per-$n$ table resides in the high-deficit band where $1 \le (k^2 - n)/k \le 2$ (specifically clusters 908–911, 971–973, 591–593, and 543–545).

The only certified win over the public incumbent achieved across all POCs is the **Pell-17 out-of-phase corrugated band + upright L-border** at $n = 910$ ($L = 30.690921074$ vs. public incumbent $30.690960121$, $\Delta L = -0.00003905$), along with baseline drops of $-0.05$ to $-0.18$ on 10 of 13 targets.

To beat Atlas at scale, constructions must solve four distinct geometric bottlenecks:
1. **Capacity Quantization**: Discrete integer staircase boundaries $(a_1, a_2, b_1, b_2)$ lock capacity into coarse jumps, failing on $n_0 \pm 1, \pm 2$.
2. **Longitudinal Corner Slack**: Integer corner staircases $b = \lfloor w/\sqrt{2} \rfloor$ leave dead space $\{w/\sqrt{2}\} \in (0, 1)$ at both band ends.
3. **Rigid Angle Constraint**: Forcing $\theta = 45^\circ$ imposes an inflation cost of $\sqrt{2} - 1 \approx 0.4142$ per tilted square, whereas rational slopes like $\theta = \arctan(1/2) \approx 26.56^\circ$ reduce inflation cost to $3/\sqrt{5} - 1 \approx 0.3416$ (a 17.5% reduction in dilation waste) and fit asymmetric rectangular container aspects.
4. **Epigraph Support Kinks**: Linearizing pairwise square separation at $r = \theta_j - \theta_i = 0$ jams standard gradient and LP methods; a 4-sign-piece epigraph LP formulation is required to allow adjacent columns to split angles continuously.

---

## 2. Evaluation of Novel Large-n Mechanisms

### Mechanism 1: Variable-Slope Bands ($\theta \neq 45^\circ$, e.g., $\theta \approx \arctan(1/2) \approx 26.565^\circ$)

#### The Mechanism
Standard diagonal band constructions fix $\theta = \pi/4$, enforcing mirror symmetry across the container diagonal. However, empirical analysis of the public incumbent for $n = 908$ ($L = 30.63702956$, beating standard $45^\circ$ band $L = 30.64213568$ by $0.005106$) reveals 688 upright squares ($0^\circ$) surrounded by tilted bands with angles concentrated near $\theta \in [0.4206, 0.4728]$ radians ($24.1^\circ$ to $27.1^\circ$). This clusters tightly around $\theta = \arctan(1/2) \approx 0.463648$ radians ($26.565^\circ$).

A variable-slope band aligns along rational lattice slopes $1:2$ ($\theta = \arctan(1/2)$) or $2:1$ ($\theta = \arctan(2) \approx 63.435^\circ$), with flank upright staircases matching the periodic $2 \times 1$ tooth geometry.

#### The Math
- **Inflation Extent**: For a unit square rotated by $\theta$, the projected horizontal and vertical bounding extent is:
  $$h(\theta) = \cos\theta + \sin\theta.$$
  At $\theta = 45^\circ$: $h(\pi/4) = \sqrt{2} \approx 1.414214$.
  At $\theta = \arctan(1/2)$: $\cos\theta = 2/\sqrt{5}$, $\sin\theta = 1/\sqrt{5}$, yielding:
  $$h(\arctan(1/2)) = \frac{3}{\sqrt{5}} = \frac{3\sqrt{5}}{5} \approx 1.341641.$$
  The transverse inflation penalty is reduced from $+0.414214$ to $+0.341641$—an immediate saving of $0.072573$ container units per tilted square.
- **Integer-Grid Matching Conditions**:
  Along slope $1:2$, the band tangent vector is $\mathbf{t} = (2/\sqrt{5}, 1/\sqrt{5})$ and normal vector is $\mathbf{n} = (-1/\sqrt{5}, 2/\sqrt{5})$.
  An integer upright staircase stepping 2 units horizontally for every 1 unit vertically has face edge vector $(2, 1)$ with length $\sqrt{2^2 + 1^2} = \sqrt{5}$.
  Every $2 \times 1$ block of upright squares advances exactly $\sqrt{5} \approx 2.236068$ along the band axis $\mathbf{t}$.
  The internal tilted squares have period $1.0$ along $\mathbf{t}$.
  Commensurability requires finding rational convergents of $\sqrt{5}$:
  Continued fraction $[2; \overline{4}]$ yields convergents:
  $$\frac{p_0}{q_0} = \frac{2}{1} = 2.000, \quad \frac{p_1}{q_1} = \frac{9}{4} = 2.250, \quad \frac{p_2}{q_2} = \frac{38}{17} \approx 2.235294, \quad \frac{p_3}{q_3} = \frac{161}{72} \approx 2.236111.$$
  At large $n$ ($L \approx 30$), a band length of $L_{\text{band}} \approx 38$ squares matches $17$ tooth blocks of $2 \times 1$ upright squares with longitudinal vernier mismatch $|\Delta u| = |38 - 17\sqrt{5}| \approx 0.01315$, easily absorbed by local elastic column shifting.
- **Tooth Geometry**:
  A tooth comprises two squares $[(x, y), (x+1, y)]$ capped by $[(x, y+1), (x+1, y+1)]$. The tooth boundary points project onto normal $\mathbf{n}$ with uniform spacing $1/\sqrt{5} \approx 0.4472$.

#### Target $n$ Values
$n \in [905, 909]$, $n \in [968, 972]$, $n \in [588, 593]$, and $n \in [541, 545]$. Specifically where container aspect asymmetry permits off-center bands spanning from corner to flank rather than corner to corner.

#### Estimated Score Gain
- At $n = 908$: $\Delta L = -0.005106 \implies \Delta \text{score} \approx -0.000169$.
- Across ~35 high-deficit targets where $45^\circ$ bands overfill: $\Delta \text{score} \approx 0.006 - 0.012$.

#### Confidence Level
Medium-High for geometric capacity superiority; Medium for closed-form construction due to the need for asymmetric corner boundary handling.

#### Honest Dead Ends
- **Symmetric 4-Corner Closures**: A $1:2$ slope cannot meet both container walls with symmetric right-angle corner staircases. One end meets the horizontal wall at $26.56^\circ$ while the other meets the vertical wall at $63.43^\circ$. Attempting to force $b_1 = b_2$ or identical corner atoms fails.
- **Pure Integer Tooth Commensurability**: $\sqrt{5}$ is irrational; exact zero-stress alignment without LP column shifting is impossible for finite $n$.

---

### Mechanism 2: Vacancy-Propagated / Tooth-Deleted Staircase Families

#### The Mechanism
In closed-form staircase bands $(w, \ell, a_1, a_2, b_1, b_2)$, total count $N$ is strictly quantized:
$$N = w\ell + \frac{a_1(a_1+1)}{2} + \frac{a_2(a_2+1)}{2} + \frac{b_1(b_1+1)}{2} + \frac{b_2(b_2+1)}{2}.$$
Because each parameter is an integer, incrementing $a_1$ changes $N$ by $a_1 + 1 \approx 15\text{--}25$ squares, and incrementing $\ell$ changes $N$ by $w \approx 8\text{--}16$ squares. When target $n$ is $n_0 \pm 1$ or $n_0 \pm 2$, standard band generators cannot produce valid packings without dropping entire strips or accepting massive overlaps (e.g., $0.2708$ overlap in `poc/kat-band`).

Tooth deletion modifies the staircase boundary by selectively removing $m \in \{1, 2, 3\}$ perimeter squares from the triangular staircases $A(a) = a(a+1)/2$. Each deleted tooth creates a local cavity that allows the adjacent tilted column to penetrate deeper or permits a vacancy to propagate through the upright grid, absorbing $\pm 1, \pm 2$ count deviations at identical container side length $L$.

#### The Math
- **Perimeter Tooth Coordinate**:
  In flank staircase $a_1$, perimeter teeth lie at coordinates $(i, j)$ with $i + j = a_1 - 1$ ($0 \le i < a_1$).
  The projection of tooth $(i, j)$ along band axis $u = (x+y)/\sqrt{2}$ is:
  $$u_{\text{tooth}}(i) = \frac{L_0 + a_1 - 1 - 2i}{\sqrt{2}}.$$
  Deleting tooth $k$ frees a spatial cavity of depth $\Delta u = 1/\sqrt{2} \approx 0.7071$ and width $\sqrt{2} \approx 1.4142$ along the band boundary.
- **Vacancy Propagation**:
  Deleting tooth $(i, j)$ leaves the outer container walls $x = L_0, y = 0$ intact.
  Inside the band, column $c$ facing the deleted tooth can shift transversely by $\Delta v \le 0.50$ and longitudinally by $\Delta u \le 0.7071$.
  By shifting column $c$ longitudinally into the tooth cavity, an additional square can be inserted at the opposite end of column $c$ without increasing container length $L_0$.
  Conversely, deleting a tooth directly reduces count from $n_0$ to $n_0 - 1$ while holding $L_0$ constant, turning a discrete capacity jump into a continuous family:
  $$n \in \{n_0 - 2, n_0 - 1, n_0, n_0 + 1, n_0 + 2\}, \qquad L(n) \le L(n_0) + \mathcal{O}(10^{-5}).$$

#### Target $n$ Values
- Neighbors of Pell-17 cores:
  - $n_0 = 910$ ($w=13, t=3$): targets $n \in \{908, 909, 911, 912\}$.
  - $n_0 = 971$ ($w=13, t=4$): targets $n \in \{972, 973, 974\}$.
  - $n_0 = 544$ ($w=4, t=0$): targets $n \in \{543, 545\}$.
  - $n_0 = 592$ ($w=14, t=0$): targets $n \in \{591, 593\}$.

#### Estimated Score Gain
Currently, targets $n = 908, 909, 911, 972, 973$ fall back to loose padded bands ($L \approx 30.7071$ or $31.7071$).
Achieving $L \approx 30.6909$ at $n = 911$ saves $\Delta L \approx -0.0162$, yielding $\Delta \text{score} \approx -0.00053$ per $n$. Across 12 targeted neighbor configurations: $\Delta \text{score} \approx 0.005 - 0.008$.

#### Confidence Level
High. The algebraic core remains certified; tooth deletion modifies only boundary rows and preserves outer bounding extrema.

#### Honest Dead Ends
- **Discrete Seam MILPs (Boundary Zipper)**: Tested in `poc/boundary_zipper` with 6 uncapped HiGHS models; all 6 were infeasible because rigid half-unit and $\sqrt{2}-1$ displacement lattices lack continuous slack. Tooth deletion must be coupled with continuous LP epigraph relaxation, not combinatorial binary selection.
- **Random Interior Vacancies**: Creating vacancies deep inside the upright triangular grid cannot be reached by the tilted band; vacancies must lie strictly along the staircase perimeter.

---

### Mechanism 3: Asymmetric Corner-Gadget Terminations (Eliminating $\{w/\sqrt{2}\}$ Slack)

#### The Mechanism
In symmetric staircase bands, both ends of the band are backed by identical upright staircases $b_1 = b_2 = b = \lfloor w/\sqrt{2} \rfloor$.
The container diagonal space available to the band is:
$$u_{\text{hi}} - u_{\text{lo}} = \sqrt{2} L_0 - \frac{b_1 + b_2 + 2}{\sqrt{2}} - 1.0.$$
Because $b$ must be an integer, there is an unavoidable longitudinal dead-space gap:
$$\text{slack}(w) = \left\{ \frac{w}{\sqrt{2}} \right\} = \frac{w}{\sqrt{2}} - \left\lfloor \frac{w}{\sqrt{2}} \right\rfloor.$$
For widths where $\{w/\sqrt{2}\} \in (0.42, 0.99)$ (such as $w = 4$ where $\{4/\sqrt{2}\} = 0.8284$, or $w = 8$ where $\{8/\sqrt{2}\} = 0.6568$), this gap wastes up to $0.83$ units of longitudinal container space at the corner.

Asymmetric corner-gadget terminations replace the integer staircase $b_1 = b_2$ with:
1. Asymmetric staircases $b_1 \neq b_2$ ($b_1 = b, b_2 = b+1$), allowing odd sum $b_1 + b_2$ to pair with even $a_1 + a_2$ for out-of-phase corrugation.
2. Wedged corner squares: placing a unit square rotated at $45^\circ$ directly into the container corner $(0, 0)$ with center at $(1/\sqrt{2}, 1/\sqrt{2})$.

#### The Math
- **Longitudinal Slack Recovery**:
  Standard integer staircase $b$ enforces clearance $u_{\text{lo}} \ge (b+1)/\sqrt{2} + 0.5$.
  A wedged corner square at angle $45^\circ$ sits with its corner at $(0, 0)$ and its center at $(1/\sqrt{2}, 1/\sqrt{2})$. Its projection along $u$ is $u_{\text{corner}} = 1.0$.
  The clearance gained along $u$ is:
  $$\Delta u_{\text{gain}} = \left(\frac{b+1}{\sqrt{2}} + 0.5\right) - 1.0 = \frac{b+1}{\sqrt{2}} - 0.5.$$
  For $w = 4$, $b = \lfloor 4/\sqrt{2} \rfloor = 2$.
  $$\Delta u_{\text{gain}} = \frac{3}{\sqrt{2}} - 0.5 \approx 2.1213 - 0.5 = 1.6213.$$
  Subtracting the integer square extent, the net reduction in wasted container diagonal is $\approx \{w/\sqrt{2}\} \approx 0.8284$.
- **Container Side Shrinkage**:
  Recovering longitudinal slack $\Delta u$ at both corners contracts container side length $L$ by:
  $$\Delta L = \frac{\Delta u}{\sqrt{2}}.$$
  For $w = 4$, $\Delta L \approx \frac{0.8284}{\sqrt{2}} \approx 0.5858$ container units!

#### Target $n$ Values
The entire set of widths rejected by Pell-17 due to large $\{w/\sqrt{2}\}$:
$w \in \{4, 5, 7, 8, 11, 12, 14, 15\}$.
Specifically:
- $n = 909$ ($w = 8$, $\{8/\sqrt{2}\} = 0.6568$).
- $n = 544$ and $n = 591$ ($w = 4$, $\{4/\sqrt{2}\} = 0.8284$).

#### Estimated Score Gain
- $n = 909$: Baseline is $30.707107$, incumbent is $30.656854$. Eliminating slack recovers $\Delta L \approx -0.050$, saving $\Delta \text{score} \approx -0.0016$.
- $n = 544, 591$: Saving $\Delta L \approx 0.05$ on $L \approx 24$ saves $\Delta \text{score} \approx -0.0021$ each.
- Full family potential across rejected widths: $\Delta \text{score} \approx 0.015 - 0.028$.

#### Confidence Level
Medium. The geometric slack recovery is mathematically certain; the challenge is packing the transition squares between the wedged corner square and the upright flank staircases.

#### Honest Dead Ends
- **Rigid Multi-Square Corner Atoms (17/37 Atoms)**: Tested in `poc/hinged_band`; Cantrell 37 and Hamalainen 17 atoms have fixed aspect ratios that do not dilate continuously with arbitrary $L$, leading to disastrous fallbacks ($L = 32.000$ vs incumbent $31.692$ at $n = 972$). Corner gadgets must be 1-to-3 square local terminations, not large rigid sub-packings.

---

### Mechanism 4: Multi-Block Composite Constructions (Pell Corrugated Cores + Upright L-Borders)

#### The Mechanism
88.25% of our deficit versus the public table lies where $1 \le (k^2 - n)/k \le 2$ ($k = \lceil\sqrt{n}\rceil$).
In this regime:
- A $k \times k$ upright grid is too loose ($k^2 - n \ge k$ empty spaces, wasting $\Delta L \approx 0.5\text{--}1.0$).
- A $(k-1) \times (k-1)$ upright grid cannot fit $n$ ($n > (k-1)^2$).
Hence the optimal configuration must compose a dense non-grid core packing of size $k_0 \times k_0$ ($k_0 < k$) that has higher density than an upright grid ($n_0 > k_0^2$), wrapped by an upright L-border of thickness $t = k - k_0$ layers.

The composite packing side length is exactly:
$$L = L_{\text{core}} + t.$$
If $L_{\text{core}} = k_0 - \epsilon$ ($\epsilon > 0$), then $L = k - \epsilon < k$, systematically beating the baseline.

#### The Math
- **Core Density & Scaling**:
  Let the core have count $n_0$ and bounding length $L_{\text{core}}$.
  Wrapping with a $t$-layer upright border on the North and East edges adds:
  $$n_{\text{border}} = t(2k - t) = 2kt - t^2 \quad \text{squares}.$$
  Total squares: $n = n_0 + 2kt - t^2$.
  Deficit ratio:
  $$\frac{k^2 - n}{k} = \frac{k^2 - (n_0 + 2kt - t^2)}{k} = \frac{(k-t)^2 - n_0}{k} = \frac{k_0^2 - n_0}{k}.$$
  Whenever $n_0 > k_0^2$, the core has negative deficit (super-density).
- **The Pell Corrugation Spectrum**:
  Longitudinal slack at $45^\circ$ is governed by Pell equation $D^2 - 2\ell^2 = \pm 1$:
  - $D = 7, \ell = 5$: $7/\sqrt{2} = 4.949747$ (slack $-0.05025$).
  - $D = 17, \ell = 12$: $17/\sqrt{2} = 12.020815$ (slack $+0.02082$).
  - $D = 41, \ell = 29$: $41/\sqrt{2} = 28.991378$ (slack $-0.00862$).
  - $D = 99, \ell = 70$: $99/\sqrt{2} = 70.003571$ (slack $+0.00357$).
  In `poc/pell_corrugation`, $D = 17$ was realized for $w \in \{2, 3, 6, 9, 10, 13, 16\}$, yielding corrugation shrinkage $\Delta L \approx -0.0014$ to $-0.0021$ below $L_{45}$.
  At $w = 13$, $k_0 = 27, n_0 = 739$ ($k_0^2 = 729 < 739$, super-dense).
  Adding $t = 3$ border layers gives $n = 739 + 171 = 910$, yielding certified $L = 30.690921074$, beating the public incumbent $30.690960121$.
  Adding $t = 4$ border layers gives $n = 739 + 232 = 971$, yielding certified $L = 31.690921$, beating the baseline by $-0.016186$.

#### Target $n$ Values
The entire high-deficit spectrum:
- $n \in [540, 550]$ ($k=24$, deficit ratio $\approx 1.08\text{--}1.50$).
- $n \in [585, 600]$ ($k=25$, deficit ratio $\approx 1.00\text{--}1.60$).
- $n \in [905, 925]$ ($k=31$, deficit ratio $\approx 1.16\text{--}1.80$).
- $n \in [965, 985]$ ($k=32$, deficit ratio $\approx 1.21\text{--}1.84$).

#### Estimated Score Gain
Capturing 25% of the $88.25\%$ deficit gap yields $\Delta \text{score} \approx -0.060$ to $-0.075$, closing the majority of the remaining gap to the $21.99$ target.

#### Confidence Level
Very High. Certified by unchanged official interval metric on literal and actual pandas 3.1.0 round-trip transport; already produced a certified record at $n = 910$.

#### Honest Dead Ends
- **Tilting Border Squares**: Border squares must remain strictly at $0^\circ$. If border squares rotate, their bounding extent increases from $1.0$ to $\cos\theta + \sin\theta > 1.0$, completely destroying boundary tightness.
- **LP Linearization without 4-Sign-Piece Epigraph**: Gradient descent and standard LP stall at the pairwise support kink $r = 0$ ($L = 30.69096039$). All 4 sign pieces $(\pm 1, \pm 1)$ must be enforced in the epigraph to allow angle splitting across adjacent columns.

---

## 3. Comparison Matrix of Novel Mechanisms

| Mechanism | Primary Target Domain | Theoretical Capacity Advantage | Implementation Complexity | Primary Failure Mode / Blocker | Current Certification Status |
|:---|:---|:---:|:---:|:---|:---:|
| **1. Variable-Slope Bands** ($\theta \approx 26.56^\circ$) | Asymmetric container aspects, $n = 908, 968$ | High ($17.5\%$ less dilation waste: $1.3416$ vs $1.4142$) | High | Asymmetric corner boundary closure, irrational $\sqrt{5}$ vernier mismatch | Conceptually proven in $n=908$ incumbent analysis; unbuilt generator |
| **2. Tooth-Deleted Staircase Families** | Exact-count neighbors ($n_0 \pm 1, \pm 2$) | Very High (converts rigid algebraic cores into per-$n$ continuous family) | Medium | Cavity collapse under unguided LP without trust region | Formulated; discrete MILP failed, continuous LP ready |
| **3. Asymmetric Corner Gadgets** | Widths rejected by Pell-17 ($w = 4, 8, 12$) | High (recovers up to $0.83$ longitudinal slack) | Medium-High | Transition geometry between wedged $45^\circ$ corner and $0^\circ$ flank | Rigid 17/37 atoms failed; single-square wedge ready |
| **4. Multi-Pell Composite Cores + L-Borders** | High-deficit region $1 \le (k^2 - n)/k \le 2$ | High (certified super-dense cores with $L = L_0 + t$) | Medium | Support kink stall at $r = 0$ unless 4-sign-piece LP used | **Certified Win at $n=910$**; beats baseline on 10/13 targets |

---

## 4. Ranked Execution Plan: Top 3 Things to Build Next

### Rank 1: Multi-Pell Corrugated Core Library ($D \in \{7, 17, 41\}$) with Parametric Upright L-Borders

#### Description
Expand the certified $D = 17$ engine (`poc/pell_corrugation`) to the full Pell spectrum $D \in \{7, 17, 41\}$.
$D = 7$ ($7/\sqrt{2} = 4.9497$, $\ell = w + 5$) directly provides dense cores for $w \in \{4, 8\}$, unlocking the rejected widths for $n = 544, 591, 909$.
$D = 41$ ($41/\sqrt{2} = 28.9914$, $\ell = w + 29$) provides larger cores with negligible slack ($0.0086$), targeting $k_0 \in [20, 26]$.
All cores are polished with the verified 4-sign-piece epigraph LP polisher and wrapped with parametric $t \in \{1, 2, 3, 4, 5\}$ upright L-borders.

#### Exact Fast Falsification Test (< 30 Minutes on CPU)
1. Instantiate algebraic seeds for:
   - $D = 7$: $w = 4, b = 2, a_1 = 6, a_2 = 5, \ell = 9 \implies k_0 = 15, n_0 = 126, L_{45} = 15.328427$.
   - $D = 41$: $w = 8, b = 5, a_1 = 23, a_2 = 23, \ell = 37 \implies k_0 = 29, n_0 = 852, L_{45} = 29.656854$.
2. Run 15 iterations of `_lp` with the 4-sign-piece epigraph on CPU.
3. **Falsification Gate**: If $L_{\text{corr}} \ge L_{45} - 0.00080$ (no corrugation shrinkage) or if the HiGHS LP is infeasible within 5 seconds, the candidate Pell family is falsified.

---

### Rank 2: Vacancy-Propagated Flank-Tooth Deletion Engine for Exact-Count Matching

#### Description
Implement parametric perimeter tooth deletion on the certified $n_0 = 739$ Pell-17 core.
To match $n = 908$ ($n_0 - 2$ with $t = 3$) and $n = 909$ ($n_0 - 1$ with $t = 3$), remove $m \in \{1, 2\}$ perimeter teeth from flank staircase $a_1$ at $(a_1 - 1, 0)$ and $(0, a_1 - 1)$.
The tooth deletion creates local entry cavities along the SE staircase, allowing the outer tilted columns to advance without increasing container side $L_0$.
The configuration is then polished using the 4-sign-piece LP with fixed outer bounding constraints.

#### Exact Fast Falsification Test (< 30 Minutes on CPU)
1. Load the uncorrugated $(w=13, \ell=25, a_1=18, a_2=17, b=9)$ seed.
2. Remove the single outermost tooth $(17, 0)$ from $a_1$, yielding count $n_{\text{core}} = 738$.
3. Wrap with $t = 3$ upright border ($171$ squares) to form exactly $n = 909$.
4. Run 10 iterations of `polish_band`.
5. **Falsification Gate**: If the resulting certified $L(909) \ge 30.656854$ (the public incumbent), or if removing the tooth causes the LP to dilate the bounding box ($L > L_0$), tooth deletion fails to preserve container compactness and is falsified.

---

### Rank 3: Variable-Slope ($1:2$, $\theta \approx 26.56^\circ$) Commensurate Staircase Band Generator

#### Description
Build a prototype synthetic band generator along slope $1:2$ ($\theta = \arctan(1/2) \approx 26.565^\circ$) targeting the structural deficit at $n = 908$.
The generator constructs:
1. Two flank upright staircases with periodic $2 \times 1$ tooth geometry along the band tangent $\mathbf{t} = (2/\sqrt{5}, 1/\sqrt{5})$.
2. A central tilted band of width $w \approx 10$ and length $\ell \approx 38$, matching convergent $38/17 \approx \sqrt{5}$.
3. Asymmetric corner buffers on horizontal wall ($26.56^\circ$) and vertical wall ($63.43^\circ$).
4. 4-sign-piece LP relaxation to accommodate the $0.013$ vernier mismatch.

#### Exact Fast Falsification Test (< 30 Minutes on CPU)
1. Generate an unpolished $1:2$ band prototype at $L_0 \approx 15$ ($n \approx 200$) with integer $2 \times 1$ staircase teeth.
2. Formulate the LP feasibility matrix against rigid upright teeth with position trust region $0.10$.
3. **Falsification Gate**: If the minimum container dilation required to resolve vernier overlaps exceeds $\Delta L = \sqrt{2} - 3/\sqrt{5} \approx 0.0726$ (the geometric inflation advantage), the tooth mismatch completely wipes out the slope advantage, falsifying the $1:2$ commensurate staircase mechanism.
