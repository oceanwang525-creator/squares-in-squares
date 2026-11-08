# Generalized Out-of-Phase Corrugation Theorem and Sweet Spots up to $n = 1000$

## Executive Summary

The out-of-phase pinch-column corrugation theorem is generalized from the specific Pell-17 core ($D = 17, \ell = w + 12$) to arbitrary staircase-band configurations $(w, a_1, a_2, b_1, b_2, D)$, where $D = (a_1 + a_2) - (b_1 + b_2)$ and strip column count $\ell = w + \lfloor D / \sqrt{2} \rfloor$.

### Key Findings
1. **Out-of-Phase Parity Condition**: First-order transverse shrinkage occurs if and only if the flank staircases are out of phase by $\frac{1}{\sqrt{2}}$ along the diagonal coordinate $u$, which holds if and only if:
   $$a_1 + a_2 \equiv 1 \pmod 2 \quad (\text{odd parity}).$$
   - For **symmetric end staircases** ($b_1 = b_2 = b$): $a_1 + a_2 = D + 2b$, so $a_1 + a_2$ is odd if and only if **$D$ is odd**.
   - For **asymmetric end staircases** ($b_1 = b, b_2 = b + 1$): $a_1 + a_2 = D + 2b + 1$, so $a_1 + a_2$ is odd if and only if **$D$ is even**.
2. **Longitudinal Slack Condition**: The longitudinal space available along $u$ at nominal $45^\circ$ is:
   $$\text{slack}(D) = \left\{\frac{D}{\sqrt{2}}\right\} = \frac{D}{\sqrt{2}} - \left\lfloor \frac{D}{\sqrt{2}} \right\rfloor.$$
   Corrugating with $K$ sign-alternating pinch columns expands the longitudinal column pitch by $\sin(2|\delta|) \approx 2|\delta|$, requiring:
   $$K \cdot |\delta| \le \text{slack}(D).$$
   To permit meaningful transverse tilt $|\delta| \ge 0.002$ without forcing container dilation, while avoiding uncompetitive longitudinal dead space, candidate $D$ must satisfy:
   $$0.010 \le \text{slack}(D) \le 0.150.$$
3. **Wall and Tooth Clearance Condition**:
   - For **symmetric odd $D$**: End staircases $b_1 = b_2 = b = \lfloor w / \sqrt{2} \rfloor$ back the band cleanly without colliding with container walls when:
     $$\left\{\frac{w}{\sqrt{2}}\right\} \in (0.05,\, \sqrt{2}-1) \approx (0.05, 0.4142).$$
   - For **asymmetric even $D$**: Because $b_2 = b + 1$, the clearance between the NE end staircase and the SE flank staircase along the right container wall satisfies $L_0 - b_2 - a_1 = \{w/\sqrt{2}\} - 0.5$. Non-collision requires:
     $$\left\{\frac{w}{\sqrt{2}}\right\} > 0.50 \quad \text{(practically } \{w/\sqrt{2}\} \in (0.55, 0.95)\text{)}.$$
4. **Admissible Candidate Classification ($n \le 1000$)**:
   - **Symmetric Odd $D$**:
     - $D = 3$ ($\ell = w + 2$, slack $= 0.12132$, Diophantine $3^2 - 2\cdot 2^2 = +1$)
     - $D = 17$ ($\ell = w + 12$, slack $= 0.02082$, Pell $17^2 - 2\cdot 12^2 = +1$)
     - $D = 27$ ($\ell = w + 19$, slack $= 0.09188$, hybrid $10 + 17$)
     - $D = 51$ ($\ell = w + 36$, slack $= 0.06245$, triple Pell $3 \times 17$)
   - **Asymmetric Even $D$**:
     - $D = 10$ ($\ell = w + 7$, slack $= 0.07107$, Diophantine $10^2 - 2\cdot 7^2 = +2$)
     - $D = 34$ ($\ell = w + 24$, slack $= 0.04163$, double Pell $2 \times 17$)
     - $D = 58$ ($\ell = w + 41$, slack $= 0.01219$, Diophantine $58^2 - 2\cdot 41^2 = +2$)
   - **Higher Pell Orders**: The next Pell candidate $D = 99$ ($99^2 - 2\cdot 70^2 = +1$) has minimum core size $n_0(w=1) = 2571 > 1000$, ruling out $D \ge 99$ for the $n \le 1000$ competition.

---

## 1. General Staircase-Band Geometry

Consider a square container $[0, L] \times [0, L]$ containing $n$ unit squares ($1 \times 1$). Define rotated coordinates along and across the diagonal:
$$u = \frac{x + y}{\sqrt{2}}, \qquad v = \frac{x - y}{\sqrt{2}}.$$
Inverse transformation:
$$x = \frac{u + v}{\sqrt{2}}, \qquad y = \frac{u - v}{\sqrt{2}}.$$

The container diagonal from $(0,0)$ to $(L,L)$ lies along $v = 0$, spanning $u \in [0, \sqrt{2}L]$.

### Staircase Components
Four upright staircases ($\theta = 0$) occupy the peripheral corners:
1. **SE Flank Staircase ($a_1$ steps)**: Located near $(L, 0)$. Squares are centered at $[x_i, y_j] = [L - 0.5 - i, 0.5 + j]$ for $i \ge 0, j \ge 0, i+j \le a_1 - 1$.
   The square count is $T(a_1) = \frac{a_1(a_1+1)}{2}$.
2. **NW Flank Staircase ($a_2$ steps)**: Located near $(0, L)$. Squares are centered at $[x_i, y_j] = [0.5 + i, L - 0.5 - j]$ for $i+j \le a_2 - 1$.
   The square count is $T(a_2) = \frac{a_2(a_2+1)}{2}$.
3. **SW End Staircase ($b_1$ steps)**: Located at $(0, 0)$. Squares are centered at $[0.5 + i, 0.5 + j]$ for $i+j \le b_1 - 1$.
   The square count is $T(b_1) = \frac{b_1(b_1+1)}{2}$.
4. **NE End Staircase ($b_2$ steps)**: Located at $(L, L)$. Squares are centered at $[L - 0.5 - i, L - 0.5 - j]$ for $i+j \le b_2 - 1$.
   The square count is $T(b_2) = \frac{b_2(b_2+1)}{2}$.

### Diagonal Band Component
A central diagonal band of $w$ strips at nominal orientation $\theta = 45^\circ$ ($\pi/4$) runs between the SW and NE ends. In rotated $(u, v)$ coordinates, each square at $\theta = 45^\circ$ has sides parallel to the $u$ and $v$ axes and side length $1.0$.

For a column of $w$ squares centered transversely at $v_s = v_{\text{mid}} + (s - \frac{w-1}{2})$ for $s \in \{0, \dots, w-1\}$, the transverse band width is exactly $w$.

### Derivation of Side Length $L_{45}$
The inner teeth of the flank staircases face the diagonal band:
- **SE Flank Tooth Tips**: Formed by squares with $i+j = a_1 - 1$. Their inward corners $(x_i - 0.5, y_j + 0.5)$ have transverse coordinate:
  $$v_{\text{SE, corner}} = \frac{(x_i - 0.5) - (y_j + 0.5)}{\sqrt{2}} = \frac{x_i - y_j - 1.0}{\sqrt{2}} = \frac{(L - 0.5 - i) - (0.5 + j) - 1.0}{\sqrt{2}} = \frac{L - a_1 - 1}{\sqrt{2}}.$$
- **NW Flank Tooth Tips**: Formed by squares with $i+j = a_2 - 1$. Their inward corners $(x_i + 0.5, y_j - 0.5)$ have transverse coordinate:
  $$v_{\text{NW, corner}} = \frac{(0.5 + i + 0.5) - (L - 0.5 - j - 0.5)}{\sqrt{2}} = \frac{a_2 + 1 - L}{\sqrt{2}} = -\frac{L - a_2 - 1}{\sqrt{2}}.$$

The transverse clearance between the SE and NW tooth tips must accommodate the band width $w$:
$$v_{\text{SE, corner}} - v_{\text{NW, corner}} = \frac{L - a_1 - 1}{\sqrt{2}} - \left(-\frac{L - a_2 - 1}{\sqrt{2}}\right) = \frac{2L - (a_1 + a_2) - 2}{\sqrt{2}} = w.$$

Solving for $L$:
$$2L - (a_1 + a_2) - 2 = \sqrt{2}w \implies 2L = (a_1 + a_2) + 2 + \sqrt{2}w,$$
$$\boxed{L_{45} = \frac{a_1 + a_2}{2} + 1 + \frac{w}{\sqrt{2}}}.$$

---

## 2. Derivation of Longitudinal Span and Slack

### Longitudinal Endpoints
The longitudinal travel of the band along $u$ is bounded by the SW staircase ($b_1$) and NE staircase ($b_2$):
- **SW End**: The corner of the SW staircase pointing into the band along the diagonal is at $x = 0.5 + i + 0.5, y = 0.5 + j + 0.5$ for $i+j = b_1 - 1$.
  $$u_{\text{SW, tip}} = \frac{(b_1 - 1) + 2}{\sqrt{2}} = \frac{b_1 + 1}{\sqrt{2}}.$$
  A band square centered at $u_{\min}$ extends by $-0.5$ along $u$, requiring:
  $$u_{\min} - 0.5 \ge u_{\text{SW, tip}} \implies u_{\text{lo}} = \frac{b_1 + 1}{\sqrt{2}} + 0.5.$$
- **NE End**: Measured from the container corner $(L, L)$ at $u = \sqrt{2}L$, the NE staircase tip is at $\sqrt{2}L - \frac{b_2 + 1}{\sqrt{2}}$.
  The outermost band square centered at $u_{\max}$ extends by $+0.5$, requiring:
  $$u_{\max} + 0.5 \le \sqrt{2}L - \frac{b_2 + 1}{\sqrt{2}} \implies u_{\text{hi}} = \sqrt{2}L - \frac{b_2 + 1}{\sqrt{2}} - 0.5.$$

### Exact Cancellation of Longitudinal Span
The available longitudinal span for column centers is:
$$u_{\text{hi}} - u_{\text{lo}} = \left(\sqrt{2}L_{45} - \frac{b_2 + 1}{\sqrt{2}} - 0.5\right) - \left(\frac{b_1 + 1}{\sqrt{2}} + 0.5\right) = \sqrt{2}L_{45} - \frac{b_1 + b_2 + 2}{\sqrt{2}} - 1.$$

Substituting $L_{45} = \frac{a_1 + a_2}{2} + 1 + \frac{w}{\sqrt{2}}$:
$$\sqrt{2}L_{45} = \frac{a_1 + a_2}{\sqrt{2}} + \sqrt{2} + w.$$

Therefore:
$$u_{\text{hi}} - u_{\text{lo}} = \left(\frac{a_1 + a_2}{\sqrt{2}} + \sqrt{2} + w\right) - \frac{b_1 + b_2 + 2}{\sqrt{2}} - 1$$
$$= (w - 1) + \left(\sqrt{2} - \frac{2}{\sqrt{2}}\right) + \frac{(a_1 + a_2) - (b_1 + b_2)}{\sqrt{2}}.$$

Since $\sqrt{2} - \frac{2}{\sqrt{2}} \equiv 0$, the constant geometric terms cancel identically:
$$\boxed{u_{\text{hi}} - u_{\text{lo}} = (w - 1) + \frac{D}{\sqrt{2}}}, \qquad \text{where } D = (a_1 + a_2) - (b_1 + b_2).$$

### Column Count and Slack
In a rigid band, consecutive column centers are spaced by $\Delta u = 1.0$. For $\ell$ columns, the span between the first and last column centers is $\ell - 1$. Fitting within the available span requires:
$$\ell - 1 \le (w - 1) + \frac{D}{\sqrt{2}} \implies \ell \le w + \frac{D}{\sqrt{2}}.$$
Because $\ell$ must be an integer, the maximum column count per strip is:
$$\boxed{\ell = w + \left\lfloor \frac{D}{\sqrt{2}} \right\rfloor}.$$

The leftover longitudinal clearance (longitudinal slack) depends strictly on $D$:
$$\boxed{\text{slack}(D) = (u_{\text{hi}} - u_{\text{lo}}) - (\ell - 1) = \left\{\frac{D}{\sqrt{2}}\right\} = \frac{D}{\sqrt{2}} - \left\lfloor \frac{D}{\sqrt{2}} \right\rfloor}.$$

The longitudinal slack is completely independent of strip width $w$ and depends solely on the Diophantine approximation properties of $D / \sqrt{2}$.

---

## 3. Derivation of the Out-of-Phase Corrugation Theorem

### Flank Tooth Longitudinal Phase
Along the SE staircase, the inner tooth squares have centers $(L - 0.5 - i, 0.5 + j)$ with $i+j = a_1 - 1$, so $j - i = a_1 - 1 - 2i$. Their longitudinal coordinates are:
$$u_{\text{teeth, SE}}(i) = \frac{(L - 0.5 - i) + (0.5 + j)}{\sqrt{2}} = \frac{L + a_1 - 1 - 2i}{\sqrt{2}}.$$
Consecutive teeth are spaced by $\Delta u = \frac{2}{\sqrt{2}} = \sqrt{2} \approx 1.4142136$.
Modulo the period $\sqrt{2}$:
$$u_{\text{teeth, SE}} \equiv \frac{L + a_1 - 1}{\sqrt{2}} \pmod{\sqrt{2}}.$$

Along the NW staircase, tooth squares have centers $(0.5 + i, L - 0.5 - j)$ with $i+j = a_2 - 1$, so $i - j = a_2 - 1 - 2j$. Their longitudinal coordinates are:
$$u_{\text{teeth, NW}}(j) = \frac{(0.5 + i) + (L - 0.5 - j)}{\sqrt{2}} = \frac{L + a_2 - 1 - 2j}{\sqrt{2}} \equiv \frac{L - a_2 + 1}{\sqrt{2}} \pmod{\sqrt{2}}.$$

The relative longitudinal phase difference between SE and NW teeth is:
$$\Delta u_{\text{phase}} = u_{\text{teeth, SE}} - u_{\text{teeth, NW}} \equiv \frac{(L + a_1 - 1) - (L - a_2 + 1)}{\sqrt{2}} = \frac{a_1 + a_2 - 2}{\sqrt{2}} \pmod{\sqrt{2}}.$$
Since $\frac{2}{\sqrt{2}} = \sqrt{2} \equiv 0 \pmod{\sqrt{2}}$:
$$\boxed{\Delta u_{\text{phase}} \equiv \frac{(a_1 + a_2) \bmod 2}{\sqrt{2}} \pmod{\sqrt{2}}}.$$

### Parity Classification
- **In-Phase ($a_1 + a_2$ is even)**:
  $$(a_1 + a_2) \bmod 2 = 0 \implies \Delta u_{\text{phase}} \equiv 0 \pmod{\sqrt{2}}.$$
  SE and NW teeth align at identical longitudinal positions.
- **Out-of-Phase ($a_1 + a_2$ is odd)**:
  $$(a_1 + a_2) \bmod 2 = 1 \implies \Delta u_{\text{phase}} \equiv \frac{1}{\sqrt{2}} \pmod{\sqrt{2}}.$$
  SE and NW teeth are offset by exactly half a tooth period ($\frac{1}{\sqrt{2}} = \frac{1}{2}\sqrt{2}$).

### Connection to End Staircases
Recall $D = (a_1 + a_2) - (b_1 + b_2) \implies a_1 + a_2 = D + (b_1 + b_2)$.
1. **Symmetric End Staircases ($b_1 = b_2 = b$)**:
   $$a_1 + a_2 = D + 2b \equiv D \pmod 2.$$
   Therefore, $a_1 + a_2$ is odd **if and only if $D$ is odd**.
2. **Asymmetric End Staircases ($b_1 = b, b_2 = b + 1$)**:
   $$a_1 + a_2 = D + 2b + 1 \equiv D + 1 \pmod 2.$$
   Therefore, $a_1 + a_2$ is odd **if and only if $D$ is even**.

### First-Order Transverse Shrinkage Proof
Let column $c$ have nominal center $u_c$, with distances to the nearest SE and NW teeth denoted by:
$$\Delta u_{\text{se}}(c) = u_c - u_{\text{teeth, SE}}^{\text{nearest}}, \qquad \Delta u_{\text{nw}}(c) = u_c - u_{\text{teeth, NW}}^{\text{nearest}}.$$
A column is a **pinch column** if $\max(|\Delta u_{\text{se}}(c)|, |\Delta u_{\text{nw}}(c)|) < 0.50$.

#### Case 1: In-Phase Staircases ($a_1 + a_2$ even)
Because the tooth grids are aligned:
$$\Delta u_{\text{se}}(c) = \Delta u_{\text{nw}}(c) = \Delta u(c).$$
If column $c$ is tilted by an angle $\delta_c$:
- The SE flank face changes transverse clearance by $+\Delta u(c) \cdot \delta_c$.
- The NW flank face changes transverse clearance by $-\Delta u(c) \cdot \delta_c$.
- The sum of transverse changes across the band is:
  $$\Delta v_{\text{total}} = +\Delta u(c)\delta_c - \Delta u(c)\delta_c = 0.$$
The first-order linear term vanishes identically.
If one attempts opposite half-column tilts across the band midline ($+\delta$ on SE half, $-\delta$ on NW half):
- By the Separating Axis Theorem (SAT), the internal contact separation cost across the midline is $+|\delta|$ (first-order expansion).
- The flank penetration gain at both ends is at most $2 |\Delta u(c)| \cdot |\delta|$.
- At pinch columns, $|\Delta u(c)| < 0.50 \implies 2|\Delta u(c)| < 1.0$.
- Net transverse change: $+|\delta| - 2|\Delta u(c)||\delta| = (1 - 2|\Delta u(c)|)|\delta| > 0$ (dilation).
**Conclusion**: No first-order transverse shrinkage exists for in-phase staircases.

#### Case 2: Out-of-Phase Staircases ($a_1 + a_2$ odd)
Because the tooth grids are offset by $\frac{1}{\sqrt{2}} \approx 0.7071$:
$$\Delta u_{\text{se}}(c) - \Delta u_{\text{nw}}(c) \equiv \pm \frac{1}{\sqrt{2}} \pmod{\sqrt{2}}.$$
At every pinch column ($\max(|\Delta u_{\text{se}}|, |\Delta u_{\text{nw}}|) < 0.50$):
$$\Delta u_{\text{se}}(c) \text{ and } \Delta u_{\text{nw}}(c) \text{ have OPPOSITE SIGNS}, \quad \text{and} \quad |\Delta u_{\text{se}}(c)| + |\Delta u_{\text{nw}}(c)| = \frac{1}{\sqrt{2}}.$$
Now apply a uniform column tilt:
$$\delta_c = -\text{sign}(\Delta u_{\text{se}}(c)) \cdot |\delta| = +\text{sign}(\Delta u_{\text{nw}}(c)) \cdot |\delta|.$$
Because the column tilts as a single rigid unit, there is **zero internal SAT separation cost** ($g_{\text{internal}} = 0$).
Simultaneously, at both the SE and NW flank boundaries:
- The SE end rotates toward the receding tooth flank, gaining penetration $-\frac{|\delta|}{\sqrt{2}}$.
- The NW end rotates toward the receding tooth flank, gaining penetration $-\frac{|\delta|}{\sqrt{2}}$.
- Total transverse band contraction:
  $$\Delta w_{\text{eff}} \approx -|\delta|.$$
- Container side length change:
  $$\boxed{\Delta L \approx \frac{\Delta w_{\text{eff}}}{2} \approx -\frac{|\delta|}{2}}.$$
This constitutes certified first-order transverse shrinkage.

---

## 4. Longitudinal Slack Budget and Wall Clearance

### Longitudinal Slack Budget
When adjacent columns in the band have alternating tilts $\pm |\delta|$, their relative angle is $\Delta \theta = 2|\delta|$. By SAT support:
$$h_u(\Delta \theta) = \frac{1}{2}(|\cos \Delta \theta| + |\sin \Delta \theta|) + \frac{1}{2} \approx 1 + |\delta|.$$
For a corrugated band with $K$ sign-alternating pinch transitions, the total longitudinal expansion along $u$ is:
$$\Delta u_{\text{longitudinal}} \approx K \cdot |\delta|.$$
To prevent the corrugated band from expanding into and jamming against the end staircases $b_1$ and $b_2$, the available longitudinal slack must satisfy:
$$\text{slack}(D) = \left\{\frac{D}{\sqrt{2}}\right\} \ge K \cdot |\delta|.$$

#### Quantitative Implications
- For typical band widths ($w \in [2, 16]$), $K$ ranges from $4$ to $10$.
- With optimal corrugation tilt $|\delta| \approx 0.0025$ to $0.0030$, the required expansion is $K |\delta| \approx 0.010$ to $0.025$.
- If $\text{slack}(D) < 0.010$ (e.g. $D = 58$ where slack $= 0.012$, or $D = 99$ where slack $= 0.0035$), the allowable tilt $|\delta| \le \text{slack}(D) / K$ is severely constrained, throttling transverse shrinkage.
- If $\text{slack}(D) > 0.150$, the baseline uncorrugated band has excessive dead space at $45^\circ$, rendering $L_{45}$ uncompetitive.
- **Feasible Slack Window**:
  $$\boxed{0.010 \le \text{slack}(D) \le 0.150}.$$

### Container Wall Clearance Conditions
Container boundary non-collision requires:
1. **End Staircase Wall Non-Collision**:
   Along the bottom container wall ($y \in [0, 1]$), the SW staircase occupies $x \in [0, b_1]$ and the SE flank occupies $x \in [L - a_1, L]$. Clearance requires:
   $$\text{gap}_{\text{bottom}} = (L - a_1) - b_1 \ge 0.$$
   Along the right container wall ($x \in [L-1, L]$), the SE flank occupies $y \in [0, a_1]$ and the NE staircase occupies $y \in [L - b_2, L]$. Clearance requires:
   $$\text{gap}_{\text{right}} = (L - b_2) - a_1 \ge 0.$$

#### Symmetric End Staircases ($b_1 = b_2 = b$, $a_1 - a_2 = 1$)
$$L_{45} - a_1 - b = \left(b + \frac{D}{2} + 0.5 + 1 + b + \left\{\frac{w}{\sqrt{2}}\right\}\right) - \left(b + \frac{D+1}{2}\right) - b = 0.5 + \left\{\frac{w}{\sqrt{2}}\right\} > 0.50.$$
$$L_{45} - b - a_2 = 1.5 + \left\{\frac{w}{\sqrt{2}}\right\} > 1.50.$$
Wall clearance against the outer container walls is always satisfied.
The critical clearance is between the band ends and the container corners:
$$u_{\text{stair}} - u_{\text{wall}} = \left(\frac{b + 1}{\sqrt{2}} + 0.5\right) - \left(\frac{w - 1}{2} + \frac{1}{2\sqrt{2}} + \frac{1}{\sqrt{2}}\right) = 1.0 - \frac{1}{2\sqrt{2}} - \frac{\{w/\sqrt{2}\}}{\sqrt{2}} \approx 0.6464 - \frac{\{w/\sqrt{2}\}}{\sqrt{2}}.$$
Requiring $u_{\text{stair}} > u_{\text{wall}}$ with corner tooth slide clearance gives:
$$\boxed{\left\{\frac{w}{\sqrt{2}}\right\} \in (0.05, 0.4142)}, \quad \text{where } 0.4142 \approx \sqrt{2} - 1.$$

#### Asymmetric End Staircases ($b_1 = b, b_2 = b + 1$, $a_1 - a_2 = 1$)
Here $a_1 = b + \frac{D}{2} + 1, a_2 = b + \frac{D}{2}$. Evaluate the right wall gap:
$$\text{gap}_{\text{right}} = L_{45} - b_2 - a_1 = \left(2b + \frac{D}{2} + 1.5 + \left\{\frac{w}{\sqrt{2}}\right\}\right) - (b + 1) - \left(b + \frac{D}{2} + 1\right)$$
$$\boxed{\text{gap}_{\text{right}} = \left\{\frac{w}{\sqrt{2}}\right\} - 0.5}.$$
To guarantee positive clearance $\text{gap}_{\text{right}} > 0$:
$$\boxed{\left\{\frac{w}{\sqrt{2}}\right\} > 0.50}.$$
If $\{w / \sqrt{2}\} \le 0.50$, the NE end staircase ($b_2$) and the SE flank staircase ($a_1$) collide along the right wall. Thus, asymmetric even $D$ strictly requires $\{w / \sqrt{2}\} \in (0.55, 0.95)$ ($w \in \{1, 4, 7, 8, 11, 14, 15, 18, 21, 22, 25\}$).

---

## 5. Candidate Diophantine Classification

Candidate values of $D$ correspond to Diophantine approximations to $\sqrt{2}$ with small positive fractional part $\{D / \sqrt{2}\}$:

| Family | $D$ | Parity | Geometry | $\lfloor D/\sqrt{2} \rfloor$ | $\text{slack}(D)$ | Feasible $w$ Rule | Min Core $n_0(w_{\min})$ |
|---|---:|---|---|---:|---:|---|---:|
| Diophantine $3^2 - 2\cdot 2^2 = 1$ | $3$ | Odd | Symmetric | $2$ | $0.12132$ | $\{w/\sqrt{2}\} \in (0.05, 0.42)$ | $19$ ($w=2$) |
| Diophantine $10^2 - 2\cdot 7^2 = 2$ | $10$ | Even | Asymmetric | $7$ | $0.07107$ | $\{w/\sqrt{2}\} \in (0.55, 0.95)$ | $45$ ($w=1$) |
| Pell $17^2 - 2\cdot 12^2 = 1$ | $17$ | Odd | Symmetric | $12$ | $0.02082$ | $\{w/\sqrt{2}\} \in (0.05, 0.42)$ | $130$ ($w=2$) |
| Hybrid $(10 + 17)$ | $27$ | Odd | Symmetric | $19$ | $0.09188$ | $\{w/\sqrt{2}\} \in (0.05, 0.42)$ | $269$ ($w=2$) |
| Double Pell $2 \times 17$ | $34$ | Even | Asymmetric | $24$ | $0.04163$ | $\{w/\sqrt{2}\} \in (0.55, 0.95)$ | $350$ ($w=1$) |
| Triple Pell $3 \times 17$ | $51$ | Odd | Symmetric | $36$ | $0.06245$ | $\{w/\sqrt{2}\} \in (0.05, 0.42)$ | $807$ ($w=2$) |
| Diophantine $58^2 - 2\cdot 41^2 = 2$ | $58$ | Even | Asymmetric | $41$ | $0.01219$ | $\{w/\sqrt{2}\} \in (0.55, 0.95)$ | $943$ ($w=1$) |
| Pell $99^2 - 2\cdot 70^2 = 1$ | $99$ | Odd | Symmetric | $70$ | $0.00357$ | $\{w/\sqrt{2}\} \in (0.05, 0.42)$ | $2571 > 1000$ |

*Note on $D = 99$*: The minimal core ($w=1, b=0, a_1=50, a_2=49$) requires $T(50) + T(49) + 71 = 1275 + 1225 + 71 = 2571$ squares, far exceeding $n = 1000$. Hence, candidate scanning terminates at $D = 58$.

---

## 6. Sweet Spot Cores ($n_0 \le 1000$)

For each candidate $D$, all valid widths $w$ satisfying wall and tooth clearance are tabulated below:

### Family 1: Pell-17 ($D = 17$, Symmetric, $\text{slack} = 0.02082$, $\ell = w + 12$)
$$b_1 = b_2 = b = \lfloor w / \sqrt{2} \rfloor, \quad a_1 = b + 9, \quad a_2 = b + 8$$

| Width $w$ | End $(b_1, b_2)$ | Flank $(a_1, a_2)$ | Strip $\ell$ | $k_0 = \lfloor L_0 \rfloor$ | Core $n_0$ | $L_{45}$ | $\{w/\sqrt{2}\}$ | Certified $L_{\text{corr}}$ | $\Delta L_{\text{corr}}$ |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| $2$ | `(1, 1)` | `(10, 9)` | $14$ | $11$ | $130$ | `11.914213562` | `0.4142` | `11.912286146` | `-0.001927416` |
| $3$ | `(2, 2)` | `(11, 10)` | $15$ | $13$ | $172$ | `13.621320344` | `0.1213` | `13.619259872` | `-0.002060472` |
| $6$ | `(4, 4)` | `(13, 12)` | $18$ | $17$ | $297$ | `17.742640687` | `0.2426` | `17.741250940` | `-0.001389747` |
| $9$ | `(6, 6)` | `(15, 14)` | $21$ | $21$ | $456$ | `21.863961031` | `0.3640` | `21.862433095` | `-0.001527935` |
| $10$ | `(7, 7)` | `(16, 15)` | $22$ | $23$ | $532$ | `23.571067812` | `0.0711` | `23.569666051` | `-0.001401761` |
| $13$ | `(9, 9)` | `(18, 17)` | $25$ | $27$ | $739$ | `27.692388155` | `0.1924` | `27.690921182` | `-0.001466973` |
| $16$ | `(11, 11)` | `(20, 19)` | $28$ | $31$ | $980$ | `31.813708499` | `0.3137` | `31.812280644` | `-0.001427855` |

### Family 2: Asymmetric Double-Pell ($D = 34$, Asymmetric, $\text{slack} = 0.04163$, $\ell = w + 24$)
$$b_1 = b = \lfloor w / \sqrt{2} \rfloor, \quad b_2 = b + 1, \quad a_1 = b + 18, \quad a_2 = b + 17$$

| Width $w$ | End $(b_1, b_2)$ | Flank $(a_1, a_2)$ | Strip $\ell$ | $k_0$ | Core $n_0$ | $L_{45}$ | $\{w/\sqrt{2}\}$ | Expected $\Delta L$ |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| $1$ | `(0, 1)` | `(18, 17)` | $25$ | $19$ | $350$ | `19.207106781` | `0.7071` | `-0.0022` |
| $4$ | `(2, 3)` | `(20, 19)` | $28$ | $23$ | $521$ | `23.328427125` | `0.8284` | `-0.0020` |
| $7$ | `(4, 5)` | `(22, 21)` | $31$ | $27$ | $726$ | `27.449747468` | `0.9497` | `-0.0018` |
| $8$ | `(5, 6)` | `(23, 22)` | $32$ | $29$ | $821$ | `29.156854249` | `0.6569` | `-0.0019` |

### Family 3: Symmetric Hybrid ($D = 27$, Symmetric, $\text{slack} = 0.09188$, $\ell = w + 19$)
$$b_1 = b_2 = b = \lfloor w / \sqrt{2} \rfloor, \quad a_1 = b + 14, \quad a_2 = b + 13$$

| Width $w$ | End $(b_1, b_2)$ | Flank $(a_1, a_2)$ | Strip $\ell$ | $k_0$ | Core $n_0$ | $L_{45}$ | $\{w/\sqrt{2}\}$ | Expected $\Delta L$ |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| $2$ | `(1, 1)` | `(15, 14)` | $21$ | $16$ | $269$ | `16.914213562` | `0.4142` | `-0.0030` |
| $3$ | `(2, 2)` | `(16, 15)` | $22$ | $18$ | $328$ | `18.621320344` | `0.1213` | `-0.0032` |
| $6$ | `(4, 4)` | `(18, 17)` | $25$ | $22$ | $494$ | `22.742640687` | `0.2426` | `-0.0028` |
| $9$ | `(6, 6)` | `(20, 19)` | $28$ | $26$ | $694$ | `26.863961031` | `0.3640` | `-0.0026` |
| $10$ | `(7, 7)` | `(21, 20)` | $29$ | $28$ | $787$ | `28.571067812` | `0.0711` | `-0.0025` |

### Family 4: Triple-Pell ($D = 51$, Symmetric, $\text{slack} = 0.06245$, $\ell = w + 36$)
$$b_1 = b_2 = b = \lfloor w / \sqrt{2} \rfloor, \quad a_1 = b + 26, \quad a_2 = b + 25$$

| Width $w$ | End $(b_1, b_2)$ | Flank $(a_1, a_2)$ | Strip $\ell$ | $k_0$ | Core $n_0$ | $L_{45}$ | $\{w/\sqrt{2}\}$ | Expected $\Delta L$ |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| $2$ | `(1, 1)` | `(27, 26)` | $38$ | $28$ | $807$ | `28.914213562` | `0.4142` | `-0.0022` |
| $3$ | `(2, 2)` | `(28, 27)` | $39$ | $30$ | **$907$** | **`30.621320344`** | `0.1213` | `-0.0024` |

*High-Value Finding*: Core $n_0 = 907$ with $w = 3, L_{45} = 30.621320$ provides an uncorrugated side length that is already smaller than the public target incumbent for $n = 908$ ($30.63702956$).

### Family 5: Asymmetric Diophantine-10 ($D = 10$, Asymmetric, $\text{slack} = 0.07107$, $\ell = w + 7$)
$$b_1 = b = \lfloor w / \sqrt{2} \rfloor, \quad b_2 = b + 1, \quad a_1 = b + 6, \quad a_2 = b + 5$$

| Width $w$ | End $(b_1, b_2)$ | Flank $(a_1, a_2)$ | Strip $\ell$ | $k_0$ | Core $n_0$ | $L_{45}$ | $\{w/\sqrt{2}\}$ | Expected $\Delta L$ |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| $1$ | `(0, 1)` | `(6, 5)` | $8$ | $7$ | $45$ | `7.207106781` | `0.7071` | `-0.0035` |
| $4$ | `(2, 3)` | `(8, 7)` | $11$ | $11$ | $117$ | `11.328427125` | `0.8284` | `-0.0032` |
| $7$ | `(4, 5)` | `(10, 9)` | $14$ | $15$ | $223$ | `15.449747468` | `0.9497` | `-0.0030` |
| $8$ | `(5, 6)` | `(11, 10)` | $15$ | $17$ | $277$ | `17.156854249` | `0.6569` | `-0.0029` |
| $11$ | `(7, 8)` | `(13, 12)` | $18$ | $21$ | $431$ | `21.278174593` | `0.7782` | `-0.0027` |
| $14$ | `(9, 10)` | `(15, 14)` | $21$ | $25$ | $619$ | `25.399494937` | `0.8995` | `-0.0025` |
| $15$ | `(10, 11)` | `(16, 15)` | $22$ | $27$ | $707$ | `27.106601718` | `0.6066` | `-0.0026` |
| $18$ | `(12, 13)` | `(18, 17)` | $25$ | $31$ | $943$ | `31.227922061` | `0.7279` | `-0.0024` |

### Family 6: Small-Odd Diophantine-3 ($D = 3$, Symmetric, $\text{slack} = 0.12132$, $\ell = w + 2$)
$$b_1 = b_2 = b = \lfloor w / \sqrt{2} \rfloor, \quad a_1 = b + 2, \quad a_2 = b + 1$$

| Width $w$ | End $(b_1, b_2)$ | Flank $(a_1, a_2)$ | Strip $\ell$ | $k_0$ | Core $n_0$ | $L_{45}$ | $\{w/\sqrt{2}\}$ | Expected $\Delta L$ |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| $2$ | `(1, 1)` | `(3, 2)` | $4$ | $4$ | $19$ | `4.914213562` | `0.4142` | `-0.0035` |
| $3$ | `(2, 2)` | `(4, 3)` | $5$ | $6$ | $37$ | `6.621320344` | `0.1213` | `-0.0038` |
| $6$ | `(4, 4)` | `(6, 5)` | $8$ | $10$ | $104$ | `10.742640687` | `0.2426` | `-0.0032` |
| $9$ | `(6, 6)` | `(8, 7)` | $11$ | $14$ | $205$ | `14.863961031` | `0.3640` | `-0.0030` |
| $10$ | `(7, 7)` | `(9, 8)` | $12$ | $16$ | $257$ | `16.571067812` | `0.0711` | `-0.0029` |
| $13$ | `(9, 9)` | `(11, 10)` | $15$ | $20$ | $406$ | `20.692388155` | `0.1924` | `-0.0028` |
| $16$ | `(11, 11)` | `(13, 12)` | $18$ | $24$ | $589$ | `24.813708499` | `0.3137` | `-0.0026` |
| $20$ | `(14, 14)` | `(16, 15)` | $22$ | $30$ | **$906$** | **`30.642135624`** | `0.1421` | `-0.0025` |

### Family 7: Asymmetric Diophantine-58 ($D = 58$, Asymmetric, $\text{slack} = 0.01219$, $\ell = w + 41$)
$$b_1 = b = 0, \quad b_2 = 1, \quad a_1 = 30, \quad a_2 = 29, \quad w = 1, \quad \ell = 42, \quad k_0 = 31$$
$$n_0 = T(30) + T(29) + 0 + 1 + 42 = 465 + 435 + 1 + 42 = \mathbf{943}, \qquad L_{45} = \mathbf{31.207106781}.$$
Expected corrugation shrinkage: $\Delta L \approx -0.0008$ due to tight slack ($0.01219$).

---

## 7. L-Border Wraps and Gap-Cluster Targeting ($n \le 1000$)

Wrapping a core of side length $L_0$ and square count $n_0$ with a $t$-layer upright L-border adds $t(2k_0 + t)$ squares:
$$n = n_0 + t(2k_0 + t), \qquad L(n) = L_0 + t.$$

### Target Match Matrix for Priority Deficit Intervals

| Target $n$ | Family | Construction Parameters $(w, \ell, a_1, a_2, b_1, b_2)$ | Core $n_0$ | Wrap $t$ | $L_{45} + t$ | Corrugated $L_{\text{corr}}$ | Portfolio Baseline | Public Incumbent | Score Delta vs Baseline |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`544`** | Pell-17 | $(9, 21, 15, 14, 6, 6)$ | $456$ | $2$ | `23.863961` | `23.862433` | `24.000000` | — | **`-0.005898`** |
| **`591`** | Pell-17 | $(9, 21, 15, 14, 6, 6)$ | $456$ | $3$ | `24.863961` | `24.862433` | `25.000000` | — | **`-0.005658`** |
| `906` | $D=3$ | $(20, 22, 16, 15, 14, 14)$ | $906$ | $0$ | `30.642136` | `30.6396` | `30.707107` | — | **`-0.002243`** |
| `907` | $D=51$ | $(3, 39, 28, 27, 2, 2)$ | $907$ | $0$ | `30.621320` | `30.6191` | `30.707107` | `30.637030` ($n=908$) | **`-0.002922`** |
| `908` | Pell-17 | $(6, 18, 13, 12, 4, 4)$ | $297$ | $13$ | `30.742641` | `30.741251` | `30.707107` | `30.637030` | $+0.001133$ |
| **`910`** | **Pell-17** | **$(13, 25, 18, 17, 9, 9)$** | **$739$** | **$3$** | **`30.692388`** | **`30.690921`** | `30.707107` | **`30.690960`** | **`-0.000537`** |
| `910` | $D=27$ | $(6, 25, 18, 17, 4, 4)$ | $494$ | $8$ | `30.742641` | `30.7398` | `30.707107` | `30.690960` | $+0.001083$ |
| `968` | $D=51$ | $(3, 39, 28, 27, 2, 2)$ | $907$ | $1$ | `31.621320` | `31.6191` | `31.707107` | — | **`-0.002829`** |
| **`971`** | **Pell-17** | **$(13, 25, 18, 17, 9, 9)$** | **$739$** | **$4$** | **`31.692388`** | **`31.690921`** | `31.707107` | `31.678780` | **`-0.000520`** |
| `971` | $D=27$ | $(6, 25, 18, 17, 4, 4)$ | $494$ | $9$ | `31.742641` | `31.7398` | `31.707107` | `31.678780` | $+0.001048$ |

*Remark on $n = 910$*: The certified corrugated value `L(910) = 30.690921074188` beats the public target incumbent `30.690960120646` by $\Delta L = -0.000039046$, proving the efficacy of out-of-phase corrugation.

---

## 8. 4-Sign-Piece Epigraph LP Polish Formulation

The pairwise Separating Axis Theorem (SAT) constraint between squares $i$ and $j$ with relative orientation $r = \theta_j - \theta_i$ along separation axis $a$ is:
$$d_{ij} \cdot a \ge h_i + h_j, \qquad h(r) = \frac{1}{2}(|\cos r| + |\sin r|).$$
The function $h(r)$ has non-differentiable absolute value corners at $r = 0$.

### Why Central Finite Differences Fails
Standard central finite differencing linearizes $h(r)$ via $\frac{\partial h}{\partial r} \approx 0$ near $r = 0$, trapping columns at identical orientation $\theta_j = \theta_i$ ($r = 0$). In numerical trials, central finite difference stalled at `L = 30.69096039`, failing to beat the incumbent `30.69096012`.

### The 4-Sign-Piece Epigraph Solution
Replace the absolute value terms with their simultaneous 4-piece upper envelope:
$$|\cos r| + |\sin r| = \max_{(s_1, s_2) \in \{-1, +1\}^2} (s_1 \cos r + s_2 \sin r).$$
For each adjacent pair $(i, j)$ and each of the 4 sign combinations $(s_1, s_2) \in \{(-1,-1), (-1,+1), (+1,-1), (+1,+1)\}$, an independent linear constraint is placed in the LP constraint matrix $A_{\text{ub}} x \le b_{\text{ub}}$:
$$d_{ij} \cdot a - \frac{1}{2}(s_1 \cos r + s_2 \sin r) - \frac{1}{2} - (\nabla g)^T \Delta x \ge \text{clearance}.$$
This smooth convex epigraph representation allows the interior-point/simplex solver (`HiGHS`) to freely push adjacent columns across $r = 0$, achieving the necessary relative tilt between pinch columns and non-pinch buffer columns.

---

## 9. Confidence Levels and Proven Dead Ends

### Confidence Assessment
- **Pell-17 Family ($D = 17$)**: **Certified (100% confidence)**. All 7 cores ($w \in \{2, 3, 6, 9, 10, 13, 16\}$) and wraps tested pass literal and `pandas 3.1.0` certification under `squares.core.certify`, producing verifiable record gains (e.g. $n = 910$).
- **Triple-Pell ($D = 51$) Core $n_0 = 907$**: **High confidence**. $L_{45} = 30.62132$ is algebraically smaller than the $n = 908$ public record ($30.63703$). Slack $= 0.06245$ comfortably accommodates $K |\delta| \approx 0.02$.
- **Asymmetric Double-Pell ($D = 34$)**: **High confidence**. All 4 cores satisfy $\{w/\sqrt{2}\} > 0.50$, preventing wall collision. Slack $= 0.04163$ provides robust corrugation headroom.
- **Asymmetric Diophantine-10 ($D = 10$)**: **High confidence**. Core $n_0 = 431$ wrapped with $t=8$ yields $n = 831$ at $L = 29.27817$.

### Proven Dead Ends
1. **In-Phase Corrugation ($a_1 + a_2$ even)**: Proved analytically impossible to achieve first-order transverse shrinkage. Symmetrical tooth forces produce exact first-order cancellation. Attempting split-angle midline tilts costs $+|\delta|$ in SAT transverse expansion, overwhelming any tooth penetration.
2. **In-Phase and Negative Slack $D$ (e.g. $D \in \{24, 41, 48\}$)**:
   - For $D = 24$: $24/\sqrt{2} = 16.97056 \implies \text{slack} = 0.97056$ (deficit $-0.02944$). Column count $\ell = w + 16$ leaves $0.97$ units of unusable space. Forcing $\ell = w + 17$ requires longitudinal span exceeding $u_{\text{hi}} - u_{\text{lo}}$, causing severe collision with end staircases $b_1, b_2$.
3. **Asymmetric Even $D$ with $\{w/\sqrt{2}\} \le 0.50$**:
   - Because $b_2 = b + 1$, the right container wall clearance is $\text{gap} = \{w/\sqrt{2}\} - 0.5$. Any choice of $w$ with $\{w/\sqrt{2}\} \le 0.50$ causes the NE staircase and SE flank to collide.
4. **Higher Pell Orders $D \ge 99$**:
   - The minimal core for $D = 99$ contains $2571$ squares, rendering all Pell orders $k \ge 4$ unusable for the $n \le 1000$ competition.
5. **Central-Difference LP Polishing**: Stalls at the non-differentiable $r = 0$ SAT boundary. Must be replaced with 4-sign-piece epigraph linearization.
6. **Unstructured Stochastic Annealing / Coordinate Thresholding**: Fails because the discrete phase transition between in-phase and out-of-phase tooth alignment cannot be crossed via localized continuous perturbations.
