# Multi-Pell Cores POC 3: Negative/Near-Zero Slack Pell Cores ($D=7, 41$)

## 1. Executive Summary

- **Objective**: Investigate whether semi-convergent Pell cores with near-zero or negative longitudinal slack ($D=7$, $D=41$) where $\ell = w + \lceil D/\sqrt{2} \rceil$ can overcome their initial longitudinal deficit through 4-sign-piece epigraph LP transverse flank tooth penetration and achieve corrugation shrinkage.
- **Falsification Gate Outcome**: **FALSIFIED AT THE GATE**.
  - For $D=7$ ($w=4, \ell=9, a_1=6, a_2=5, b=2, L_{45}=9.328427$): HiGHS LP is strictly **infeasible** within $L_{45} + 0.15$. Allowing container dilation expands the container to $L = 9.509957$ ($\Delta L = +0.181530 \gg 0$), yielding zero corrugation shrinkage.
  - For $D=41$ ($w=8, \ell=37, a_1=23, a_2=23, b=5, L_{45}=29.656854$): HiGHS LP is strictly **infeasible** (HiGHS Status 8) due to hyper-static column jamming between flank and corner staircases.
- **Core Mechanism Insight**: Longitudinal slack along the diagonal band is governed by the Pell equation $D^2 - 2\ell^2 = \pm 1$.
  - When $D^2 - 2\ell^2 = +1$ (e.g. $D=17, \ell_0=12$), the diagonal slack is strictly **positive** ($\sigma = +0.020815$). This positive longitudinal breathing room allows out-of-phase pinch columns to tilt by $\pm\delta$ without jamming.
  - When $D^2 - 2\ell^2 = -1$ (e.g. $D=7, \ell_0=5$ and $D=41, \ell_0=29$), the diagonal slack is **negative** ($\sigma_7 = -0.050253$, $\sigma_{41} = -0.008622$).
  - Forcing $\ell = w + \lceil D/\sqrt{2} \rceil$ columns into this negative-slack corridor forces the mean longitudinal column pitch strictly below $1.0000$ ($0.9937$ for $D=7$). Because congruent unit squares in parallel columns require a center-to-center longitudinal clearance of at least $1.0000$, transverse flank tooth penetration cannot compress the metric along the band axis, causing mandatory geometric infeasibility.

---

## 2. Diophantine Pell Slack Spectrum

In an out-of-phase corrugated staircase band with parameters $(w, \ell, a_1, a_2, b_1, b_2)$:
The container side length is:
$$L_0 = \frac{1}{2}(a_1 + a_2) + 1.0 + \frac{w}{\sqrt{2}}$$
The corner staircases $b_1, b_2$ sit at the SW corner $(0, 0)$ and NE corner $(L_0, L_0)$.
The diagonal bounding range along the band axis $u = (x+y)/\sqrt{2}$ available to the tilted band is:
$$u_{\text{hi}} - u_{\text{lo}} = \sqrt{2} L_0 - \frac{b_1 + b_2 + 2}{\sqrt{2}} - 1.0 = \frac{a_1 + a_2 - b_1 - b_2}{\sqrt{2}} + w - 1.0$$

Setting $a_1 + a_2 - b_1 - b_2 = D$:
$$u_{\text{hi}} - u_{\text{lo}} = w + \frac{D}{\sqrt{2}} - 1.0$$

To place $\ell$ tilted columns along the diagonal band, the required longitudinal distance between the first and last column centers is:
$$\Delta u_{\text{req}} = \ell - 1$$

If we set $\ell = w + \ell_0$, then the available diagonal space versus the required space yields the **longitudinal slack**:
$$\sigma(D, \ell_0) = (u_{\text{hi}} - u_{\text{lo}}) - (w + \ell_0 - 1) = \frac{D}{\sqrt{2}} - \ell_0$$

| Family | $D$ | $\ell_0 = \lceil D/\sqrt{2} \rceil$ | $D/\sqrt{2}$ | Longitudinal Slack $\sigma$ | Diophantine Residue $D^2 - 2\ell_0^2$ | Status |
|:---|---:|---:|---:|---:|---:|:---|
| **Pell-7** | 7 | 5 | $4.949747$ | **$-0.050253$** (Deficit) | $-1$ | **Infeasible (Falsified)** |
| **Pell-17** | 17 | 12 | $12.020815$ | **$+0.020815$** (Positive) | $+1$ | **Certified Win ($n=910$)** |
| **Pell-41** | 41 | 29 | $28.991378$ | **$-0.008622$** (Deficit) | $-1$ | **Infeasible (Falsified)** |
| **Pell-99** | 99 | 70 | $70.003571$ | **$+0.003571$** (Positive) | $+1$ | Candidate (Ultra-large $n$) |

Notice that the sign of the slack $\sigma$ is uniquely determined by the sign of the Pell residue $D^2 - 2\ell_0^2$:
- When $D^2 - 2\ell_0^2 = -1$, $D/\sqrt{2} < \ell_0 \implies \sigma < 0$.
- When $D^2 - 2\ell_0^2 = +1$, $D/\sqrt{2} > \ell_0 \implies \sigma > 0$.

---

## 3. The Longitudinal Pitch Compression Theorem

### 3.1 Statement of the Problem
The hypothesis tested in POC 3 was:
> Can transverse tooth penetration along the flank staircases ($a_1, a_2$) allow columns to shift transversely into staircase cavities, thereby compressing the longitudinal column pitch $\Delta u$ and fitting $\ell$ columns into a negative-slack corridor?

### 3.2 Geometric Infeasibility Proof
1. **Longitudinal Center Separation**:
   Consider two squares $S_1, S_2$ in adjacent columns $c, c+1$ at the same transverse band index $s \in \{0, \dots, w-1\}$.
   Their nominal centers are $\mathbf{q}_1 = (u_c, v_s)$ and $\mathbf{q}_2 = (u_{c+1}, v_s)$ in band coordinates $(u, v)$.
   The Euclidean distance between them is:
   $$\|\mathbf{q}_2 - \mathbf{q}_1\| = |u_{c+1} - u_c| = \Delta u$$
   For non-overlapping unit squares with nearly parallel orientations ($|\theta_2 - \theta_1| < 0.02$ rad), the Separating Axis Theorem along the column normal requires:
   $$\Delta u \ge \cos(\Delta\theta) \approx 1 - \frac{(\Delta\theta)^2}{2} \approx 1 - 2 \times 10^{-4} \approx 1.0000$$

2. **Available Pitch Under Negative Slack**:
   With $\ell$ columns distributed uniformly between $u_{\text{lo}}$ and $u_{\text{hi}}$:
   $$\overline{\Delta u} = \frac{u_{\text{hi}} - u_{\text{lo}}}{\ell - 1} = \frac{(\ell - 1) + \sigma}{\ell - 1} = 1 + \frac{\sigma}{\ell - 1}$$
   - For $D=7$ ($w=4, \ell=9$):
     $$\overline{\Delta u} = 1 + \frac{-0.050253}{8} = 1 - 0.006282 = 0.993718 < 1.0000$$
   - For $D=41$ ($w=8, \ell=37$):
     $$\overline{\Delta u} = 1 + \frac{-0.008622}{36} = 1 - 0.000240 = 0.999760 < 1.0000$$

3. **Impotence of Transverse Tooth Penetration**:
   Tooth penetration shifts square centers along the transverse axis $v$ by $\Delta v(s)$.
   The center separation between adjacent column squares becomes:
   $$\|\mathbf{q}_2 - \mathbf{q}_1\|^2 = (\Delta u)^2 + (\Delta v)^2$$
   While $(\Delta v)^2 \ge 0$ increases the center Euclidean distance, the SAT projection along the transverse axis normal $\mathbf{n}_v$ remains:
   $$\text{gap}_v = |\Delta v| - 1.0 < 0 \quad (\text{for small transverse shifts})$$
   Along the longitudinal axis $\mathbf{n}_u$, the separation gap is:
   $$\text{gap}_u = \Delta u - 1.0 = -0.006282 < 0$$
   Because the squares are congruent unit squares, any non-zero $\Delta v$ that could separate them requires $|\Delta v| \ge 1.0$ (displacing an entire square width), which is impossible inside a narrow width-$w$ band.
   
4. **Conclusion**:
   Transverse tooth penetration cannot relieve longitudinal compression. The columns are mechanically jammed, making positive longitudinal slack an absolute physical requirement.

---

## 4. Empirical Evaluation and Falsification Gate

### 4.1 Candidate Seeds Evaluated
1. **$D=7$ Candidate Seed**:
   - Parameters: $w=4, b=2, a_1=6, a_2=5, \ell=9$.
   - Nominal side: $L_{45} = 9.328427$.
   - Square count: $n_0 = 78$.
   - Longitudinal slack: $\sigma = -0.050253$.
   - Measured column pitch: $\text{mean} = 0.9948$, $\text{min} = 0.9937$.
   - **LP Result**: Infeasible under `_lp(q, L45 + 0.15)`.
   - **Dilation Result**: With upper bound relaxed to $L_{45} + 0.50$, the container dilates to $L = 9.509957$ ($\Delta L = +0.181530$).

2. **$D=41$ Candidate Seed**:
   - Parameters: $w=8, b=5, a_1=23, a_2=23, \ell=37$.
   - Nominal side: $L_{45} = 29.656854$.
   - Square count: $n_0 = 878$.
   - Longitudinal slack: $\sigma = -0.008622$.
   - Measured column pitch: $\text{mean} = 0.9998$, $\text{min} = 0.9984$.
   - **LP Result**: Infeasible under `_lp(q, L45 + 0.15)` (HiGHS Status 8).

### 4.2 Falsification Gate Criterion
Under Q5 §4, Rank 1:
$$\text{Gate Criterion}: \quad L_{\text{corr}} < L_{45} - 0.00080 \quad \text{and} \quad \text{LP is feasible within 5s}$$

Both $D=7$ and $D=41$ fail both conditions:
- LP is infeasible within the nominal container bound.
- Relaxation forces $\Delta L > 0$ (dilation rather than contraction).

**Final Verdict**: Candidate Pell families $D \in \{7, 41\}$ are **falsified**. Research effort should remain focused exclusively on positive-slack Pell convergents ($D=17$).
