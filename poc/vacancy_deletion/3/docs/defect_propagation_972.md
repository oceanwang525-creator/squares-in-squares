# Defect Propagation on Public $n = 972$ Incumbent (POC 3)

## 1. Executive Summary

This POC investigates the defect propagation mechanism applied to the public incumbent packing at $n = 972$.
While synthetic pinch-column corrugation on equal-count configurations (`poc/pell_corrugation`) yields $\Delta L \approx -0.001467$, the public leader for $n = 971$ achieves a dramatically larger shrinkage of $\Delta L = -0.013608$ ($9.28\times$ larger).
Anatomical dissection reveals that $n = 971$ is obtained by introducing a vacancy defect into $n = 972$, eliminating the longitudinal contact jamming along the band columns and permitting the surrounding upright boundary to contract inward.

POC 3 implements:
1. Deconstruction of the public $n = 972$ incumbent ($L = 31.69238816$) into its 739 upright squares and 233 tilted squares across 13 columns.
2. Targeted deletion of $m \in \{1, 2\}$ squares from the ends of the 13 columns.
3. Longitudinal defect propagation shifting column squares into the vacancy void.
4. Active container shrink LP polishing using the 4-sign-piece epigraph formulation.

---

## 2. Structural Anatomy of $n = 972$

The public incumbent at $n = 972$ has certified side length:
$$L = 31.692388155428386$$
This matches the exact algebraic formula $L_{45}(w=13) + 4$:
$$L_{45} = 1 + \frac{18 + 17}{2} + \frac{13}{\sqrt{2}} = 27.692388155425117 \implies L = L_{45} + 4 = 31.692388155428386.$$

### Component Partition
- **739 Upright Squares ($\theta = 0$):**
  - **231 Squares (SW Staircase):** Complete triangular staircase of side $k = 21$:
    $$\sum_{i=1}^{21} i = \frac{21 \times 22}{2} = 231 \text{ squares}.$$
  - **508 Squares (NE Upright Block):** Incorporating the NE flank staircase and the 4-layer upright outer L-border ($t = 4$).
- **233 Tilted Squares ($\theta \approx 45^\circ$):**
  - Grouped into exactly 13 columns along transverse axis $v = (-x + y)/\sqrt{2}$.
  - Strip length profile: $(13, 13, 21, 21, 21, 21, 21, 21, 21, 21, 13, 13, 13)$.
  - Total count: $5 \times 13 + 8 \times 21 = 65 + 168 = 233$ squares.

---

## 3. The Defect Propagation Mechanism

### 3.1 Jamming Condition in $n = 972$
In the incumbent $n = 972$, every tilted column is jammed between:
1. The bottom-left flank staircase or container corner.
2. The top-right flank staircase or container wall.

Because all columns have full-rank contact graphs, infinitesimal translations are locked by rigid boundary contacts.

### 3.2 Vacancy Creation and Longitudinal Propagation
Deleting square $k$ at the end of a column creates a spatial void along $u = (x + y)/\sqrt{2}$ of dimension $\Delta u \approx 1.0$.
Squares in that column can translate longitudinally towards the void by $\delta u \approx 0.15\text{--}0.25$.
This shifts the vacancy defect from the interior/column end to the boundary interface with the upright staircase teeth.

### 3.3 Active Container Shrink LP
Once the rigid jamming condition along column $c$ is eliminated:
1. Flank staircase teeth can collapse inward by $\delta u \approx 1 - 1/\sqrt{2} \approx 0.2929$.
2. Neighboring tilted columns $c \pm 1$ can tilt or shift transversely into the column gap.
3. The 4-sign-piece LP minimizes container side length $L$ under full SAT non-overlap constraints:
   $$\min_{\Delta \mathbf{x}, \Delta \boldsymbol{\theta}, \Delta L} \Delta L$$
   $$\text{s.t. } \mathbf{x}_i \pm \mathbf{h}_i \in [0, L] \times [0, L], \quad \text{SAT}(i, j) \ge \text{clearance}, \quad \|\Delta \mathbf{x}\|_\infty \le \delta_{\text{pos}}, \quad \|\Delta \boldsymbol{\theta}\|_\infty \le \delta_{\text{ang}}.$$

---

## 4. Targets and Benchmarks

| Target | Incumbent / Baseline $L$ | Source Construction | Mechanism | Expected / Measured Gain |
|:---:|:---:|:---:|:---:|:---:|
| **$n = 971$** | `31.70710678` (Portfolio)<br>`31.67877988` (Public) | $n = 972$ minus 1 tilted end square | Longitudinal vacancy propagation + 4-sign LP | $\Delta L \approx -0.0162$ vs portfolio; targets public $31.678780$ |
| **$n = 970$** | `32.00000000` (Grid)<br>`31.70710678` (Fallback) | $n = 972$ minus 2 tilted end squares | Dual column vacancy propagation | $\Delta L \ge -0.015$ below fallback |

---

## 5. Verification standard

All generated solutions must satisfy:
1. Literal CSV transport certification via `squares._official_metric._parse_submission_csv` and `squares.core.certify`.
2. Pandas 3.1.0 read/write roundtrip certification.
3. Strict positive clearance ($\ge 5 \times 10^{-9}$) across all square pairs.
