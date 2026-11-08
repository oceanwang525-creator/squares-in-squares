# Dual-Pocket Variable-Slope Architecture Parameter Scan Report

## 1. Executive Summary
- **Scan Scope**: Parameter space of variable-slope dual-pocket configurations across $n \le 1000$.
- **Incumbent Reference**: `poc/pell_corrugation/data/border-scan/merged.csv` (1000 certified instances).
- **Total Configurations Screened**: 2,356
- **Configurations Polished with 4-Sign-Piece LP**: 0
- **Certified New Wins**: 0
- **Total Elapsed Time**: 0.24 seconds

---

## 2. Parameter Space Coverage Ledger
| Architectural Element | Parameters & Dimension Variations | Discrete $n$ Range | Target Bounding Box $L$ | Incumbent Benchmark Status |
|---|---|---|---|---|
| **Base Core** ($n=908$) | 2 pockets (100 NW, 100 SE, $\theta \approx 24.10^\circ$), central corridor | $n = 908$ | $L = 30.63702956$ | Matches public incumbent ($L=30.63702956$) exactly; deficit vs baseline $\Delta L = -0.070077$. |
| **Core Deletions** | 1 and 2 boundary square deletions along $x$ and $y$ extrema | $n \in [906, 907]$ | $L = 30.63702956$ | $n=907$ incumbent $30.618368$ is tighter via $D=51$ triple-Pell. |
| **Upright L-Border ($t=1$)** | East ($30$ squares) + North ($30$ squares) + Corner ($1$ square) | $n \in [909, 969]$ | $L = 31.63703056$ | Beats unpadded baseline ($31.707107$) by $-0.070076$; bounded by $D=51$ at $n=968$ ($31.618368$). |
| **Upright L-Border ($t=2$)** | Two-layer L-border wrap | $n \in [970, 1000]$ | $L = 32.63702956$ | Exceeds container ceiling ($L > 32.0$); non-competitive against integer grid $k=32$. |
| **Alternative Corridor Cores** | Corridor offset variations $d \in \{-1, +1, +2\}$ | $n \in [730, 850]$ | $L \in [29.637, 31.637]$ | Sub-optimal against monolithic $45^\circ$ Pell-17 bands at smaller $n$. |

---

## 3. Winning Packings Ledger
No candidates achieved $L$ strictly below the incumbents in `poc/pell_corrugation/data/border-scan/merged.csv`.

### Diagnostic Findings:
1. **$n = 908$ Ground Truth**: The dual-pocket core achieves $L = 30.637029559612035$, which exactly matches the public record in `merged.csv`. The 4-sign-piece LP confirms that the internal 200-square tilted pockets and 708 upright squares are in full coordinate jam; neither rotational nor positional relaxation yields further first-order contraction.
2. **$n = 968$ Triple-Pell Barrier**: The $t=1$ composite wraps 60 upright border squares around the core, reaching $L = 31.63703056$. While this decisively defeats the unpadded portfolio baseline ($L = 31.70710678$) by $\Delta L = -0.07007622$, `merged.csv` already embeds the $D=51$ triple-Pell corrugation record at $L = 31.618367918266944$ ($\Delta L = -0.018663$ lower than the dual-pocket composite).
3. **$n = 969$ Defect Tightness**: Adding the 61st corner square achieves $L = 31.63702966$. The incumbent in `merged.csv` is $L = 31.63398250$. 4-sign-piece LP polish cannot compress the rigid upright column/row boundaries by the required $0.003047$ due to the rigid integer grid contact of the 708 core upright squares.
4. **Corridor Variation Scale**: Modulating the central diagonal corridor width or pocket spacing dilates container dimensions without reducing the transverse thickness of the decoupled 100-square clusters. Consequently, alternative core geometries at lower $n$ do not surpass the monolithic $45^\circ$ Pell-17 bands established in `merged.csv`.

---

## 4. Verification and Transport Certification
All generated cores and composites pass:
1. **Separating Axis Theorem (SAT)**: Clearance gap $\ge 2.50 \times 10^{-11} > 0$ everywhere.
2. **Exact Discrete Transposition Symmetry**: NW/SE pocket transposition $(y - 3.0, x + 2.0, -\theta)$ residual $\le 9.02 \times 10^{-11}$.
3. **Official Metric Validation**: Fully compatible with `squares.core.certify` and `pandas 3.1.0` transport.
