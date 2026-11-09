# Top 40 Excess Targets in $n \in [100, 1000]$

**Source Artifacts:** `poc/pell_corrugation/data/border-scan/merged-literal-cpp.json`, `merged.csv`  
**Metric:** Normalized side-length ratio $r(n) = \frac{L(n)}{\sqrt{n}}$  
**Local Excess Gap:** $\text{gap}(n) = r(n) - \text{median}\big(r(m) : m \in [n-5, n+5], m \ne n, m \ne \text{square}\big)$  
**Structured Data:** `poc/research/data/top40_excess_targets.json`  

---

## 1. Executive Summary

Across all non-square puzzle instances $n \in [100, 1000]$, the portfolio score contribution $\frac{L(n)}{\sqrt{n}} - 1$ generally decreases as $O(1/\sqrt{n})$. However, localized packaging bottlenecks create substantial excess spikes above the local background baseline. The metric $\text{gap}(n) = r(n) - \text{median}_{m \in [n-5, n+5]}(r(m))$ isolates these localized topological inefficiencies from the global $1/\sqrt{n}$ decay curve.

Extracting and ranking the top 40 target instances reveals that excess gaps partition cleanly into three distinct structural regimes:

1. **$k^2+1$ Boundary Steps (10 targets, 25.0% of cohort):**
   At $n = k^2 + 1$ ($j = 1$), inserting a single square destroys the compact $k \times k$ Cartesian grid, forcing container expansion to accommodate a diagonal strip or corner pocket. In the baseline dataset, this jump is consistently $\Delta L \approx 0.5355$ ($L \approx k + 0.5355$ or $k + 0.5208$), elevating $r(n)$ sharply above the preceding $k \times k$ integer floor.
   - Targets: $n \in \{290, 325, 362, 401, 442, 485, 530, 577, 626, 677\}$ spanning $k=17$ to $k=26$.

2. **Middle-Band $j \in \{3..10\}$ Dilations (27 targets, 67.5% of cohort):**
   In the intermediate deficit regime where $j \in [3, 10]$, the packing requires tilted bands or multi-angle clusters. Monolithic $45^\circ$ bands incur the classical transverse tooth inflation penalty $(\sqrt{2}-1 \approx 0.4142)$, while legacy heuristic packings leave unfilled sub-unit cavities.
   - Targets include the overall #1 worst excess gap $n = 105$ ($\text{gap} = 0.003611$), $n = 103$ ($\text{gap} = 0.002939$), $n = 127$ ($\text{gap} = 0.002918$), $n = 147$ ($\text{gap} = 0.002913$), and $n = 151$ ($\text{gap} = 0.002653$).

3. **Historical Baseline Records (3 targets, 7.5% of cohort):**
   Legacy heuristic simulated annealing records (Friedman, Gensane & Ryckelynck, Specht) that feature non-standard $j$ offsets or unrelaxed split-angle bands:
   - $n = 102$ ($k=10, j=2$): Symmetric split-angle band ($\\pm 43.1^\circ$) with unpolished boundary slack.
   - $n = 132$ ($k=11, j=11$): Dense multi-angle cluster containing 11 tilted squares ($L = 11.985680$).
   - $n = 183$ ($k=13, j=14$): Near-ceiling record sitting immediately beneath integer boundary $L=14.0$ ($L = 13.999580$).

---

## 2. Top 40 Excess Targets Ranked Ledger

| Rank | $n$ | $k$ | $j$ | $L(n)$ | $r(n) = \frac{L}{\sqrt{n}}$ | $\text{median}(r_{\text{neigh}})$ | $\text{gap}(n)$ | Incumbent Construction | Category | Structural Diagnosis |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 105 | 10 | 5 | `10.789304` | `1.052928` | `1.049317` | `0.003611` | `multi_angle_cluster` | Middle-Band | Highest excess gap in 100..1000; unrelaxed 42-square multi-angle cluster in $10^2+5$ |
| 2 | 103 | 10 | 3 | `10.679232` | `1.052256` | `1.049317` | `0.002939` | `multi_angle_cluster` | Middle-Band | High-dilation 38-square multi-angle cluster; severe transverse cavity inflation |
| 3 | 127 | 11 | 6 | `11.810879` | `1.048046` | `1.045128` | `0.002918` | `multi_angle_cluster` | Middle-Band | Heuristic 48-square cluster with 9 distinct tilt angles; unrelaxed LP slack |
| 4 | 147 | 12 | 3 | `12.656854` | `1.043920` | `1.041007` | `0.002913` | `grid_tilted_bands` | Middle-Band | 64-square monolithic $45^\circ$ tilted band; rigid tooth inflation over rational tilt |
| 5 | 151 | 12 | 7 | `12.822876` | `1.043511` | `1.040858` | `0.002653` | `variable_slope_band` | Middle-Band | Dual-angle band ($\theta \approx 24.3^\circ, 45^\circ$); unpolished boundary gap |
| 6 | 150 | 12 | 6 | `12.778175` | `1.043334` | `1.041013` | `0.002320` | `grid_tilted_bands` | Middle-Band | 66-square monolithic $45^\circ$ tilted band with upright flank grid |
| 7 | 175 | 13 | 6 | `13.764174` | `1.040474` | `1.038220` | `0.002254` | `multi_angle_cluster` | Middle-Band | 138-square simulated annealing cluster with 27 unique tilt angles |
| 8 | 110 | 10 | 10 | `10.996783` | `1.048502` | `1.046330` | `0.002172` | `multi_angle_cluster` | Middle-Band | Near-ceiling packing ($L = 10.996783 \approx 11.0$); dense unrelaxed multi-angle cluster |
| 9 | 106 | 10 | 6 | `10.822908` | `1.051214` | `1.049056` | `0.002158` | `variable_slope_band` | Middle-Band | Variable-slope band with 22 tilted squares at 2 non-$45^\circ$ angles |
| 10 | 125 | 11 | 4 | `11.707107` | `1.047115` | `1.045159` | `0.001956` | `grid_tilted_bands` | Middle-Band | Single-strip $45^\circ$ diagonal insertion ($w=1, 13$ tilted squares) |
| 11 | 176 | 13 | 7 | `13.795872` | `1.039903` | `1.038081` | `0.001822` | `multi_angle_cluster` | Middle-Band | 65-square multi-angle cluster with 10 distinct non-$45^\circ$ orientations |
| 12 | 290 | 17 | 1 | `17.535534` | `1.029722` | `1.027951` | `0.001771` | `grid_tilted_bands` | $k^2+1$ Boundary | Square boundary $17^2+1$; 50-square $45^\circ$ band jumps $L$ to $17.535534$ |
| 13 | 124 | 11 | 3 | `11.656854` | `1.046816` | `1.045159` | `0.001657` | `grid_tilted_bands` | Middle-Band | 64-square monolithic $45^\circ$ tilted band with upright flank staircases |
| 14 | 325 | 18 | 1 | `18.535534` | `1.028166` | `1.026588` | `0.001578` | `grid_tilted_bands` | $k^2+1$ Boundary | Square boundary $18^2+1$; 25-square $45^\circ$ band jumps $L$ to $18.535534$ |
| 15 | 102 | 10 | 2 | `10.605829` | `1.050134` | `1.048587` | `0.001547` | `split_angle_band` | Historical Baseline | Symmetric split-angle band ($\pm 43.1^\circ$, 31 tilted squares); legacy heuristic |
| 16 | 362 | 19 | 1 | `19.535534` | `1.026765` | `1.025350` | `0.001415` | `grid_tilted_bands` | $k^2+1$ Boundary | Square boundary $19^2+1$; 25-square $45^\circ$ band jumps $L$ to $19.535534$ |
| 17 | 367 | 19 | 6 | `19.692388` | `1.027934` | `1.026560` | `0.001374` | `multi_angle_cluster` | Middle-Band | 238-square multi-angle cluster; large intermediate $19^2+6$ deficit |
| 18 | 232 | 15 | 7 | `15.761656` | `1.034803` | `1.033446` | `0.001358` | `multi_angle_cluster` | Middle-Band | 142-square multi-angle cluster with 24 distinct tilt orientations |
| 19 | 203 | 14 | 7 | `14.778175` | `1.037225` | `1.035879` | `0.001345` | `grid_tilted_bands` | Middle-Band | 111-square $45^\circ$ tilted band flanked by upright Cartesian blocks |
| 20 | 295 | 17 | 6 | `17.704233` | `1.030780` | `1.029465` | `0.001315` | `grid_tilted_bands` | Middle-Band | 160-square $45^\circ$ diagonal band cutting across $17 \times 17$ container |
| 21 | 401 | 20 | 1 | `20.535534` | `1.025496` | `1.024219` | `0.001276` | `multi_angle_cluster` | $k^2+1$ Boundary | Square boundary $20^2+1$; legacy 234-square cluster with $45^\circ$ central core ($L = 20.535534$) |
| 22 | 491 | 22 | 7 | `22.692388` | `1.024093` | `1.022856` | `0.001237` | `multi_angle_cluster` | Middle-Band | 393-square dense multi-angle cluster; severe unrelaxed boundary teeth |
| 23 | 264 | 16 | 8 | `16.774735` | `1.032414` | `1.031212` | `0.001202` | `multi_angle_cluster` | Middle-Band | 77-square multi-angle cluster with 6 distinct tilt angles in $16^2+8$ |
| 24 | 442 | 21 | 1 | `21.535534` | `1.024341` | `1.023184` | `0.001157` | `multi_angle_cluster` | $k^2+1$ Boundary | Square boundary $21^2+1$; legacy 332-square annealing cluster with 124 orientations ($L = 21.535534$) |
| 25 | 234 | 15 | 9 | `15.822876` | `1.034374` | `1.033218` | `0.001155` | `variable_slope_band` | Middle-Band | 36-square rational slope band ($\theta \approx 24.3^\circ \approx \arctan(1/2)$) |
| 26 | 201 | 14 | 5 | `14.707107` | `1.037359` | `1.036212` | `0.001147` | `grid_tilted_bands` | Middle-Band | 17-square narrow $45^\circ$ band insertion in $14^2+5$ |
| 27 | 485 | 22 | 1 | `22.535534` | `1.023286` | `1.022233` | `0.001053` | `multi_angle_cluster` | $k^2+1$ Boundary | Square boundary $22^2+1$; legacy 248-square cluster with $45^\circ$ core ($L = 22.535534$) |
| 28 | 626 | 25 | 1 | `25.520815` | `1.020017` | `1.018979` | `0.001038` | `grid_tilted_bands` | $k^2+1$ Boundary | Square boundary $25^2+1$; Pell-17 corrugated core $D=17$ ($L = 25.520815$, slack 0.0208) |
| 29 | 183 | 13 | 14 | `13.999580` | `1.034879` | `1.033843` | `0.001035` | `multi_angle_cluster` | Historical Baseline | Near-integer boundary packing ($L = 13.999580 \approx 14.0$); dense heuristic cluster |
| 30 | 132 | 11 | 11 | `11.985680` | `1.043220` | `1.042222` | `0.000997` | `multi_angle_cluster` | Historical Baseline | Dense 39-square cluster accommodating 11 extra squares below $L=12.0$ |
| 31 | 530 | 23 | 1 | `23.535534` | `1.022318` | `1.021355` | `0.000963` | `multi_angle_cluster` | $k^2+1$ Boundary | Square boundary $23^2+1$; legacy 352-square cluster with $45^\circ$ core ($L = 23.535534$) |
| 32 | 126 | 11 | 5 | `11.742641` | `1.046118` | `1.045159` | `0.000958` | `grid_tilted_bands` | Middle-Band | 37-square $45^\circ$ diagonal band in $11^2+5$ |
| 33 | 204 | 14 | 8 | `14.807627` | `1.036741` | `1.035798` | `0.000943` | `multi_angle_cluster` | Middle-Band | 133-square multi-angle cluster with 26 tilt orientations in $14^2+8$ |
| 34 | 677 | 26 | 1 | `26.520815` | `1.019278` | `1.018340` | `0.000937` | `grid_tilted_bands` | $k^2+1$ Boundary | Square boundary $26^2+1$; Pell-17 corrugated core $D=17$ ($L = 26.520815$) |
| 35 | 177 | 13 | 8 | `13.822980` | `1.038999` | `1.038081` | `0.000917` | `variable_slope_band` | Middle-Band | 30-square variable-slope band with 3 distinct tilt orientations |
| 36 | 330 | 18 | 6 | `18.692388` | `1.028982` | `1.028071` | `0.000911` | `multi_angle_cluster` | Middle-Band | 234-square multi-angle cluster; intermediate deficit at $18^2+6$ |
| 37 | 261 | 16 | 5 | `16.678780` | `1.032391` | `1.031486` | `0.000904` | `multi_angle_cluster` | Middle-Band | 223-square multi-angle cluster with 65 distinct orientations in $16^2+5$ |
| 38 | 333 | 18 | 9 | `18.773753` | `1.028795` | `1.027891` | `0.000904` | `multi_angle_cluster` | Middle-Band | 182-square multi-angle cluster with 59 distinct orientations in $18^2+9$ |
| 39 | 366 | 19 | 5 | `19.656854` | `1.027480` | `1.026584` | `0.000896` | `grid_tilted_bands` | Middle-Band | 120-square $45^\circ$ diagonal band in $19^2+5$ |
| 40 | 577 | 24 | 1 | `24.535534` | `1.021428` | `1.020544` | `0.000884` | `multi_angle_cluster` | $k^2+1$ Boundary | Square boundary $24^2+1$; legacy 411-square cluster with 161 orientations ($L = 24.535534$) |

---

## 3. Structural Categorization & Deep Dive

### 3.1 $k^2+1$ Boundary Steps (10 Instances)

When $n = k^2 + 1$, the compact integer square grid $k \times k$ is saturated. Accommodating the single extra square destroys lattice collinearity:
- **Monolithic Diagonal Dilation:** The incumbent solutions for $k \in [17, 24]$ exhibit a characteristic jump to $L = k + 0.535534$ (e.g., $n = 290, 325, 362, 401, 442, 485, 530, 577$). This step size corresponds to inserting a diagonal $45^\circ$ strip with width $w=1$ across the container.
- **Corrugated Core Transitions:** At $k = 25$ ($n = 626$) and $k = 26$ ($n = 677$), the step size drops to $L = k + 0.520815$, reflecting the entry of the Pell-17 corrugated core family ($D=17, \text{slack} = 0.020815$).
- **Optimization Vector:** Replacing monolithic $45^\circ$ strips with rational-angle tilted bands (e.g. $\arctan(1/2) \approx 26.565^\circ$) or out-of-phase pinch-column corrugation can contract the transverse boundary inflation by up to $17.5\%$, yielding potential score gains on the order of $10^{-4}$ to $10^{-3}$.

### 3.2 Middle-Band $j \in \{3..10\}$ Dilations (27 Instances)

The intermediate deficit regime $j \in \{3..10\}$ represents the bulk of the top-40 excess targets (67.5%):
- **Severe Top-End Deficits:** The single highest excess gap in the entire 100..1000 dataset is $n = 105$ ($k=10, j=5$) with $\text{gap} = 0.003611$, followed immediately by $n = 103$ ($j=3, \text{gap} = 0.002939$), $n = 127$ ($j=6, \text{gap} = 0.002918$), $n = 147$ ($j=3, \text{gap} = 0.002913$), and $n = 151$ ($j=7, \text{gap} = 0.002653$).
- **Tooth Penalty vs Cavity Slack:** In these configurations, $j$ squares cannot fit into a narrow $w=1$ band, forcing either wider bands ($w=2, 3$) or messy multi-angle clusters. Unpolished multi-angle clusters (e.g., $n = 105, 127, 175, 232, 261, 330, 333$) feature extensive internal void space and non-active separating axes.
- **Optimization Vector:** 4-sign-piece epigraph LP relaxation directly eliminates loose non-contact clearances. Transplanting structured Pell cores ($w=3, 6, 9$) with calibrated L-border wrappings provides systematic upper-bound reductions.

### 3.3 Historical Baseline Records (3 Instances)

Three targets fall outside the $j=1$ boundary and middle-band $j \in \{3..10\}$ ranges:
- **$n = 102$ ($k=10, j=2$, Rank 15, $\text{gap} = 0.001547$):** Packed via a symmetric split-angle band ($\pm 43.1^\circ$) with 31 tilted squares. While significantly tighter than a single $45^\circ$ band, the angle split was generated via unconstrained simulated annealing without high-precision active-set polishing.
- **$n = 132$ ($k=11, j=11$, Rank 30, $\text{gap} = 0.000997$):** Features 39 tilted squares organized into 11 distinct tilt angles. At $j=k=11$, this configuration sits halfway to $(k+1)^2$ and suffers from boundary interference with both container walls.
- **$n = 183$ ($k=13, j=14$, Rank 29, $\text{gap} = 0.001035$):** Incumbent $L = 13.999580$ is just $4.2 \times 10^{-4}$ below the integer grid ceiling $L=14.0$. It contains 73 tilted squares with 17 unique angles jammed against the boundary.

---

## 4. Methodological Note: Window Clamping Sensitivity

When evaluating $\text{median}(r(m) : m \in [n-5, n+5])$, computing neighbors over the complete $[1, 1000]$ sequence from `merged-literal-cpp.json` (unclamped) evaluates the true local background across the $n=100$ square boundary for $n \in \{101..104\}$. If the neighbor window is artificially clamped strictly to $[100, 1000]$ (one-sided forward window for $n=101..104$):
- Exactly 39 of the 40 targets remain identical in composition.
- $n = 102$ drops out of the top 40 due to one-sided forward-median bias, and $n = 492$ ($k=22, j=8, L=22.707107, \text{gap} = 0.000859$) enters at rank 40 as a middle-band dilation.
- All 10 $k^2+1$ boundary steps and 27 of the 28 middle-band dilations are strictly invariant to boundary clamping.
