# High-$n$ Excess Gap Ranking & Structural Audit ($n \in [600, 1000]$)

**Source Incumbents:** `poc/pell_corrugation/data/border-scan/merged.csv`  
**Public Benchmark:** `poc/pell_corrugation/data/border-scan/baseline-official.json`  
**Evaluated Cohort:** 394 non-square instances ($n \in [600, 1000]$, excluding perfect squares 625, 676, 729, 784, 841, 900, 961)  
**Primary Metric:** Incumbent ratio $r(n) = \frac{L(n)}{\sqrt{n}} - 1.0$  
**Local Excess Gap:** $\text{gap}(n) = r(n) - \text{median}_{m \in [n-5, n+5], m \ne n, m \ne \text{square}}(r(m))$  
**Structured Dataset:** `poc/research/data/high_n_gap_ranking.json`  

---

## 1. Executive Summary

Across the upper competition tier $n \in [600, 1000]$, normalized side-length overhead $r(n) = L(n)/\sqrt{n} - 1.0$ drops from $\sim 0.020$ down to $\sim 0.012$. However, discrete geometric packaging barriers generate pronounced localized excess spikes relative to the background trend. By calculating localized baseline medians over symmetric neighbor windows ($m \in [n-5, n+5]$, excluding perfect squares), this audit removes the global $O(1/\sqrt{n})$ curvature and exposes exact structural inefficiencies.

The 394 non-square instances segment into three primary behavioral regimes:

1. **$k^2+1$ Boundary Steps (7 Targets, 100% in Top 12):**
   Every single $k^2+1$ instance in $600..1000$ ($n \in \{626, 677, 730, 785, 842, 901, 962\}$) occupies the upper echelon of the ranking (Ranks 1, 2, 3, 5, 7, 8, and 12). In every case, crossing from the $k \times k$ Cartesian integer grid into $k^2+1$ incurs an identical left-delta step of $\Delta L = 0.52081528$, jumping to $L = k + 0.520815$ (the Pell-17 $D=17$ diagonal strip). Because this step size is invariant while $\sqrt{n}$ grows, the excess gap decays monotonically from $0.001038$ at $n=626$ down to $0.000581$ at $n=962$.

2. **Corrugation Deficit Windows & Core Truncation Plateaus:**
   The four Pell-17 corrugated core improvements in `merged.csv` ($n = 739, 794, 851, 910$) achieve certified wins over the public baseline ($\Delta L = -3.89 \times 10^{-5}$ each), but their immediate neighbor flanks exhibit severe unrelaxed truncation bottlenecks. Most notably, **$n = 738$** ($27^2+9$, Rank 6, $\text{gap} = 0.000745$) sits on an unrelaxed container plateau ($L = 27.690919$) virtually identical to $n = 739$ ($L = 27.690921$, right delta $= +0.000002$). Similarly, **$n = 851$** (Rank 13, $\text{gap} = 0.000547$) and **$n = 910$** (Rank 9, $\text{gap} = 0.000627$) anchor high-excess clusters with right-neighbor plateaus at $n = 852$ (Rank 15, $\text{gap} = 0.000504$) and $n = 911$ (Rank 10, $\text{gap} = 0.000605$).

3. **Middle-Band $j \in [7, 13]$ Dilations:**
   Intermediate square deficits $j \in [7, 13]$ require multi-layer tilted bands or dense multi-angle clusters. The highest non-boundary excess target is **$n = 634$** ($25^2+9$, Rank 4, $\text{gap} = 0.000826$), which incurs a large container inflation ($L = 25.707107 \approx 25 + 1/\sqrt{2}$) with zero right delta into $n = 635$. Other major middle-band dilations include $n = 685$ ($26^2+9$, Rank 11), $n = 742$ ($27^2+13$, Rank 14), $n = 632$ ($25^2+7$, Rank 16), and $n = 687$ ($26^2+11$, Rank 17).

---

## 2. Top 50 High-$n$ Excess Targets Ledger

| Rank | $n$ | $k$ | $j$ | $L(n)$ | $r(n)$ | Local Baseline | Excess Gap | $\Delta L_{\text{left}}$ | $\Delta L_{\text{right}}$ | Benchmark Gap (vs Public) | Construction | Category |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **626** | 25 | 1 | `25.520815` | `0.020017` | `0.018979` | **`+0.001038`** | `+0.520815` | `+0.014719` | `0.0` (Tied) | `grid_tilted_bands` | k^2+1 Boundary Step |
| 2 | **677** | 26 | 1 | `26.520815` | `0.019278` | `0.018340` | **`+0.000937`** | `+0.520815` | `+0.014719` | `0.0` (Tied) | `grid_tilted_bands` | k^2+1 Boundary Step |
| 3 | **730** | 27 | 1 | `27.520815` | `0.018591` | `0.017743` | **`+0.000848`** | `+0.520815` | `+0.014719` | `0.0` (Tied) | `grid_tilted_bands` | k^2+1 Boundary Step |
| 4 | **634** | 25 | 9 | `25.707107` | `0.020960` | `0.020134` | **`+0.000826`** | `+0.050253` | `-0.000000` | `0.0` (Tied) | `grid_tilted_bands` | Middle-Band Dilation |
| 5 | **785** | 28 | 1 | `28.520815` | `0.017952` | `0.017182` | **`+0.000770`** | `+0.520815` | `+0.014719` | `0.0` (Tied) | `multi_angle_cluster` | k^2+1 Boundary Step |
| 6 | **738** | 27 | 9 | `27.690919` | `0.019317` | `0.018572` | **`+0.000745`** | `+0.052027` | `+0.000002` | `0.0` (Tied) | `multi_angle_cluster` | Corrugation Deficit Window |
| 7 | **842** | 29 | 1 | `29.520815` | `0.017354` | `0.016655` | **`+0.000699`** | `+0.520815` | `+0.014719` | `0.0` (Tied) | `multi_angle_cluster` | k^2+1 Boundary Step |
| 8 | **901** | 30 | 1 | `30.520815` | `0.016796` | `0.016159` | **`+0.000637`** | `+0.520815` | `+0.014719` | `0.0` (Tied) | `multi_angle_cluster` | k^2+1 Boundary Step |
| 9 | **910** | 30 | 10 | `30.690921` | `0.017394` | `0.016767` | **`+0.000627`** | `+0.034067` | `+0.016186` | `-3.90e-05` | `pell17_corrugated_band` | Corrugation Deficit Window |
| 10 | **911** | 30 | 11 | `30.707107` | `0.017372` | `0.016767` | **`+0.000605`** | `+0.016186` | `+0.000000` | `0.0` (Tied) | `grid_tilted_bands` | Corrugation Deficit Window |
| 11 | **685** | 26 | 9 | `26.692388` | `0.019864` | `0.019280` | **`+0.000584`** | `+0.035534` | `+0.014719` | `0.0` (Tied) | `multi_angle_cluster` | Middle-Band Dilation |
| 12 | **962** | 31 | 1 | `31.520815` | `0.016272` | `0.015691` | **`+0.000581`** | `+0.520815` | `+0.014719` | `0.0` (Tied) | `multi_angle_cluster` | k^2+1 Boundary Step |
| 13 | **851** | 29 | 10 | `29.690922` | `0.017792` | `0.017245` | **`+0.000547`** | `+0.034068` | `+0.016185` | `-3.81e-05` | `pell17_corrugated_band` | Corrugation Deficit Window |
| 14 | **742** | 27 | 13 | `27.749514` | `0.018717` | `0.018207` | **`+0.000509`** | `+0.021592` | `+0.007969` | `0.0` (Tied) | `multi_angle_cluster` | Middle-Band Dilation |
| 15 | **852** | 29 | 11 | `29.707107` | `0.017749` | `0.017245` | **`+0.000504`** | `+0.016185` | `+0.000000` | `0.0` (Tied) | `grid_tilted_bands` | Corrugation Deficit Window |
| 16 | **632** | 25 | 7 | `25.656854` | `0.020575` | `0.020102` | **`+0.000473`** | `+0.029437` | `+0.000000` | `0.0` (Tied) | `multi_angle_cluster` | Middle-Band Dilation |
| 17 | **687** | 26 | 11 | `26.727922` | `0.019734` | `0.019280` | **`+0.000454`** | `+0.020815` | `+0.000000` | `0.0` (Tied) | `grid_tilted_bands` | Middle-Band Dilation |
| 18 | **747** | 27 | 18 | `27.824481` | `0.018045` | `0.017623` | **`+0.000421`** | `+0.022052` | `+0.003566` | `0.0` (Tied) | `multi_angle_cluster` | General Tilted Interior |
| 19 | **797** | 28 | 13 | `28.741908` | `0.018091` | `0.017673` | **`+0.000418`** | `+0.013986` | `+0.009049` | `0.0` (Tied) | `multi_angle_cluster` | Middle-Band Dilation |
| 20 | **638** | 25 | 13 | `25.771959` | `0.020322` | `0.019908` | **`+0.000413`** | `+0.026839` | `+0.009177` | `0.0` (Tied) | `multi_angle_cluster` | Middle-Band Dilation |
| 21 | **971** | 31 | 10 | `31.678780` | `0.016620` | `0.016211` | **`+0.000410`** | `+0.021926` | `+0.013608` | `0.0` (Tied) | `multi_angle_cluster` | Corrugation Deficit Window |
| 22 | **686** | 26 | 10 | `26.707107` | `0.019682` | `0.019280` | **`+0.000402`** | `+0.014719` | `+0.020815` | `0.0` (Tied) | `grid_tilted_bands` | Middle-Band Dilation |
| 23 | **848** | 29 | 7 | `29.633983` | `0.017635` | `0.017240` | **`+0.000395`** | `+0.027381` | `+0.003047` | `0.0` (Tied) | `multi_angle_cluster` | Middle-Band Dilation |
| 24 | **814** | 28 | 30 | `28.961261` | `0.015092` | `0.014699` | **`+0.000393`** | `+0.028443` | `+0.006033` | `0.0` (Tied) | `multi_angle_cluster` | General Tilted Interior |
| 25 | **902** | 30 | 2 | `30.535534` | `0.016722` | `0.016330` | **`+0.000392`** | `+0.014719` | `+0.000000` | `0.0` (Tied) | `multi_angle_cluster` | Middle-Band Dilation |
| 26 | **793** | 28 | 9 | `28.678780` | `0.018413` | `0.018045` | **`+0.000368`** | `+0.041750` | `+0.012141` | `0.0` (Tied) | `multi_angle_cluster` | Corrugation Deficit Window |
| 27 | **690** | 26 | 14 | `26.776673` | `0.019371` | `0.019004` | **`+0.000367`** | `+0.025717` | `+0.010039` | `0.0` (Tied) | `multi_angle_cluster` | Middle-Band Dilation |
| 28 | **855** | 29 | 14 | `29.748737` | `0.017385` | `0.017025` | **`+0.000361`** | `+0.020815` | `+0.002219` | `0.0` (Tied) | `multi_angle_cluster` | Middle-Band Dilation |
| 29 | **972** | 31 | 11 | `31.692388` | `0.016534` | `0.016208` | **`+0.000326`** | `+0.013608` | `+0.014719` | `0.0` (Tied) | `multi_angle_cluster` | Corrugation Deficit Window |
| 30 | **994** | 31 | 33 | `31.957113` | `0.013618` | `0.013303` | **`+0.000315`** | `+0.024294` | `+0.003747` | `0.0` (Tied) | `multi_angle_cluster` | General Tilted Interior |
| 31 | **932** | 30 | 32 | `30.960210` | `0.014135` | `0.013825` | **`+0.000311`** | `+0.026064` | `+0.008475` | `0.0` (Tied) | `multi_angle_cluster` | General Tilted Interior |
| 32 | **963** | 31 | 2 | `31.535534` | `0.016218` | `0.015910` | **`+0.000308`** | `+0.014719` | `+0.000000` | `0.0` (Tied) | `multi_angle_cluster` | Middle-Band Dilation |
| 33 | **976** | 31 | 15 | `31.746671` | `0.016186` | `0.015882` | **`+0.000304`** | `+0.018749` | `+0.004286` | `0.0` (Tied) | `multi_angle_cluster` | General Tilted Interior |
| 34 | **796** | 28 | 12 | `28.727922` | `0.018234` | `0.017932` | **`+0.000302`** | `+0.020815` | `+0.013986` | `0.0` (Tied) | `grid_tilted_bands` | Middle-Band Dilation |
| 35 | **791** | 28 | 7 | `28.637030` | `0.018216` | `0.017940` | **`+0.000276`** | `+0.024173` | `+0.000000` | `0.0` (Tied) | `multi_angle_cluster` | Middle-Band Dilation |
| 36 | **973** | 31 | 12 | `31.707107` | `0.016483` | `0.016208` | **`+0.000275`** | `+0.014719` | `+0.000000` | `0.0` (Tied) | `grid_tilted_bands` | Middle-Band Dilation |
| 37 | **980** | 31 | 19 | `31.795730` | `0.015677` | `0.015403` | **`+0.000275`** | `+0.016300` | `+0.001711` | `0.0` (Tied) | `multi_angle_cluster` | General Tilted Interior |
| 38 | **646** | 25 | 21 | `25.889392` | `0.018605` | `0.018339` | **`+0.000265`** | `+0.019074` | `+0.005559` | `0.0` (Tied) | `multi_angle_cluster` | General Tilted Interior |
| 39 | **652** | 25 | 27 | `25.973717` | `0.017209` | `0.016956` | **`+0.000253`** | `+0.015238` | `+0.006921` | `0.0` (Tied) | `multi_angle_cluster` | General Tilted Interior |
| 40 | **868** | 29 | 27 | `29.907222` | `0.015117` | `0.014872` | **`+0.000246`** | `+0.019604` | `+0.005106` | `0.0` (Tied) | `multi_angle_cluster` | General Tilted Interior |
| 41 | **705** | 26 | 29 | `26.983940` | `0.016274` | `0.016032` | **`+0.000242`** | `+0.020639` | `+0.007774` | `0.0` (Tied) | `multi_angle_cluster` | General Tilted Interior |
| 42 | **928** | 30 | 28 | `30.907790` | `0.014598` | `0.014364` | **`+0.000234`** | `+0.018827` | `+0.004539` | `0.0` (Tied) | `multi_angle_cluster` | General Tilted Interior |
| 43 | **803** | 28 | 19 | `28.826459` | `0.017264` | `0.017030` | **`+0.000233`** | `+0.019399` | `+0.003257` | `0.0` (Tied) | `multi_angle_cluster` | General Tilted Interior |
| 44 | **970** | 31 | 9 | `31.656854` | `0.016440` | `0.016211` | **`+0.000230`** | `+0.022872` | `+0.021926` | `0.0` (Tied) | `multi_angle_cluster` | Middle-Band Dilation |
| 45 | **736** | 27 | 7 | `27.637030` | `0.018714` | `0.018486` | **`+0.000228`** | `+0.019457` | `+0.001862` | `0.0` (Tied) | `multi_angle_cluster` | Middle-Band Dilation |
| 46 | **874** | 29 | 33 | `29.987156` | `0.014331` | `0.014103` | **`+0.000227`** | `+0.019371` | `+0.006540` | `0.0` (Tied) | `multi_angle_cluster` | General Tilted Interior |
| 47 | **921** | 30 | 21 | `30.828016` | `0.015818` | `0.015591` | **`+0.000227`** | `+0.022596` | `+0.008837` | `0.0` (Tied) | `multi_angle_cluster` | General Tilted Interior |
| 48 | **695** | 26 | 19 | `26.848463` | `0.018420` | `0.018199` | **`+0.000222`** | `+0.020416` | `+0.008709` | `0.0` (Tied) | `multi_angle_cluster` | General Tilted Interior |
| 49 | **984** | 31 | 23 | `31.838995` | `0.014990` | `0.014772` | **`+0.000218`** | `+0.022138` | `+0.008439` | `0.0` (Tied) | `multi_angle_cluster` | General Tilted Interior |
| 50 | **760** | 27 | 31 | `27.990627` | `0.015327` | `0.015113` | **`+0.000214`** | `+0.019020` | `+0.007193` | `0.0` (Tied) | `multi_angle_cluster` | General Tilted Interior |

---

## 3. Structural Regime Deep Dive & Anomalies

### 3.1 The $k^2+1$ Boundary Step Anomaly

Across all $k \in [25, 31]$, the step from the perfect square grid $n = k^2$ ($L = k$) to $n = k^2+1$ exhibits a rigid, invariant delta of exactly:
$$\Delta L_{\text{step}} = L(k^2+1) - L(k^2) = 0.52081528$$
This value corresponds to the uncorrugated $D=17, w=1$ diagonal boundary strip. Inserting even a single extra square forces container dilation across the entire diagonal.

| $k$ | $k^2$ | $n = k^2+1$ | $L(k^2+1)$ | $\Delta L_{\text{left}}$ | Local Baseline $r$ | Excess Gap | Rank |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |

| 25 | 625 | **626** | `25.520815` | `+0.520815` | `0.018979` | **`+0.001038`** | #1 |
| 26 | 676 | **677** | `26.520815` | `+0.520815` | `0.018340` | **`+0.000937`** | #2 |
| 27 | 729 | **730** | `27.520815` | `+0.520815` | `0.017743` | **`+0.000848`** | #3 |
| 28 | 784 | **785** | `28.520815` | `+0.520815` | `0.017182` | **`+0.000770`** | #5 |
| 29 | 841 | **842** | `29.520815` | `+0.520815` | `0.016655` | **`+0.000699`** | #7 |
| 30 | 900 | **901** | `30.520815` | `+0.520815` | `0.016159` | **`+0.000637`** | #8 |
| 31 | 961 | **962** | `31.520815` | `+0.520815` | `0.015691` | **`+0.000581`** | #12 |

**Key Takeaway for Solvers:** The $k^2+1$ targets represent the most predictable, high-value candidates in $600..1000$. Replacing the rigid $45^\circ$ single strip with a rational-angle tilted pocket (such as $\arctan(1/2) \approx 26.565^\circ$) or out-of-phase pinch corrugation could contract this step from $0.5208$ down toward $\sim 0.44$, yielding gains of $\Delta L \approx 0.08$ on each boundary step.

### 3.2 Corrugation Deficit Windows Around Pell-17 Cores

The border-scan optimization incorporated four certified Pell-17 corrugated wins into `merged.csv`:
- $n = 739$ ($k=27, j=10, w=13$): $L = 27.69092118$, $\Delta L_{\text{public}} = -3.894 \times 10^{-5}$
- $n = 794$ ($k=28, j=10, w=13 + 1\text{b}$): $L = 28.69092119$, $\Delta L_{\text{public}} = -3.893 \times 10^{-5}$
- $n = 851$ ($k=29, j=10, w=13 + 2\text{b}$): $L = 29.69092200$, $\Delta L_{\text{public}} = -3.812 \times 10^{-5}$
- $n = 910$ ($k=30, j=10, w=13 + 3\text{b}$): $L = 30.69092107$, $\Delta L_{\text{public}} = -3.905 \times 10^{-5}$

However, analyzing neighbor deltas reveals severe asymmetric boundary step anomalies:
1. **The $n=738$ Deficit Trap (Rank 6):** $L(738) = 27.69091941$ and $L(739) = 27.69092118$. Removing one square from $n=739$ yields an infinitesimal container shrinkage of only $\Delta L = 1.77 \times 10^{-6}$. Because $n=738$ was constructed by crude extreme-prefix coordinate truncation without structural vacancy defect relaxation, it retains the container dimensions of $n=739$, causing a huge excess gap of $+0.000745$.
2. **Post-Core Plateaus at $n=852$ and $n=911$:** Adding one square past the corrugated core at $n=852$ and $n=911$ forces a jump to $L \approx k + 1/\sqrt{2} \approx k + 0.707107$, creating sharp plateaus where $\Delta L_{\text{right}} = 0.0$ ($L(852) = L(853) = 29.707107$ and $L(911) = L(912) = 30.707107$).
3. **Public Leader Deficits at $n=971, 972$:** $n=971$ (Rank 21, $\text{gap} = 0.000410$) and $n=972$ (Rank 29, $\text{gap} = 0.000326$) reflect the boundary between 13-strip diagonal cores and four-block dual-pocket architectures, where single-band corrugation stalls.

### 3.3 Middle-Band $j \in [7, 13]$ Dilations & Plateaus

Outside the $k^2+1$ steps and Pell windows, the largest excess gaps are concentrated in intermediate offsets $j \in [7, 13]$:
- **$n = 634$ ($k=25, j=9$, Rank 4, $\text{gap} = 0.000826$):** $L = 25.707107$. Preceded by $L(633) = 25.656854$ ($\Delta L_{\text{left}} = 0.050253$) and completely flat into $n=635$ ($\Delta L_{\text{right}} = -0.000000$). This $j=9$ bottleneck corresponds to the onset of a 2-layer upright flank constraint against a $45^\circ$ core.
- **$n = 685$ ($k=26, j=9$, Rank 11, $\text{gap} = 0.000584$):** $L = 26.692388$. Large jump $\Delta L_{\text{left}} = 0.035534$.
- **$n = 742$ ($k=27, j=13$, Rank 14, $\text{gap} = 0.000509$):** $L = 27.749514$. Multi-angle heuristic packing suffering from unpolished interior void spaces.
- **$n = 632$ ($k=25, j=7$, Rank 16, $\text{gap} = 0.000473$):** $L = 25.656854$. Container ceiling plateau shared with $n=633$.

---

## 4. Priority Attack Roster for Research Solvers

Based on localized excess gap, container rigidity, and structural solvability, the top target priorities in $600..1000$ are:

1. **$n = 738$ ($k=27, j=9$, Gap: $+0.000745$):** Vacancy deletion defect relaxation on the $w=13, D=17$ corrugated core. Targeting $L < 27.685$ by relaxing outer flank perimeter teeth.
2. **The $k^2+1$ Family ($n \in \{626, 677, 730, 785, 842, 901, 962\}$):** Deploying rational-angle $\arctan(1/2)$ or dual-pocket corner insertions to shatter the rigid $0.520815$ step barrier.
3. **$n = 634$ ($k=25, j=9$, Gap: $+0.000826$):** Decomposing the monolithic $45^\circ$ band into a corrugated or hinged strip to bridge between $L(633) = 25.656854$ and $L(635) = 25.707107$.
4. **$n = 852$ ($k=29, j=11$, Gap: $+0.000504$) and $n = 911$ ($k=30, j=11$, Gap: $+0.000605$):** Absorbing the 11th surplus square into the Pell-17 corrugated core via subtle non-$45^\circ$ flank tilts, avoiding the $k + 0.707107$ plateau.
