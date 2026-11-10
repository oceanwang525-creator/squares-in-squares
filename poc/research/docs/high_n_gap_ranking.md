# High-$n$ Excess Gap Ranking & Structural Audit ($n \in [600, 1000]$)

**Regenerated:** 2026-11-10 from the current `merged.csv` via `poc/research/poc/rank_high_n_gaps.py`; narrative figures edited by hand to match (the script's hardcoded prose is stale).  
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

1. **$k^2+1$ Boundary Steps (7 Targets, all in Top 11):**
   Every single $k^2+1$ instance in $600..1000$ ($n \in \{626, 677, 730, 785, 842, 901, 962\}$) occupies the upper echelon of the ranking (Ranks 1, 3, 4, 5, 7, 9, and 11). In every case, crossing from the $k \times k$ Cartesian integer grid into $k^2+1$ incurs an identical left-delta step of $\Delta L = 0.52081528$, jumping to $L = k + 0.520815$ (the Pell-17 $D=17$ diagonal strip). Because this step size is invariant while $\sqrt{n}$ grows, the excess gap decays monotonically from $0.001038$ at $n=626$ down to $0.000581$ at $n=962$.

2. **Corrugation Deficit Windows & Core Truncation Plateaus:**
   Since the first version of this audit, `merged.csv` has absorbed relaxed incumbents around the Pell-17 cores: $n = 738$ now has $L = 27.679381$ (Rank 14, $\text{gap} = 0.000464$, $-1.15 \times 10^{-2}$ vs public), $n = 851$ has $L = 29.684144$ (Rank 16, $\text{gap} = 0.000373$) and $n = 910$ has $L = 30.679154$ (Rank 19, $\text{gap} = 0.000328$). The old 738/739 plateau ($L \approx 27.6909$) is gone: $n = 739$ ($L = 27.690921$, $-3.90 \times 10^{-5}$ vs public, Rank 50) now sits $0.011540$ above $n = 738$. Remaining high-excess flanks in this family are $n = 793$ (Rank 13, $\text{gap} = 0.000547$), $n = 971$ (Rank 10, $\text{gap} = 0.000605$), $n = 972$ (Rank 18) and $n = 911$ (Rank 21).

3. **Middle-Band $j \in [7, 13]$ Dilations:**
   Intermediate square deficits $j \in [7, 13]$ require multi-layer tilted bands or dense multi-angle clusters. The highest non-boundary excess target is **$n = 634$** ($25^2+9$, Rank 2, $\text{gap} = 0.000943$), which incurs a large container inflation ($L = 25.707107 \approx 25 + 1/\sqrt{2}$) with zero right delta into $n = 635$. Other major middle-band dilations include $n = 687$ ($26^2+11$, Rank 6, $\text{gap} = 0.000730$), $n = 686$ ($26^2+10$, Rank 8) and $n = 685$ ($26^2+9$, Rank 12).

---

## 2. Top 50 High-$n$ Excess Targets Ledger

| Rank | $n$ | $k$ | $j$ | $L(n)$ | $r(n)$ | Local Baseline | Excess Gap | $\Delta L_{\text{left}}$ | $\Delta L_{\text{right}}$ | Benchmark Gap (vs Public) | Construction | Category |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **626** | 25 | 1 | `25.520815` | `0.020017` | `0.018979` | **`+0.001038`** | `+0.520815` | `+0.014719` | `0.0` (Tied) | `grid_tilted_bands` | k^2+1 Boundary Step |
| 2 | **634** | 25 | 9 | `25.707107` | `0.020960` | `0.020016` | **`+0.000943`** | `+0.050253` | `-0.000000` | `0.0` (Tied) | `grid_tilted_bands` | Middle-Band Dilation |
| 3 | **677** | 26 | 1 | `26.520815` | `0.019278` | `0.018340` | **`+0.000937`** | `+0.520815` | `+0.014719` | `0.0` (Tied) | `grid_tilted_bands` | k^2+1 Boundary Step |
| 4 | **730** | 27 | 1 | `27.520815` | `0.018591` | `0.017743` | **`+0.000848`** | `+0.520815` | `+0.014719` | `0.0` (Tied) | `grid_tilted_bands` | k^2+1 Boundary Step |
| 5 | **785** | 28 | 1 | `28.520815` | `0.017952` | `0.017182` | **`+0.000770`** | `+0.520815` | `+0.014719` | `0.0` (Tied) | `multi_angle_cluster` | k^2+1 Boundary Step |
| 6 | **687** | 26 | 11 | `26.727922` | `0.019734` | `0.019004` | **`+0.000730`** | `+0.020815` | `+0.000000` | `0.0` (Tied) | `grid_tilted_bands` | Middle-Band Dilation |
| 7 | **842** | 29 | 1 | `29.520815` | `0.017354` | `0.016655` | **`+0.000699`** | `+0.520815` | `+0.014719` | `0.0` (Tied) | `multi_angle_cluster` | k^2+1 Boundary Step |
| 8 | **686** | 26 | 10 | `26.707107` | `0.019682` | `0.019004` | **`+0.000679`** | `+0.022963` | `+0.020815` | `0.0` (Tied) | `grid_tilted_bands` | Middle-Band Dilation |
| 9 | **901** | 30 | 1 | `30.520815` | `0.016796` | `0.016159` | **`+0.000637`** | `+0.520815` | `+0.014719` | `0.0` (Tied) | `multi_angle_cluster` | k^2+1 Boundary Step |
| 10 | **971** | 31 | 10 | `31.676795` | `0.016557` | `0.015952` | **`+0.000605`** | `+0.037807` | `+0.007266` | `-1.98e-03` | `multi_angle_cluster` | Corrugation Deficit Window |
| 11 | **962** | 31 | 1 | `31.520815` | `0.016272` | `0.015691` | **`+0.000581`** | `+0.520815` | `+0.014719` | `0.0` (Tied) | `multi_angle_cluster` | k^2+1 Boundary Step |
| 12 | **685** | 26 | 9 | `26.684144` | `0.019549` | `0.018988` | **`+0.000561`** | `+0.041493` | `+0.022963` | `-8.24e-03` | `multi_angle_cluster` | Middle-Band Dilation |
| 13 | **793** | 28 | 9 | `28.678780` | `0.018413` | `0.017866` | **`+0.000547`** | `+0.041750` | `+0.005364` | `0.0` (Tied) | `multi_angle_cluster` | Corrugation Deficit Window |
| 14 | **738** | 27 | 9 | `27.679381` | `0.018892` | `0.018428` | **`+0.000464`** | `+0.040489` | `+0.011540` | `-1.15e-02` | `multi_angle_cluster` | Corrugation Deficit Window |
| 15 | **902** | 30 | 2 | `30.535534` | `0.016722` | `0.016330` | **`+0.000392`** | `+0.014719` | `+0.000000` | `0.0` (Tied) | `multi_angle_cluster` | Middle-Band Dilation |
| 16 | **851** | 29 | 10 | `29.684144` | `0.017559` | `0.017187` | **`+0.000373`** | `+0.027289` | `+0.015453` | `-6.82e-03` | `pell17_corrugated_band` | Corrugation Deficit Window |
| 17 | **791** | 28 | 7 | `28.637029` | `0.018216` | `0.017866` | **`+0.000349`** | `+0.028317` | `-0.000000` | `-1.57e-07` | `multi_angle_cluster` | Middle-Band Dilation |
| 18 | **972** | 31 | 11 | `31.684061` | `0.016267` | `0.015926` | **`+0.000340`** | `+0.007266` | `+0.012670` | `-8.33e-03` | `multi_angle_cluster` | Corrugation Deficit Window |
| 19 | **910** | 30 | 10 | `30.679154` | `0.017004` | `0.016676` | **`+0.000328`** | `+0.022300` | `+0.016468` | `-1.18e-02` | `pell17_corrugated_band` | Corrugation Deficit Window |
| 20 | **627** | 25 | 2 | `25.535534` | `0.019791` | `0.019467` | **`+0.000324`** | `+0.014719` | `+0.000000` | `0.0` (Tied) | `grid_tilted_bands` | Middle-Band Dilation |
| 21 | **911** | 30 | 11 | `30.695622` | `0.016991` | `0.016676` | **`+0.000315`** | `+0.016468` | `+0.007638` | `-1.15e-02` | `multi_angle_cluster` | Corrugation Deficit Window |
| 22 | **796** | 28 | 12 | `28.721540` | `0.018008` | `0.017697` | **`+0.000311`** | `+0.017544` | `+0.012440` | `-6.38e-03` | `multi_angle_cluster` | Middle-Band Dilation |
| 23 | **963** | 31 | 2 | `31.535534` | `0.016218` | `0.015910` | **`+0.000308`** | `+0.014719` | `+0.000000` | `0.0` (Tied) | `multi_angle_cluster` | Middle-Band Dilation |
| 24 | **852** | 29 | 11 | `29.699596` | `0.017491` | `0.017187` | **`+0.000305`** | `+0.015453` | `+0.007510` | `-7.51e-03` | `multi_angle_cluster` | Corrugation Deficit Window |
| 25 | **786** | 28 | 2 | `28.535534` | `0.017829` | `0.017524` | **`+0.000304`** | `+0.014719` | `+0.000000` | `0.0` (Tied) | `multi_angle_cluster` | Middle-Band Dilation |
| 26 | **629** | 25 | 4 | `25.583782` | `0.020092` | `0.019791` | **`+0.000301`** | `+0.048248` | `+0.020819` | `0.0` (Tied) | `multi_angle_cluster` | Middle-Band Dilation |
| 27 | **742** | 27 | 13 | `27.741947` | `0.018439` | `0.018145` | **`+0.000294`** | `+0.014025` | `+0.012150` | `-7.57e-03` | `multi_angle_cluster` | Middle-Band Dilation |
| 28 | **745** | 27 | 16 | `27.788344` | `0.018086` | `0.017793` | **`+0.000293`** | `+0.028338` | `+0.010690` | `0.0` (Tied) | `multi_angle_cluster` | General Tilted Interior |
| 29 | **636** | 25 | 11 | `25.727922` | `0.020178` | `0.019886` | **`+0.000292`** | `+0.020815` | `+0.016813` | `0.0` (Tied) | `multi_angle_cluster` | Middle-Band Dilation |
| 30 | **741** | 27 | 12 | `27.727922` | `0.018611` | `0.018319` | **`+0.000292`** | `+0.020928` | `+0.014025` | `0.0` (Tied) | `grid_tilted_bands` | Middle-Band Dilation |
| 31 | **736** | 27 | 7 | `27.637029` | `0.018714` | `0.018427` | **`+0.000287`** | `+0.028317` | `+0.001862` | `-1.57e-07` | `multi_angle_cluster` | Middle-Band Dilation |
| 32 | **646** | 25 | 21 | `25.884323` | `0.018405` | `0.018129` | **`+0.000276`** | `+0.019710` | `+0.005645` | `-5.07e-03` | `multi_angle_cluster` | General Tilted Interior |
| 33 | **758** | 27 | 29 | `27.958641` | `0.015504` | `0.015234` | **`+0.000270`** | `+0.026719` | `+0.011854` | `0.0` (Tied) | `multi_angle_cluster` | General Tilted Interior |
| 34 | **691** | 26 | 15 | `26.786712` | `0.019015` | `0.018745` | **`+0.000270`** | `+0.015396` | `+0.007488` | `0.0` (Tied) | `multi_angle_cluster` | General Tilted Interior |
| 35 | **644** | 25 | 19 | `25.856197` | `0.018877` | `0.018615` | **`+0.000262`** | `+0.021721` | `+0.008416` | `0.0` (Tied) | `multi_angle_cluster` | General Tilted Interior |
| 36 | **800** | 28 | 16 | `28.779288` | `0.017501` | `0.017250` | **`+0.000252`** | `+0.019430` | `+0.001848` | `-1.85e-03` | `multi_angle_cluster` | General Tilted Interior |
| 37 | **843** | 29 | 2 | `29.535534` | `0.017258` | `0.017008` | **`+0.000250`** | `+0.014719` | `+0.000000` | `0.0` (Tied) | `multi_angle_cluster` | Middle-Band Dilation |
| 38 | **854** | 29 | 13 | `29.727922` | `0.017269` | `0.017019` | **`+0.000250`** | `+0.020815` | `+0.006058` | `0.0` (Tied) | `multi_angle_cluster` | Middle-Band Dilation |
| 39 | **683** | 26 | 7 | `26.637029` | `0.019238` | `0.018988` | **`+0.000249`** | `+0.028317` | `+0.005621` | `-1.86e-03` | `multi_angle_cluster` | Middle-Band Dilation |
| 40 | **936** | 30 | 36 | `30.997511` | `0.013185` | `0.012938` | **`+0.000247`** | `+0.017629` | `+0.002489` | `-2.49e-03` | `multi_angle_cluster` | General Tilted Interior |
| 41 | **969** | 31 | 8 | `31.633983` | `0.016230` | `0.015990` | **`+0.000239`** | `+0.025270` | `+0.005006` | `0.0` (Tied) | `multi_angle_cluster` | Middle-Band Dilation |
| 42 | **797** | 28 | 13 | `28.733980` | `0.017810` | `0.017578` | **`+0.000232`** | `+0.012440` | `+0.011623` | `-7.93e-03` | `multi_angle_cluster` | Middle-Band Dilation |
| 43 | **973** | 31 | 12 | `31.696731` | `0.016151` | `0.015922` | **`+0.000228`** | `+0.012670` | `+0.010376` | `-1.04e-02` | `multi_angle_cluster` | Middle-Band Dilation |
| 44 | **630** | 25 | 5 | `25.604601` | `0.020112` | `0.019887` | **`+0.000225`** | `+0.020819` | `+0.004111` | `0.0` (Tied) | `multi_angle_cluster` | Middle-Band Dilation |
| 45 | **635** | 25 | 10 | `25.707107` | `0.020155` | `0.019938` | **`+0.000217`** | `-0.000000` | `+0.020815` | `0.0` (Tied) | `grid_tilted_bands` | Middle-Band Dilation |
| 46 | **803** | 28 | 19 | `28.820376` | `0.017049` | `0.016839` | **`+0.000210`** | `+0.019248` | `+0.009339` | `-6.08e-03` | `multi_angle_cluster` | General Tilted Interior |
| 47 | **992** | 31 | 31 | `31.919957` | `0.013460` | `0.013251` | **`+0.000208`** | `+0.017994` | `+0.004834` | `-2.60e-05` | `multi_angle_cluster` | General Tilted Interior |
| 48 | **690** | 26 | 14 | `26.771316` | `0.019167` | `0.018960` | **`+0.000207`** | `+0.025691` | `+0.015396` | `-5.36e-03` | `multi_angle_cluster` | Middle-Band Dilation |
| 49 | **857** | 29 | 16 | `29.769231` | `0.016898` | `0.016698` | **`+0.000200`** | `+0.023611` | `+0.011906` | `0.0` (Tied) | `multi_angle_cluster` | General Tilted Interior |
| 50 | **739** | 27 | 10 | `27.690921` | `0.018627` | `0.018428` | **`+0.000199`** | `+0.011540` | `+0.016073` | `-3.90e-05` | `pell17_corrugated_band` | Corrugation Deficit Window |

---

## 3. Structural Regime Deep Dive & Anomalies

### 3.1 The $k^2+1$ Boundary Step Anomaly

Across all $k \in [25, 31]$, the step from the perfect square grid $n = k^2$ ($L = k$) to $n = k^2+1$ exhibits a rigid, invariant delta of exactly:
$$\Delta L_{\text{step}} = L(k^2+1) - L(k^2) = 0.52081528$$
This value corresponds to the uncorrugated $D=17, w=1$ diagonal boundary strip. Inserting even a single extra square forces container dilation across the entire diagonal.

| $k$ | $k^2$ | $n = k^2+1$ | $L(k^2+1)$ | $\Delta L_{\text{left}}$ | Local Baseline $r$ | Excess Gap | Rank |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |

| 25 | 625 | **626** | `25.520815` | `+0.520815` | `0.018979` | **`+0.001038`** | #1 |
| 26 | 676 | **677** | `26.520815` | `+0.520815` | `0.018340` | **`+0.000937`** | #3 |
| 27 | 729 | **730** | `27.520815` | `+0.520815` | `0.017743` | **`+0.000848`** | #4 |
| 28 | 784 | **785** | `28.520815` | `+0.520815` | `0.017182` | **`+0.000770`** | #5 |
| 29 | 841 | **842** | `29.520815` | `+0.520815` | `0.016655` | **`+0.000699`** | #7 |
| 30 | 900 | **901** | `30.520815` | `+0.520815` | `0.016159` | **`+0.000637`** | #9 |
| 31 | 961 | **962** | `31.520815` | `+0.520815` | `0.015691` | **`+0.000581`** | #11 |

**Key Takeaway for Solvers:** The $k^2+1$ targets represent the most predictable, high-value candidates in $600..1000$. Replacing the rigid $45^\circ$ single strip with a rational-angle tilted pocket (such as $\arctan(1/2) \approx 26.565^\circ$) or out-of-phase pinch corrugation could contract this step from $0.5208$ down toward $\sim 0.44$, yielding gains of $\Delta L \approx 0.08$ on each boundary step.

### 3.2 Corrugation Deficit Windows Around Pell-17 Cores

The border-scan optimization originally incorporated four certified Pell-17 corrugated wins into `merged.csv`. Current `merged.csv` values (regenerated 2026-11-10):
- $n = 739$ ($k=27, j=10$): $L = 27.69092108$, $\Delta L_{\text{public}} = -3.90 \times 10^{-5}$ (Rank 50)
- $n = 794$ ($k=28, j=10$): $L = 28.68414361$, $\Delta L_{\text{public}} = -6.82 \times 10^{-3}$ (Rank 93)
- $n = 851$ ($k=29, j=10$): $L = 29.68414361$, $\Delta L_{\text{public}} = -6.82 \times 10^{-3}$ (Rank 16)
- $n = 910$ ($k=30, j=10$): $L = 30.67915402$, $\Delta L_{\text{public}} = -1.18 \times 10^{-2}$ (Rank 19)

Neighbor observations:
1. **$n=738$ is no longer a plateau (Rank 14):** $L(738) = 27.67938125$ and $L(739) = 27.69092108$, so $\Delta L_{\text{right}} = 0.011540$. The earlier figure $L(738) = 27.690919$ is superseded; the remaining excess $+0.000464$ is no longer an unrelaxed truncation.
2. **Post-core flanks:** $n = 852$ ($L = 29.699596$, Rank 24) and $n = 911$ ($L = 30.695622$, Rank 21) no longer jump to $k + 1/\sqrt{2}$; their right deltas are $+0.007510$ and $+0.007638$.
3. **Public Leader Deficits at $n=971, 972$:** $n=971$ (Rank 10, $\text{gap} = 0.000605$) and $n=972$ (Rank 18, $\text{gap} = 0.000340$) remain among the largest non-boundary excesses.

### 3.3 Middle-Band $j \in [7, 13]$ Dilations & Plateaus

Outside the $k^2+1$ steps and Pell windows, the largest excess gaps are concentrated in intermediate offsets $j \in [7, 13]$:
- **$n = 634$ ($k=25, j=9$, Rank 2, $\text{gap} = 0.000943$):** $L = 25.707107$. $\Delta L_{\text{left}} = 0.050253$ and flat into $n=635$. This $j=9$ bottleneck corresponds to the onset of a 2-layer upright flank constraint against a $45^\circ$ core.
- **$n = 687$ ($k=26, j=11$, Rank 6, $\text{gap} = 0.000730$):** $L = 26.727922$, flat into $n=688$.
- **$n = 686$ ($k=26, j=10$, Rank 8, $\text{gap} = 0.000679$):** $L = 26.707107$.
- **$n = 685$ ($k=26, j=9$, Rank 12, $\text{gap} = 0.000561$):** $L = 26.684144$, $\Delta L_{\text{left}} = 0.041493$.
- **$n = 742$ ($k=27, j=13$, Rank 27, $\text{gap} = 0.000294$):** $L = 27.741947$.


---

## 4. Priority Attack Roster for Research Solvers

Based on localized excess gap, container rigidity, and structural solvability, the top target priorities in $600..1000$ are:

1. **$n = 634$ ($k=25, j=9$, Gap: $+0.000943$):** Decomposing the monolithic $45^\circ$ band into a corrugated or hinged strip to bridge toward $L(633)$ and $L(635) = 25.707107$.
2. **The $k^2+1$ Family ($n \in \{626, 677, 730, 785, 842, 901, 962\}$):** Deploying rational-angle $\arctan(1/2)$ or dual-pocket corner insertions to shatter the rigid $0.520815$ step barrier.
3. **$n = 687, 686$ ($k=26$, Gaps: $+0.000730$, $+0.000679$):** Flat $L \approx 26.707$-$26.728$ plateau; look for a sub-$45^\circ$ tilted flank.
4. **$n = 971$ ($k=31, j=10$, Gap: $+0.000605$) and $n = 793$ ($k=28, j=9$, Gap: $+0.000547$):** Highest-ranked Corrugation Deficit Windows still unrelaxed.
