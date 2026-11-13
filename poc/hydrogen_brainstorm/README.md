# Hydrogen Brainstorm: New Constructions & Search Families Beyond 45° Bands

- **Current Certified Incumbent (`merged.csv`)**: `22.173715` (`score_ours_float = 22.173715450068446`)
- **Per-$n$ Public Frontier (`per_n_gap.csv`)**: `22.050884` (`total_positive_gap = 0.124158` across **401** losing $n$)
- **Target**: `22.0999` (need `-0.073815` total score reduction = `59.5%` of the gap to frontier)
- **Status**: **0 certified wins in Top-3 POCs; 2 strong structural leads (POC 1, POC 3 taper), 1 clean negative falsification (POC 2 + checkerboard herringbone).** Nothing is claimed as a win unless it strictly lowers $L$ for the same $n$, has zero overlaps, is non-grid, and passes both the official literal check and pandas 3.1.0 roundtrip.

---

## 1. Dead Ends Already Exhausted (Not Re-Proposed)

From auditing [`poc/loop_log.md`](../loop_log.md), all 77 `poc/*/README.md`, and the `claude` branch (`diamond_cell_milp`, `hard_square_npt_mc`, `slope_interface_scan`), the following families are exhausted and excluded from this brainstorm:

1. **Local LP / SLP / Epigraph Polish on Fixed Incumbents**: Loops 1–28 and `lip_slp` / `alln_fixed_lp` converged existing contact topologies down to machine-precision KKT optima (`0` remaining headroom in `45_only` or existing basins).
2. **Rigid $45^\circ$ / Pell / Corrugated / Hinged Bands**: $D \in \{1, 2, 7, 17, 41\}$ bands, upright border wraps (`poc/pell_corrugation`, `diamond_cell_milp`), and zipper/donor-slice surgery (`arm_gadget`, `strip_cell` where $m=14, P=1.9838$ proved infeasible for $W \ge 8.0$).
3. **Straight-Slope Band Scans & Corner-Anchored Lattices**: `slope_interface_scan` tested periodic $k=3$ teeth cells against corner-anchored lattices at shift $(-f, f)$ and never beat the trivial shifted-lattice baseline $W_{\text{base}} = 2fL$.
4. **Local Deletion / Insertion & MC / Freeze-Relax from Own Incumbents**: `hard_square_npt_mc` ran NPT MC from 1–3% expanded incumbents and scored `0` wins because single-basin compression cannot jump grain-boundary topological winding numbers.
5. **Harvesting Public Kernels / `.npz` / Kaggle Banks**: Already saturated against public releases.

---

## 2. Empirical Gap Anatomy & Non-$45^\circ$ Diagnostic (`gap_summary.json`, `math_tables.json`)

Measured directly from [`merged.csv`](../pell_corrugation/data/border-scan/merged.csv) and [`per_n_gap.csv`](../frontier_nov12/docs/per_n_gap.csv) via [`docs/gap_summary.json`](docs/gap_summary.json):

| Incumbent Regime | Losing $n$ Count | $\sum \Delta S_+$ | Mean $\Delta L$ | Share of Positive Gap |
|---|---:|---:|---:|---:|
| `has_non45` | 370 | `0.111532` | `0.007902` | **89.8%** |
| `45_only` | 25 | `0.011849` | `0.011343` | **9.5%** |
| `grid_only` | 6 | `0.000776` | `0.003892` | **0.6%** |
| **Total** | **401** | **`0.124158`** | — | **100.0%** |

### Key Structural Clues Observed in Code
1. **$D=1$ Roots ($n = k^2 + 1$, 31 roots total, 12 losing for $\sum \Delta S = 0.006812$)**:
   - Up through $k=19$ ($n=362$), the frontier sits on `f = 0.535534` ($45^\circ$). Starting at $k=20$ ($n=401$), the frontier fractional part drops monotonically:
     - $n=401$ ($k=20$): `L_ours = 20.535534` vs `L_front = 20.526135` ($f = 0.52614$)
     - $n=530$ ($k=23$): `L_ours = 23.535534` vs `L_front = 23.517305` ($f = 0.51730$)
     - $n=626$ ($k=25$): `L_ours = 25.520815` vs `L_front = 25.511049` ($f = 0.51105$)
     - $n=842$ ($k=29$): `L_ours = 29.520815` vs `L_front = 29.502153` ($f = 0.50215$)
     - $n=962$ ($k=31$): `L_ours = 31.520815` vs `L_front = 31.502153` ($f = 0.50215$)
   - Notice $f \to 0.502153$ saturates at large $k$ rather than shrinking $O(1/k)$, indicating a **bounded-width tilted/sheared corner or edge gadget** whose $O(1)$ geometry unlocks at $k \ge 20$.
2. **Angle Decomposition of High-Performing Non-$45^\circ$ Incumbents (`math_tables.json`)**:
   - Inspecting our own best non-$45^\circ$ incumbents reveals **Zero-Sawtooth Sheared-Row Pockets**, NOT rigid rotated lattices:
     - `n=50` ($L=7.571429$): `34` at $0^\circ$, `16` at **Pythagorean 3-4-5** (`36.870°`).
     - `n=269` ($L=16.889730$, tied with frontier): `217` at $0^\circ$, `34` at `41.970°`.
     - `n=598` ($L=24.903160$): `342` at $0^\circ$, `35` at `18.606°`, `20` at `18.611°`, `18` at `17.922°`, `9` at `18.984°`.
     - `n=913` ($L=30.721540$): `501` at $0^\circ$, `232` at `19.085°`.
     - `n=971` ($L=31.664318$): `681` at $0^\circ$, `164` at `25.202°`, `20` at `24.859°`.
   - In `n=971`, a pocket of width $W = 7 + f = 7.6643$ fits $w=8$ edge-to-edge squares at $\theta = 25.202^\circ$ spanning $8\cos\theta + \sin\theta = 7.6643$ with **vertical pitch $\sec\theta$ and zero sawtooth gap against the parallel wall**!

---

## 3. Ranked Portfolio of 12 Genuinely New Ideas

Ranked by **Expected Score Value** ($\text{EV} = P(\text{breakthrough}) \times \sum \Delta S_{\text{target}}$). Every idea is geometrically distinct from Loops 1–28 and the `claude` branch.

### Rank 1 — Zero-Sawtooth Sheared-Row L-Pockets with Mitered Corner & Continuous Taper *(POC 1)*
- **Geometric Intuition**: Rather than rotating a 2D square lattice inside a boundary strip (which loses $\sim \frac{1}{2}$ square per unit length in sawtooth triangles along flat walls), place $w = P + 1$ (or $P + 2$) unit squares **edge-to-edge along $(\cos\theta, \pm\sin\theta)$** inside a side pocket of width $W = P + f$, solving $w\cos\theta + \sin\theta = W$. Stack rows vertically with zero horizontal drift at pitch $\Delta y = \sec\theta$. Pair two pockets (West & South) around a $K_x \times K_y$ upright core, joined at the SW corner by either an upright corner sub-block or a $45^\circ$-ish **mitered seam** with continuous angle tapering on the terminal rows to recover end triangles.
- **Target $n$ Families**: High-$n$ `has_non45` and `45_only` with $K \ge 18$ ($n=598, 634, 791, 793, 908, 910, 911, 969, 971, 972, 973$, plus $D=1$ at $k \ge 20$ when end-filled). Covers ~220 losing $n$, $\sum \Delta S \approx 0.085$.
- **$L$ & Score Gain Math**:
  - Linear square density per unit height is $\rho_{\text{shear}} = w\cos\theta = P + f - \sin\theta$.
  - Gain over $P$ upright columns is $\Delta\rho = f - \sin\theta > 0$ whenever $\sin\theta < f$.
  - Because $w\cos\theta + \sin\theta = \sqrt{w^2+1}\cos(\theta - \arctan(1/w)) = W$, the exact tilt angle is $\theta = \arctan(1/w) + \arccos(W / \sqrt{w^2+1})$. Numerically at $W=7.661, w=8$ ($f=0.661 \implies \theta=25.278^\circ$, [`poc3_skyline_dp.json`](docs/poc3_skyline_dp.json)): $\Delta\rho = 0.661 - 0.4270 = +0.234$ squares/unit height ($\sim +7.4$ bulk squares over height $31.66$), minus single-end triangle waste $E_1 = \frac{1}{2} W^2 \tan\theta$ (cut roughly in half by miter/taper).
  - Recovering $\Delta N = +3$ squares on $n \approx 911$ ($K=30$) reduces $L$ by $\Delta L \approx \frac{\Delta N}{2K} \approx 0.049$ (or $\Delta S \approx \frac{0.049}{30.2} \approx 0.0016$), exceeding the full gap on top losers (`dL = 0.025..0.035`).
- **Cheap Falsification Test (<10 CPU min)**: Analytic sweep over $(K_x, K_y, P_1, P_2, w_1, w_2, \text{signs}, \text{SW mode})$ at $L_{\text{ours}} - 0.002$ followed by greedy void fill + L-BFGS + fixed-angle LP tighten on top candidates (**implemented & run in [`poc/poc1_sheared_pockets.py`](poc/poc1_sheared_pockets.py)**).

---

### Rank 2 — Continuous End-Taper / Smectic Fan Unfolding Caps for Finite Strips *(Tested inside POC 3)*
- **Geometric Intuition**: In a uniform sheared pocket of width $W$ and angle $\theta$, the terminal row rises diagonally across width $W$ by height $W\tan\theta$, wasting empty triangle area $E_1 = \frac{1}{2}W^2\tan\theta$ at both top and bottom ends (`math_tables.json` shows raw empty end waste $= 8.2\text{--}12.4$ squares even at $P=4$, killing $D=1$ gains if left raw!). Gradually **unfolding row tilt angle** $\theta_r \to \theta_{r-1} \to \dots \to 0^\circ$ (a smooth smectic fan / upright starter transition of 1–4 rows) flattens the slanted boundary wedge against the orthogonal wall without triggering full-height upright fallback.
- **Target $n$ Families**: All finite sheared pockets ($K \in [15, 32]$), especially `D=1` ($n=401, 530, 626, 842, 962$) and top losers `911, 910, 971, 972, 634`. Estimated $\sum \Delta S \approx 0.035\text{--}0.055$.
- **$L$ & Score Gain Math**:
  - Two raw empty ends of a $w$-square pocket of width $W$ waste $W^2\tan\theta$ area. Replacing a fraction $\eta \in [0.25, 0.5]$ of each end triangle by an angled fan recovers $\Delta N \approx \eta W^2 \tan\theta$ squares.
  - For $W=7.661, \theta=25.28^\circ$: $\frac{1}{2}W^2\tan\theta \approx 13.85$; saving even `0.15` height per end recovers $\sim 1.1$ squares/pocket, giving $\Delta L \approx \frac{1.1}{2L} \approx 0.017$ at $L \approx 31$.
- **Cheap Falsification Test (<10 CPU min)**: Compare compacted box height $H$ of `uniform_shear` vs `upright_bottom_taper` inside fixed $[0, W]$ strips via bisection + L-BFGS (**implemented & observed positive on 4 widths in [`poc/poc3_skyline_dp.py`](poc/poc3_skyline_dp.py)**). Next step: multi-row smooth angle schedule $\theta_r = \frac{r}{R}\theta$.

---

### Rank 3 — De Novo Inflation Compression from Oriented Multi-Grain Seeds (a: 0.88 → 1.00)
- **Geometric Intuition**: Previous `hard_square_npt_mc` failed (`0` wins) because it started from compressed incumbents trapped in rigid grain topologies. Instead, initialize $n$ tiny squares (side $a_0 = 0.88$) inside the **exact target box $[0, L_{\text{target}}]^2$** ($L_{\text{target}} = L_{\text{frontier}} + 0.002$) with **soft macro-domain angle priors** (e.g., two orthogonal pockets with smooth angle field $\theta(x,y) \in \{\pm 20^\circ, 0^\circ\}$ or 4-fold chiral swirl) and run Lubachevsky–Stillinger / adaptive L-BFGS inflation $a \uparrow 1.0$ with periodic shear-row alignment + swap moves at $a \in [0.92, 0.97]$ before jam lock-in.
- **Target $n$ Families**: Top-50 gap losers (`911, 367, 910, 972, 791, 634, 973, 793, 908, 969`) plus all `D=1` roots (`401..962`). Target pool $\sum \Delta S \approx 0.060$.
- **$L$ & Score Gain Math**: At fixed $n$, if inflation locks cleanly to $a=1.0$ at $L_{\text{target}} = L_{\text{ours}} - \delta L$, the per-$n$ score gain is $\Delta S = \delta L / \sqrt{n}$. Reaching halfway to frontier on the top 40 `has_non45` losers gives $\sum_{40} \frac{1}{2}\Delta S_{\text{top40}} \approx 0.016$.
- **Cheap Falsification Test (<10 CPU min)**: 50 restarts × 8 inflation steps ($a: 0.88 \to 1.0$) using Numba SAT gradient on $n=50$ (calibration against known $7.5714$) and $n=401, 598$ on 8 CPU cores (~4 min total).

---

### Rank 4 — Asymmetric 4-Edge "Pinched Frame" (4 Independent Sheared Boundary Strips)
- **Geometric Intuition**: Almost all our incumbents break square packing into a corner-tucked upright block + 2 L-pockets. A **4-edge pinched frame** surrounds an off-center/rotated upright or tilted core with **4 narrow sheared strips** along North, South, East, and West walls, each choosing its own width $W_i = P_i + f_i$ ($\sum f_i$ coupled across opposite pairs modulo integer offsets) so that every side has $\sin\theta_i < f_i$ simultaneously, converting 4-corner waste into 4 small cyclic corner wedges.
- **Target $n$ Families**: Non-$D=1$ large-$n$ with $f \in [0.15, 0.45]$ or $[0.60, 0.90]$ where an L-pocket leaves excess $f$ unused along one axis ($n \in [400, 1000]$, ~150 losing $n$).
- **$L$ & Score Gain Math**: If a single width-$W$ pocket leaves $f_1 = 0.30$ too small for shear ($\sin\theta > 0.30$ unless $w \ge 12$), splitting side slack unevenly as $W_N = P_N + 0.72$, $W_S = P_S + (f - 0.72 + 1)$ (Wait: $W_N + K_y + W_S = L$, so fractional sum $f_N + f_S \equiv f \pmod 1$ allows **$f_N = 0.72, f_S = 0.78$ when $f = 0.50$!!**). At $f_N = 0.75, w = 6 \implies \theta \approx 22.8^\circ, \sin\theta = 0.388 \implies \Delta\rho = 0.75 - 0.388 = +0.362$/unit height on *both* pockets!
- **Cheap Falsification Test (<10 CPU min)**: Integer decomposition sweep $W_1 + K_{\text{core}} + W_2 = L$ with $f_1 + f_2 \in \{f, f+1\}$ for $n=401, 530, 842, 911$, testing feasibility of $w_i \cos\theta_i + \sin\theta_i = W_i$ (~1 min).

---

### Rank 5 — 4-Fold Chiral Pinwheel / Tilted Core + Orthogonal Tapered Wings
- **Geometric Intuition**: Instead of an upright central core touching sheared wall pockets, place a **tilted $A \times B$ rectangular core** at angle $\phi \in (0, 45^\circ)$ in the interior, flanked by 4 symmetric congruent triangular/trapezoidal wings rotated by $0^\circ, 90^\circ, 180^\circ, 270^\circ$ ($C_4$ chiral symmetry). In exact $C_4$ symmetry, the bounding square side is controlled continuously by core rotation $\phi$ and radial wing offset, directly bypassing rigid $L \in \mathbb{Z} + k/\sqrt{2}$ lattices at **$D=1$ roots**.
- **Target $n$ Families**: Square-plus-small-excess $n = k^2 + r$ ($r \in \{1..5\}$, including **$D=1$ roots $n=401, 442, 485, 530, 577, 626, 677, 730, 785, 842, 901, 962$**) and $C_4$-symmetric $4m+1$ / $4m$ sizes. Target $\sum \Delta S \approx 0.012$.
- **$L$ & Score Gain Math**: An upright core $m \times m$ tilted at angle $\phi$ spans $m(\cos\phi + \sin\phi)$ along the diagonal; surrounding it with 4 identical $w \times h$ upright/sheared blocks yields outer side $L(\phi) \approx \max(m(\cos\phi+\sin\phi), \text{wing span})$. Choosing $m \approx k-4$ with 4 identical boundary blocks converts discrete step jumps $\Delta L \approx 0.015\text{--}0.035$ into smooth $C_4$ equilibria with $f \approx 0.502\text{--}0.526$.
- **Cheap Falsification Test (<10 CPU min)**: Parametrize 1-quadrant asymmetric sub-block ($n/4$ squares) with $C_4$ periodic $90^\circ$-rotation constraints inside $[0, L]^2$ and run 200-start L-BFGS inflation for $n \in \{401, 530, 626, 842\}$ (~5 min, 4× fewer free variables!).

---

### Rank 6 — Skyline Beam-Search Transfer-Matrix DP & Alternating Row Motifs *(POC 3)*
- **Geometric Intuition**: Replace hand-engineered sheared pockets in strips $W = P + f$ with a **stateful skyline beam search** maximizing packed count per height $H$, paired with continuous local L-BFGS compaction to test whether non-periodic mixed-angle rows, interlocking pairs, or upright-to-sheared end transitions beat uniform sheared density $P + f - \sin\theta$.
- **Target $n$ Families**: Strip subproblems inside wide $L$-pockets ($W \in [2.5, 10.5]$) and narrow asymmetric cells for $n \in [200, 1000]$.
- **$L$ & Score Gain Math**: Every `+0.05` density increase over strip height $H \approx 28$ yields $+1.4$ squares per pocket ($+2.8$ across 2 pockets $\implies \Delta L \approx 0.045$ at $K=30$).
- **Cheap Falsification Test (<10 CPU min)**: Exact Numba SAT vertical-drop skyline beam search + L-BFGS motif bisection on $W \in \{2.502..7.661\}$ (**implemented & run in [`poc/poc3_skyline_dp.py`](poc/poc3_skyline_dp.py)**; negative for quantized-$x$ square-by-square beam and checkerboard herringbone, positive for upright-bottom end-taper). Future variant must snap candidates onto **contact graphs** rather than fixed-$x$ grids.

---

### Rank 7 — Pythagorean Exact-Commensurate Tilted Cores & Rational Wall Tilings
- **Geometric Intuition**: For primitive Pythagorean triples $(a, b, c)$ ($a^2+b^2=c^2$), rotating an integer $m_1 \times m_2$ lattice by $\theta = \arctan(a/b)$ makes step vectors $(c\cos\theta, -c\sin\theta) = (b, -a)$ and $(c\sin\theta, c\cos\theta) = (a, b)$ **exact integers**! That allows a tilted $c \times c$-period block to flush-weld onto upright integer blocks with **exact commensurate period $c$** instead of irrational moiré mismatch.
- **Target $n$ Families**: `grid_only` / near-integer $f$ regimes (`f ≈ 0.24, 0.308, 0.353, 0.400`), calibrated by `n=50` using $(3,4,5)$ at $L=7 + 4/7$. Exact values from [`docs/math_tables.json`](docs/math_tables.json):
  - `(3,4,5)` ($\theta=36.8699^\circ$): $\cos\theta+\sin\theta = 1.400000$ ($5$-unit period)
  - `(5,12,13)` ($\theta=22.6199^\circ$): $\cos\theta+\sin\theta = 1.307692$ ($13$-unit period)
  - `(8,15,17)` ($\theta=28.0725^\circ$): $\cos\theta+\sin\theta = 1.352941$ ($17$-unit period)
  - `(7,24,25)` ($\theta=16.2602^\circ$): $\cos\theta+\sin\theta = 1.240000$ ($25$-unit period)
- **$L$ & Score Gain Math**: An isolated $(a,b,c)$ tilted row cell of width $W = c$ spanning one $c$-period holds tilted blocks with fractional surplus $\cos\theta + \sin\theta - 1 \in \{0.240, 0.308, 0.353, 0.400\}$, matching frontier $f$ peaks near $0.24..0.40$ without irrational $L$ drift.
- **Cheap Falsification Test (<10 CPU min)**: Exact MILP/DP over integer tileable cells of period $c \in \{5, 13, 17\}$ docked to upright walls for $n \in \{171, 260, 364, 580, 803\}$ (~3 min).

---

### Rank 8 — V-Shaped / Chevron Bent Sheared Rows
- **Geometric Intuition**: A single straight sheared row of width $W$ drifts vertically by $W\tan\theta$, creating end-triangle waste $\frac{1}{2}W^2\tan\theta$. Bending each row at its midpoint into a symmetric **chevron** (left half at $+\theta$, right half at $-\theta$ with a single trapezoidal wedge vertex, OR staggered $+ \theta / -\theta$ chevron nested along $y$) cuts total vertical excursion from $W\tan\theta$ to $\frac{W}{2}\tan\theta$, **reducing quadratic end-triangle area waste by $4\times$** (from $\frac{1}{2}W^2\tan\theta$ to $2 \cdot \frac{1}{2}(W/2)^2\tan\theta \times \text{apex loss}$) while allowing nested identical chevrons to translate vertically by $\sec\theta$!
- **Target $n$ Families**: Wide pockets ($W \in [8, 16], K \ge 22$) where single-sheared end waste dominates ($n \in [500, 1000]$).
- **$L$ & Score Gain Math**: At $W = 10, \theta = 20.9^\circ$: straight end-triangle is $\frac{1}{2}(100)\tan(20.9^\circ) = 19.1$ units of area; halving excursion saves $\sim 9.5$ units of end-ramp area minus $\sim 1\text{--}2$ apex joint squares, yielding $+4$ to $+6$ net squares on $K=30$ ($\Delta L \approx 0.06$ if apex seam stays tight!).
- **Cheap Falsification Test (<10 CPU min)**: Construct a single chevron strip $[0, 10.5] \times [0, 30]$ with apex angle gap $\theta$ filled by alternating $0^\circ / 45^\circ$ keystones and run L-BFGS compaction comparing total capacity against straight $w=11$ sheared strip (~2 min).

---

### Rank 9 — Symmetric Tilt Grain-Boundary Bicrystals ($\pm \theta/2$ Interface Seam)
- **Geometric Intuition**: Putting an upright $0^\circ$ lattice directly against a $\theta$-tilted lattice wastes a triangular sawtooth notch every step along the boundary. In materials science, a **symmetric tilt grain boundary** tilts *both* lattices by $+\theta/2$ and $-\theta/2$ relative to the straight interface seam! Then **every node along the interface is a symmetric reflection-coincident vertex** ($y_{\text{step},A} = y_{\text{step},B}$ with identical period $\sec(\theta/2)$), eliminating interface sawtooth cavities between the two grains while turning wall boundaries into shallow $\pm \theta/2$ sheared pockets!
- **Target $n$ Families**: Mid-to-large $n$ (`45_only` & `has_non45`, e.g. $n=634, 686, 908, 971$) where a flat $0^\circ$/$\theta$ internal boundary leaks capacity.
- **$L$ & Score Gain Math**: Removing a $\frac{1}{2}\sin\theta$ periodic tooth cavity per unit length along a $K=25$ internal seam recovers up to $\sim 25 \times 0.15 \approx 3.7$ area units (net $+2\text{--}3$ squares once outer boundary tilt matches), worth $\Delta L \approx 0.04\text{--}0.06$ on $K=25\text{--}30$.
- **Cheap Falsification Test (<10 CPU min)**: Test $2$-grain $(+\alpha, -\alpha)$ packing inside $[0, L_{\text{ours}}-0.01]^2$ for $\alpha \in [2^\circ, 15^\circ]$ via L-BFGS + LP tighten on $n=401, 634, 908$ (~4 min).

---

### Rank 10 — Macro-Tile Guillotine + L-Cut Composition DP Over Certified Non-$45^\circ$ Slabs
- **Geometric Intuition**: `q4_multiblock_composition.md` analyzed excess preservation ($L = L_{\text{core}} + t$), but we do not yet have a global DP over **mixed-angle macro-tiles** (upright $a \times b$, sheared strip $(W, H, w, \theta)$, mitered corner wedge $M(W_1, W_2)$, tapered cap). Precomputing a small bank of exact certified variable-aspect rectangular & L-shaped macro-tiles and solving a 2D guillotine + 1-cut L-packing DP directly assembles multi-domain global layouts for any $(n, L_{\text{target}})$ without local minima.
- **Target $n$ Families**: Composite multi-$n$ sweep across all 370 `has_non45` losers once Ranks 1–5 supply a clean tile library (~0.025–0.040 total $\Delta S$).
- **$L$ & Score Gain Math**: Replacing one suboptimal sub-block in any composite incumbent by a tighter certified macro-tile of $\ge +2$ square density shifts $L$ by $\Delta L \ge \frac{2}{2L} \approx 0.032$ ($n \approx 960$).
- **Cheap Falsification Test (<10 CPU min)**: Build a 200-entry library of certified sheared rectangular slabs + corner miters and run L-decomposition DP for $n \in [500, 1000]$ in pure Python/NumPy (~3 min).

---

### Rank 11 — Off-$45^\circ$ Sturmian-Staircase Diagonal Band *(POC 2 — FALSIFIED NEGATIVE)*
- **Geometric Intuition**: Perturb classic $45^\circ$ bands to $45^\circ - \delta$ ($\delta \in [0.3^\circ, 6^\circ]$) where $\cos\theta + \sin\theta < \sqrt{2}$, shrinking diagonal band thickness while filling its stepped flanks with aperiodic 1:1/2:1 Sturmian wedge squares.
- **Target $n$ Families**: `45_only` high-gap $n$ (`328, 367, 492, 634, 686`) and off-$45^\circ$ `n=269` (`41.97°`).
- **$L$ & Score Gain Math**: Along a $45^\circ$ diagonal of length $\sqrt{2}K$, rotating by $\delta = 3^\circ$ lowers band footprint $\cos\theta + \sin\theta$ from `1.41421` to `1.41228` (`-0.00193` per square along normal), which *if* flank sawteeth did not widen would save $\Delta L \sim 0.004\text{--}0.015$.
- **Cheap Falsification Test (<10 CPU min)**: Parametrized off-$45^\circ$ diagonal strip + corner-anchored upright flanks + soft-overlap L-BFGS (**implemented & falsified in [`poc/poc2_off45_sturmian_band.py`](poc/poc2_off45_sturmian_band.py)**; rotated sawtooth flanks lose `8..16` squares before interface relaxation can even engage).

---

### Rank 12 — High-Order Rational-Slope Continuation & Pell Dual-Angle Corner Junctions
- **Geometric Intuition**: Rather than discrete straight slope bands ($1:1, 1:2$) or single $D \in \{1, 2, 7, 17\}$ Pell corrugations, interpolate boundary wedges via high-order continued-fraction convergents $p/q \in \{2/3, 3/5, 5/8, 4/7\}$ at dual-angle corner junctions where an east $p_1/q_1$ wedge meets a south $p_2/q_2$ wedge inside non-integer $f$ slack.
- **Target $n$ Families**: Medium-$n$ transition points ($n \in [100, 400]$) where `f` sits between $45^\circ$ Pell roots and sheared-pocket threshold $P \ge 6$.
- **$L$ & Score Gain Math**: At $n \approx 200$ ($K=14$), saving even $+1$ square via a $3:5$ corner junction yields $\Delta L \approx \frac{1}{28} \approx 0.036$ ($\Delta S \approx 0.0025$).
- **Cheap Falsification Test (<10 CPU min)**: Corner-wedge local MILP over $4 \times 4$ cell grids at angles $\arctan(p/q)$ docked to upright cores for $n \in \{171, 215, 269, 328\}$ (~4 min).

---

## 4. Top-3 POC Execution Results (Strictly Verified From Code)

All three POCs share [`poc/geom_utils.py`](poc/geom_utils.py) (verified Numba SAT overlap + analytical gradient, `FD max rel error = 8.59e-09`, and official check `sc.evaluate` + `sc.kaggle_roundtrip` calibrated on `n=50 -> 7.571428571428577`).

### POC 1: Zero-Sawtooth Sheared-Row L-Pockets ([`poc/poc1_sheared_pockets.py`](poc/poc1_sheared_pockets.py) → [`docs/poc1_sheared_pockets.json`](docs/poc1_sheared_pockets.json))

| $n$ | $L_{\text{ours}}$ | $L_{\text{front}}$ | $L_{\text{probe}}$ | `best_cap` | Deficit | Winning Geometry Found ($K_x \times K_y$, $P_1/w_1@\theta_1$, $P_2/w_2@\theta_2$) | Runtime |
|---:|---:|---:|---:|---:|---:|---|---:|
| 598 | `24.903160` | `24.900375` | `24.902360` | `594` | **4** | `19x18`, `P1=5/w1=6@23.452°`, `P2=6/w2=7@20.671°`, `sw_upright`, `(1,-1)` | 4.87s |
| 401 | `20.535534` | `20.526135` | `20.533534` | `394` | **7** | `15x15`, `P1=5/w1=6@33.998°`, `P2=5/w2=6@33.998°`, `sw_upright`, `(-1,-1)` | 3.29s |
| 530 | `23.535534` | `23.517305` | `23.533534` | `523` | **7** | `18x18`, `P1=5/w1=6@33.998°`, `P2=5/w2=6@33.998°`, `sw_upright`, `(-1,-1)` | 5.72s |
| 842 | `29.520815` | `29.502153` | `29.518815` | `835` | **7** | `24x17`, `P1=5/w1=6@34.329°`, `P2=12/w2=13@20.628°`, `sw_upright`, `(-1,-1)` | 9.10s |
| 634 | `25.707107` | `25.682309` | `25.705107` | `631` | **3** | `19x19`, `P1=6/w1=7@26.645°`, `P2=6/w2=7@26.645°`, `sw_upright`, `(1,1)` | 7.59s |
| 908 | `30.634071` | `30.605885` | `30.632071` | `904` | **4** | `20x20`, `P1=10/w1=11@20.918°`, `P2=10/w2=11@20.918°`, `sw_upright`, `(1,1)` | 11.16s |
| 911 | `30.695622` | `30.660913` | `30.693622` | `907` | **4** | `20x21`, `P1=10/w1=11@19.693°`, `P2=9/w2=10@21.012°`, `sw_upright`, `(-1,1)` | 12.42s |
| 971 | `31.664318` | `31.636789` | `31.662318` | `967` | **4** | `21x21`, `P1=10/w1=11@20.328°`, `P2=10/w2=11@20.328°`, `sw_upright`, `(-1,-1)` | 12.43s |

- **Verdict — Strong Lead (0 certified wins)**:
  - Even *before* any continuous angle/position L-BFGS relaxation or end-taper surgery, rigid zero-sawtooth sheared pockets hit within **3–4 squares** of `L_ours - 0.002` on `n=598, 634, 908, 911, 971` (note `634` is currently `45_only` at `25.707107`, and pure rigid sheared pockets at `26.645°` immediately came within 3 squares!).
  - Notably, on $D=1$ roots (`401, 530, 842`), the deficit remained **7 squares** (`relaxed_L = null`), confirming mathematically that a **pure uniform sheared row loses too much at its two flat ends** unless paired with tapered end-caps (Rank 2) or asymmetric pinched/chiral cores (Ranks 4–5).

---

### POC 2: Off-$45^\circ$ Sturmian-Staircase Diagonal Band ([`poc/poc2_off45_sturmian_band.py`](poc/poc2_off45_sturmian_band.py) → [`docs/poc2_off45_sturmian_band.json`](docs/poc2_off45_sturmian_band.json))

| $n$ | $L_{\text{ours}}$ | $L_{\text{front}}$ | Rigid Cap (Deficit) | Best Rigid Config | Relax Cap | Runtime |
|---:|---:|---:|---:|---|---:|---:|
| 269 | `16.889730` | `16.889730` | `261` (`8`) | `44.5°, w=1, dv=0.0, corner_L0` | `261` (`pen=5.55e-16`) | 6.79s |
| 328 | `18.614369` | `18.599535` | `318` (`10`) | `44.5°, w=2, dv=0.0, corner_L0` | `0` (`cap < n-8`, skipped) | 14.80s |
| 367 | `19.692388` | `19.670857` | `357` (`10`) | `44.5°, w=2, dv=-1.5, corner_L0` | `0` (`cap < n-8`, skipped) | 17.62s |
| 492 | `22.707107` | `22.690540` | `480` (`12`) | `44.5°, w=2, dv=-1.5, corner_L0` | `0` (`cap < n-8`, skipped) | 21.37s |
| 634 | `25.707107` | `25.682309` | `620` (`14`) | `44.5°, w=5, dv=0.0, corner_L0` | `0` (`cap < n-8`, skipped) | 26.65s |
| 686 | `26.707107` | `26.686615` | `670` (`16`) | `44.5°, w=2, dv=-1.5, corner_L0` | `0` (`cap < n-8`, skipped) | 27.82s |

- **Verdict — Clean Negative Falsification (0 wins)**:
  - Even on calibration target `n=269` (whose incumbent actually uses `41.970°`), a single off-$45^\circ$ diagonal band against upright blocks left an `8`-square deficit after L-BFGS relaxation. On `328..686`, rigid capacities missed `n_target - 8` by `2..8` additional squares (`deficit = 10..16`), proving that off-$45^\circ$ diagonal bands suffer widening sawtooth mismatches against $0^\circ$ lattices that cannot match POC 1's wall-aligned sheared-row strategy.

---

### POC 3: Skyline Beam-Search DP & Motif Compaction in Strips ([`poc/poc3_skyline_dp.py`](poc/poc3_skyline_dp.py) → [`docs/poc3_skyline_dp.json`](docs/poc3_skyline_dp.json))

| Strip $W$ | $N$ | Upright $H_{\text{up}}$ ($\rho$) | Shear Baseline $\rho_{\text{shear}}$ ($w@\theta$) | Square-by-Square DP $H_{\text{DP}}$ ($\rho_{\text{tot}}$) | `uniform_shear` $H$ | `upright_bottom_taper` $H$ | `checker_12deg` $H$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `2.502` | 24 | `12` (`2.0`) | `2.000` (`w=2@0°`) | `13.0` (`1.846`) | — | — | `14.443` |
| `3.502` | 28 | `10` (`3.0`) | `3.000` (`w=3@0°`) | `10.0` (`2.800`) | — | — | `16.835` |
| `4.502` | 32 | `8` (`4.0`) | `4.000` (`w=4@0°`) | `9.085` (`3.522`) | `10.554` (`w=5`) | **`10.337`** (`w=5`) | `13.247` |
| `5.502` | 36 | `8` (`5.0`) | `5.000` (`w=5@0°`) | `8.0` (`4.500`) | `10.470` (`w=6`) | **`10.304`** (`w=6`) | — |
| `6.502` | 40 | `7` (`6.0`) | `6.000` (`w=6@0°`) | `8.0` (`5.000`) | `9.450` (`w=7`) | `9.850` (`w=7`) | — |
| `7.502` | 44 | `7` (`7.0`) | `7.023` (`w=8@28.611°`) | `7.0` (`6.286`) | `9.415` (`w=8`) | **`9.326`** (`w=8`) | — |
| `7.661` | 44 | `7` (`7.0`) | `7.234` (`w=8@25.278°`) | `7.0` (`6.286`) | `8.894` (`w=8`) | **`8.838`** (`w=8`) | — |

- **Verdict — Mixed Technical Findings (0 wins, 1 validated Rank-2 lead, 2 falsifications)**:
  1. **Validated Rank-2 Lead (`upright_bottom_taper`)**: Replacing the slanted bottom end of a sheared strip with a compacted upright transition row reduced strip height vs `uniform_shear` for `w = P+1` on 4 of the 5 widths above $W=4$ (`W=4.502`: `-0.217`, `5.502`: `-0.166`, `7.502`: `-0.089`, `7.661`: `-0.056`; on `W=6.502` (`w=7`), `9.850` lost to `uniform_shear` `9.450` while `w=8` improved `11.548` vs `11.874`).
  2. **Falsified Alternating Checkerboard (`checker_12deg`)**: Alternating $(+12^\circ, -12^\circ)$ squares inside a strip creates X-wedges and loses catastrophically (`H = 14.44` vs `12.0` upright at `W=2.502`; `16.83` vs `10.0` at `W=3.502`).
  3. **Falsified Fixed-Grid Square-by-Square Skyline Beam**: Even with angle-stratified beam buckets (`beam_width=24`), discrete $0.085$-grid placement collapsed to $0^\circ$ on 6/7 strips (`angs={0.0: N}`) because exact edge-to-edge sheared chains are measure-zero on a fixed grid.

---

## 5. Next Engineering Priorities (Not Built Here)

1. **Upgrade POC 1 + Rank 2 into an end-tapered L-pocket solver**: Relax the last 2–3 rows of each sheared pocket plus the SW miter triangle with **free angles + continuous L-BFGS + fixed-angle LP polish** on `n=598, 634, 908, 911, 971` (where rigid `deficit` is already `3..4`).
2. **Test Rank 5 ($C_4$ chiral pinwheel + 4-quadrant symmetric inflation)** specifically on $D=1$ roots (`401, 530, 626, 842, 962`) to test whether continuous chiral rotation explains the `0.535534 -> 0.502153` sequence.
3. **Test Rank 3 (De novo inflation from oriented priors)** on top losers `(911, 367, 910, 972, 791, 634)` using vectorized Numba SAT batches across all 8 M1 cores.

---

## Files in `poc/hydrogen_brainstorm/`
- [`README.md`](README.md) — This report.
- [`poc/geom_utils.py`](poc/geom_utils.py) — Shared Numba SAT penalty/gradient, L-BFGS, LP tighten, and official literal + pandas roundtrip verifier.
- [`poc/poc1_sheared_pockets.py`](poc/poc1_sheared_pockets.py) — POC 1 runner.
- [`poc/poc2_off45_sturmian_band.py`](poc/poc2_off45_sturmian_band.py) — POC 2 runner.
- [`poc/poc3_skyline_dp.py`](poc/poc3_skyline_dp.py) — POC 3 runner.
- [`docs/gap_summary.json`](docs/gap_summary.json) — Exact `merged.csv` vs `per_n_gap.csv` breakdown.
- [`docs/math_tables.json`](docs/math_tables.json) — $D=1$ sheared math check, Pythagorean triples, incumbent angle modes.
- [`docs/poc1_sheared_pockets.json`](docs/poc1_sheared_pockets.json), [`docs/poc2_off45_sturmian_band.json`](docs/poc2_off45_sturmian_band.json), [`docs/poc3_skyline_dp.json`](docs/poc3_skyline_dp.json) — Full POC run outputs.
- [`data/top_gap_targets.csv`](data/top_gap_targets.csv) — Small 20-row target summary table (top 15 positive gaps + 5 losing $D=1$ root samples).
