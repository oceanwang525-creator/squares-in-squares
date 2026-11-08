# Pell-17 Out-of-Phase Pinch-Column Corrugated Bands + L-Border, CPU POC

## 1. Construction Overview

This is a 100% synthetic CPU construction for large-$n$ packings, not an incumbent coordinate replay or frozen-atlas polish. Every emitted layout is generated from closed-form algebraic staircase-band parameters and certified by the unchanged official interval metric (`squares.core.certify`) on both literal CSV strings and actual `pandas.read_csv(...).to_csv(index=False)` round-trip transport under `pandas 3.1.0`.

### The In-Phase vs. Out-of-Phase Staircase-Band Theorem
In a staircase-band packing with flank upright staircases $(a_1, a_2)$ and end upright staircases $(b_1, b_2)$ around a $w$-strip diagonal band at $45^\circ$:
$$L_{45} = \frac{a_1 + a_2}{2} + 1 + \frac{w}{\sqrt{2}}, \qquad u_{\text{hi}} - u_{\text{lo}} = (w - 1) + \frac{D}{\sqrt{2}}, \quad D = (a_1 + a_2) - (b_1 + b_2).$$
Hence the column count per strip is $\ell = w + \lfloor D/\sqrt{2} \rfloor$, and the longitudinal slack at $45^\circ$ depends only on $D$:
$$\text{slack}(D) = \left\{\frac{D}{\sqrt{2}}\right\} = \frac{D}{\sqrt{2}} - \left\lfloor \frac{D}{\sqrt{2}} \right\rfloor.$$

- **Even $a_1 + a_2$ (In-Phase Staircases, e.g., $n = 909$ with $w = 8, a = (23, 23)$)**:
  Every column $c$ has identical tooth offsets $\Delta u_{\text{se}}(c) = \Delta u_{\text{nw}}(c)$ of the same sign. A uniform column tilt $\delta_c$ produces canceling first-order tooth gains ($+\Delta u\delta_c - \Delta u\delta_c = 0$), while opposite half-column tilts across the midline cost $+|\delta|$ in internal SAT separation and gain at most $-2|\Delta u(c)|\cdot|\delta| < |\delta|$ at pinch columns ($|\Delta u(c)| < 0.50$). Thus no first-order transverse shrinkage exists for in-phase staircases.
- **Odd $a_1 + a_2$ (Out-of-Phase Staircases, e.g., $n = 910$ with $w = 13, a = (18, 17)$)**:
  The SE and NW staircase teeth are out of phase by $\frac{1}{\sqrt{2}}$ along $u$. Every **pinch column** ($\max(|\Delta u_{\text{se}}(c)|, |\Delta u_{\text{nw}}(c)|) < 0.50$) has $\Delta u_{\text{se}}(c)$ and $\Delta u_{\text{nw}}(c)$ of **opposite signs** with $|\Delta u_{\text{se}}(c)| + |\Delta u_{\text{nw}}(c)| = \frac{1}{\sqrt{2}}$. Tilting column $c$ uniformly by $\delta_c = -\text{sign}(\Delta u_{\text{se}}(c))|\delta|$ causes both ends of column $c$ to penetrate their respective staircases simultaneously by $-\frac{|\delta|}{\sqrt{2}}$ with zero internal transverse expansion, shrinking the box side length by $\Delta L \approx -\frac{|\delta|}{2}$.

### The Pell-$17$ Corrugated Band Family ($D = 17, \ell = w + 12$)
Because $17^2 - 2\cdot 12^2 = +1$ (Pell equation), $D = 17$ gives $\frac{17}{\sqrt{2}} = 12 + 0.02081528$. Setting $b_1 = b_2 = b = \lfloor w/\sqrt{2} \rfloor$ and $a_1 = b + 9, a_2 = b + 8$ yields $a_1 + a_2 = 2b + 17$ (odd, out-of-phase) and $D = 17$. Whenever $\{w/\sqrt{2}\} \in (0.05, 0.42)$ — which holds for $w \in \{2, 3, 6, 9, 10, 13, 16\}$ — the $b$-staircases back the band ends with wall clearance, and our 4-rule **Pinch-Column Corrugation Engine** + **4-Sign-Piece Epigraph LP Polisher** shrinks every core by $-0.00139$ to $-0.00206$ below $L_{45}$:

| Width $w$ | $b$ | $(a_1, a_2)$ | $\ell = w + 12$ | $k_0$ | Core $n_0$ | $L_{45}$ | Corrugated $L_{\text{corr}}$ | $\Delta L = L_{\text{corr}} - L_{45}$ |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| $2$ | $1$ | $(10, 9)$ | $14$ | $11$ | $130$ | `11.914213562` | `11.912286146` | `-0.001927416` |
| $3$ | $2$ | $(11, 10)$ | $15$ | $13$ | $172$ | `13.621320344` | `13.619259872` | `-0.002060472` |
| $6$ | $4$ | $(13, 12)$ | $18$ | $17$ | $297$ | `17.742640687` | `17.741250940` | `-0.001389747` |
| $9$ | $6$ | $(15, 14)$ | $21$ | $21$ | $456$ | `21.863961031` | `21.862433095` | `-0.001527935` |
| $10$ | $7$ | $(16, 15)$ | $22$ | $23$ | $532$ | `23.571067812` | `23.569666051` | `-0.001401761` |
| $13$ | $9$ | $(18, 17)$ | $25$ | $27$ | $739$ | `27.692388155` | `27.690921182` | `-0.001466973` |
| $16$ | $11$ | $(20, 19)$ | $28$ | $31$ | $980$ | `31.813708499` | `31.812280644` | `-0.001427855` |

### Why 4-Sign-Piece Epigraph Support Beats `30.69096012` at $n = 910$
Wrapping the $(w = 13, b = 9, k_0 = 27, n_0 = 739)$ Pell-$17$ core with a $t = 3$ upright L-border ($171$ squares) produces $n = 910$. Linearizing the pairwise SAT support $\frac{1}{2}(|\cos r| + |\sin r|)$ with central finite differences stalls at the $r = \theta_j - \theta_i = 0$ kink (`L = 30.69096039`, matching the public incumbent `30.69096012`). Imposing all 4 smooth linear sign pieces $(s_1, s_2) \in \{-1, +1\}^2$ simultaneously in the LP epigraph allows adjacent columns in a tilted block to split relative angles across $r = 0$, converging to certified **`L(910) = 30.690921074188281`** (`delta_L = -0.000039046458` vs. the public incumbent `30.690960120646093`).

---

## 2. Certified Results Across All 13 Target $n$s

Elapsed CPU wall time was `237.700` seconds under `pandas 3.1.0`. Every emitted candidate passes unchanged `squares.core.certify` on both literal CSV and actual `pandas.read_csv(...).to_csv(index=False)` round-trip transport:

| $n$ | Construction | $(k_0, t, w, \ell, a_1, a_2, b_1, b_2)$ | Certified Literal $L$ | Certified Pandas $3.1.0$ $L$ | Portfolio Baseline $L$ | $\Delta L$ vs. Portfolio | Public Target Incumbent $L$ | $\Delta L$ vs. Public Incumbent |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| `543` | `staircase_band_l_border` | $(23, 0, 16, 16, 12, 11, 11, 12)$ | `23.813708544612` | `23.813708544612` | `24.000000000000` | **`-0.186291455`** | — | — |
| `544` | `staircase_band_l_border` (alt Pell-17: `23.862433008`) | $(23, 0, 4, 28, 20, 20, 3, 3)$ | `23.828427170403` | `23.828427170403` | `24.000000000000` | **`-0.171572830`** | — | — |
| `545` | `staircase_band_l_border` | $(20, 3, 14, 14, 10, 10, 10, 10)$ | `23.899494982411` | `23.899494982411` | `24.000000000000` | **`-0.100505018`** | — | — |
| `591` | `staircase_band_l_border` (alt Pell-17: `24.862432990`) | $(23, 1, 4, 28, 20, 20, 3, 3)$ | `24.828427172403` | `24.828427172403` | `25.000000000000` | **`-0.171572828`** | — | — |
| `592` | `staircase_band_l_border` | $(24, 0, 14, 21, 14, 14, 9, 9)$ | `24.899494984411` | `24.899494984411` | `25.000000000000` | **`-0.100505016`** | — | — |
| `593` | `staircase_band_l_border` | $(24, 0, 14, 21, 14, 14, 9, 9)$ | `24.899494984411` | `24.899494984411` | `25.000000000000` | **`-0.100505016`** | — | — |
| `908` | `staircase_band_l_border` | $(29, 1, 20, 20, 15, 14, 14, 15)$ | `30.642135683015` | `30.642135683015` | `30.707106781187` | **`-0.064971098`** | `30.637029559612` | `+0.005106123` |
| `909` | `staircase_band_l_border` | $(29, 1, 8, 32, 23, 23, 6, 6)$ | `30.656854308806` | `30.656854308806` | `30.707106781187` | **`-0.050252472`** | `30.656854249493` | `+5.931e-8` |
| **`910`** | **`pell17_corrugated_band_l_border`** | **$(27, 3, 13, 25, 18, 17, 9, 9)$** | **`30.690921074188`** | **`30.690921074188`** | `30.707106781187` | **`-0.016185707`** | **`30.690960120646`** | **`-0.000039046`** |
| `911` | `staircase_band_l_border` | $(30, 0, 1, 42, 29, 29, 0, 0)$ | `30.707106840601` | `30.707106840601` | `30.707106781187` | `+5.941e-8` | `30.707106781187` | `+5.941e-8` |
| `971` | `pell17_corrugated_band_l_border` | $(27, 4, 13, 25, 18, 17, 9, 9)$ | `31.690921202134` | `31.690921202134` | `31.707106781187` | **`-0.016185579`** | `31.678779875343` | `+0.012141327` |
| `972` | `staircase_band_l_border` | $(31, 0, 1, 43, 30, 30, 0, 0)$ | `31.707106842601` | `31.707106842601` | `31.707106781187` | `+6.141e-8` | `31.692388155428` | `+0.014718687` |
| `973` | `staircase_band_l_border` | $(31, 0, 1, 43, 30, 30, 0, 0)$ | `31.707106842601` | `31.707106842601` | `31.707106781187` | `+6.141e-8` | `31.707106781187` | `+6.141e-8` |

## 3. Reproducibility & Validation Gates

Run from the repository root:
```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=poc/baseline/poc/src python3 poc/pell_corrugation/poc/pell_corrugation.py
pytest -q poc/pell_corrugation/poc/test_pell_corrugation.py
```
All outputs (`output.csv`, `output-pandas.csv`, `results.json`) are saved in `poc/pell_corrugation/data/`.
