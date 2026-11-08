# Multi-Pell Corrugated Cores: POC 1 (Symmetric Odd-D Cores)

Extension of out-of-phase pinch-column corrugation to the full spectrum of symmetric odd-$D$ staircase-band configurations:
- **Diophantine $D = 3$**: $\ell = w + 2$, $\text{slack} = 0.12132$ ($3^2 - 2\cdot 2^2 = +1$).
- **Hybrid $D = 27$**: $\ell = w + 19$, $\text{slack} = 0.09188$.
- **Triple-Pell $D = 51$**: $\ell = w + 36$, $\text{slack} = 0.06245$ ($3 \times 17$).

## Algebraic Properties of Symmetric Odd-$D$ Families

For any symmetric odd $D$ and strip width $w$ with end staircases $b_1 = b_2 = b = \lfloor w / \sqrt{2} \rfloor$:
1. **Flank Staircase Parity**:
   $$a_1 = b + \frac{D + 1}{2}, \quad a_2 = b + \frac{D - 1}{2} \implies a_1 + a_2 = 2b + D \equiv 1 \pmod 2.$$
   Because $a_1 + a_2$ is odd, SE and NW staircase teeth are out of phase by exactly $\frac{1}{\sqrt{2}}$.
2. **First-Order Transverse Shrinkage**:
   Tilting pinch columns ($\max(|\Delta u_{\text{se}}|, |\Delta u_{\text{nw}}|) < 0.50$) uniformly by $\delta_c = -\text{sign}(\Delta u_{\text{se}})|\delta|$ penetrates both flank staircases simultaneously with zero internal SAT separation cost, yielding:
   $$\Delta L \approx -\frac{|\delta|}{2}.$$
3. **Longitudinal Slack Budget**:
   The available slack $\text{slack}(D) = \{D/\sqrt{2}\}$ satisfies $0.060 \le \text{slack} \le 0.122$, easily absorbing the longitudinal expansion $K|\delta| \approx 0.015 - 0.025$ without jamming the end staircases.
4. **Wall Clearance Rule**:
   Clean separation from container walls requires $\{w/\sqrt{2}\} \in (0.05, 0.42)$.

## Sweet Spot Tabulation ($n_0 \le 1000$)

| Family | Width $w$ | $b$ | $(a_1, a_2)$ | Strip $\ell$ | $k_0$ | Core $n_0$ | $L_{45}$ | Slack | $\{w/\sqrt{2}\}$ |
|:---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| **$D=3$** | 2 | 1 | (3, 2) | 4 | 4 | 19 | 4.914213562 | 0.12132 | 0.4142 |
| | 3 | 2 | (4, 3) | 5 | 6 | 37 | 6.621320344 | 0.12132 | 0.1213 |
| | 6 | 4 | (6, 5) | 8 | 10 | 104 | 10.742640687 | 0.12132 | 0.2426 |
| | 9 | 6 | (8, 7) | 11 | 14 | 205 | 14.863961031 | 0.12132 | 0.3640 |
| | 10 | 7 | (9, 8) | 12 | 16 | 257 | 16.571067812 | 0.12132 | 0.0711 |
| | 13 | 9 | (11, 10) | 15 | 20 | 406 | 20.692388155 | 0.12132 | 0.1924 |
| | 16 | 11 | (13, 12) | 18 | 24 | 589 | 24.813708499 | 0.12132 | 0.3137 |
| | **20** | 14 | (16, 15) | 22 | 30 | **906** | **30.642135624** | 0.12132 | 0.1421 |
| **$D=27$** | 2 | 1 | (15, 14) | 21 | 16 | 269 | 16.914213562 | 0.09188 | 0.4142 |
| | 3 | 2 | (16, 15) | 22 | 18 | 328 | 18.621320344 | 0.09188 | 0.1213 |
| | 6 | 4 | (18, 17) | 25 | 22 | 494 | 22.742640687 | 0.09188 | 0.2426 |
| | 9 | 6 | (20, 19) | 28 | 26 | 694 | 26.863961031 | 0.09188 | 0.3640 |
| | 10 | 7 | (21, 20) | 29 | 28 | 787 | 28.571067812 | 0.09188 | 0.0711 |
| **$D=51$** | 2 | 1 | (27, 26) | 38 | 28 | 807 | 28.914213562 | 0.06245 | 0.4142 |
| | **3** | 2 | (28, 27) | 39 | 30 | **907** | **30.621320344** | 0.06245 | 0.1213 |

### High-Value Core Findings
- **$D=51, w=3$ Core ($n_0 = 907$)**:
  The uncorrugated algebraic side length $L_{45} = 30.621320344$ is strictly smaller than the public incumbent for $n=908$ ($L = 30.6370295596$).
- **$D=3, w=20$ Core ($n_0 = 906$)**:
  Yields $L_{45} = 30.642135624$, directly adjacent to the high-deficit target cluster $908-911$.
- **$D=27, w=6$ Wrapped Core ($n_0 = 494, t=8 \implies n = 910$)**:
  Provides an independent alternative composite route to $n = 910$.

## Usage

```sh
# Print full sweet-spot table
python3 poc/multi_pell_cores/poc/multi_pell_cores.py --all-sweet-spots

# Generate and polish a specific core (e.g. D=51, w=3)
python3 poc/multi_pell_cores/poc/multi_pell_cores.py --D 51 --w 3 --iterations 35

# Run unit tests
pytest -q poc/multi_pell_cores/poc/test_multi_pell_cores.py
```
