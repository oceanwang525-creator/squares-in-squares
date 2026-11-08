# Multi-Pell Corrugated Cores POC 1: Symmetric Odd-D Cores (D=3, 27, 51)

## 1. Overview and Theoretical Foundation

This POC implements the generalized out-of-phase pinch-column corrugation engine for all admissible symmetric odd-$D$ staircase-band configurations ($D=3, 27, 51$).

### Parity and Out-of-Phase Condition
In any staircase-band configuration with flank staircases $(a_1, a_2)$ and end staircases $(b_1, b_2)$:
$$D = (a_1 + a_2) - (b_1 + b_2).$$
When end staircases are symmetric ($b_1 = b_2 = b$):
$$a_1 + a_2 = 2b + D.$$
Because $2b$ is even, $a_1 + a_2$ is odd if and only if **$D$ is odd**.

Setting $a_1 = b + \frac{D+1}{2}$ and $a_2 = b + \frac{D-1}{2}$ produces:
- Flank step difference: $a_1 - a_2 = 1$ (near-symmetric flank staircases).
- Flank tooth relative phase:
  $$\Delta u_{\text{phase}} \equiv \frac{a_1 + a_2 - 2}{\sqrt{2}} \equiv \frac{1}{\sqrt{2}} \pmod{\sqrt{2}}.$$
- Opposing pinch-column tooth offsets:
  $$|\Delta u_{\text{se}}(c)| + |\Delta u_{\text{nw}}(c)| = \frac{1}{\sqrt{2}} \approx 0.7071068.$$

### Longitudinal Slack and Strip Length
The longitudinal column span along the $u$-diagonal is:
$$\ell = w + \left\lfloor \frac{D}{\sqrt{2}} \right\rfloor.$$
The remaining longitudinal slack is:
$$\text{slack}(D) = \left\{\frac{D}{\sqrt{2}}\right\} = \frac{D}{\sqrt{2}} - \left\lfloor \frac{D}{\sqrt{2}} \right\rfloor.$$
- For $D = 3$: $\lfloor 3/\sqrt{2} \rfloor = 2$, $\text{slack} = 0.12132$, $\ell = w + 2$.
- For $D = 27$: $\lfloor 27/\sqrt{2} \rfloor = 19$, $\text{slack} = 0.09188$, $\ell = w + 19$.
- For $D = 51$: $\lfloor 51/\sqrt{2} \rfloor = 36$, $\text{slack} = 0.06245$, $\ell = w + 36$.

All three families satisfy the admissible slack criterion $0.010 \le \text{slack}(D) \le 0.150$.

---

## 2. Complete Family Parametrizations ($n_0 \le 1000$)

### Family 1: Diophantine $D = 3$ ($\text{slack} = 0.12132$, $\ell = w + 2$)
$$b_1 = b_2 = b = \lfloor w/\sqrt{2} \rfloor, \quad a_1 = b + 2, \quad a_2 = b + 1$$
$$n_0 = \frac{(b+2)(b+3)}{2} + \frac{(b+1)(b+2)}{2} + 2\frac{b(b+1)}{2} + w(w+2)$$

| Width $w$ | $b$ | Flank $(a_1, a_2)$ | Strip $\ell$ | $k_0$ | Core $n_0$ | $L_{45}$ | $\{w/\sqrt{2}\}$ |
|---:|---:|:---:|---:|---:|---:|---:|---:|
| 2 | 1 | (3, 2) | 4 | 4 | 19 | 4.914213562 | 0.4142 |
| 3 | 2 | (4, 3) | 5 | 6 | 37 | 6.621320344 | 0.1213 |
| 6 | 4 | (6, 5) | 8 | 10 | 104 | 10.742640687 | 0.2426 |
| 9 | 6 | (8, 7) | 11 | 14 | 205 | 14.863961031 | 0.3640 |
| 10 | 7 | (9, 8) | 12 | 16 | 257 | 16.571067812 | 0.0711 |
| 13 | 9 | (11, 10) | 15 | 20 | 406 | 20.692388155 | 0.1924 |
| 16 | 11 | (13, 12) | 18 | 24 | 589 | 24.813708499 | 0.3137 |
| **20** | 14 | (16, 15) | 22 | 30 | **906** | **30.642135624** | 0.1421 |

### Family 2: Hybrid $D = 27$ ($\text{slack} = 0.09188$, $\ell = w + 19$)
$$b_1 = b_2 = b = \lfloor w/\sqrt{2} \rfloor, \quad a_1 = b + 14, \quad a_2 = b + 13$$
$$n_0 = \frac{(b+14)(b+15)}{2} + \frac{(b+13)(b+14)}{2} + 2\frac{b(b+1)}{2} + w(w+19)$$

| Width $w$ | $b$ | Flank $(a_1, a_2)$ | Strip $\ell$ | $k_0$ | Core $n_0$ | $L_{45}$ | $\{w/\sqrt{2}\}$ |
|---:|---:|:---:|---:|---:|---:|---:|---:|
| 2 | 1 | (15, 14) | 21 | 16 | 269 | 16.914213562 | 0.4142 |
| 3 | 2 | (16, 15) | 22 | 18 | 328 | 18.621320344 | 0.1213 |
| 6 | 4 | (18, 17) | 25 | 22 | 494 | 22.742640687 | 0.2426 |
| 9 | 6 | (20, 19) | 28 | 26 | 694 | 26.863961031 | 0.3640 |
| 10 | 7 | (21, 20) | 29 | 28 | 787 | 28.571067812 | 0.0711 |

### Family 3: Triple-Pell $D = 51$ ($\text{slack} = 0.06245$, $\ell = w + 36$)
$$b_1 = b_2 = b = \lfloor w/\sqrt{2} \rfloor, \quad a_1 = b + 26, \quad a_2 = b + 25$$
$$n_0 = \frac{(b+26)(b+27)}{2} + \frac{(b+25)(b+26)}{2} + 2\frac{b(b+1)}{2} + w(w+36)$$

| Width $w$ | $b$ | Flank $(a_1, a_2)$ | Strip $\ell$ | $k_0$ | Core $n_0$ | $L_{45}$ | $\{w/\sqrt{2}\}$ |
|---:|---:|:---:|---:|---:|---:|---:|---:|
| 2 | 1 | (27, 26) | 38 | 28 | 807 | 28.914213562 | 0.4142 |
| **3** | 2 | (28, 27) | 39 | 30 | **907** | **30.621320344** | 0.1213 |

---

## 3. High-Value Discoveries for Competition Targets

1. **Sub-Incumbent Uncorrugated Seed at $n = 907$**:
   The $D=51, w=3$ core has:
   $$L_{45} = 28.5 + \frac{3}{\sqrt{2}} \approx 30.621320344.$$
   The official public incumbent for $n = 908$ is $L(908) = 30.6370295596$.
   Thus, uncorrugated $L_{45}(907)$ is already $-0.015709$ smaller than the $n = 908$ public record! Corrugation shrinkage further widens this margin.

2. **Direct Adjacent Core at $n = 906$**:
   The $D=3, w=20$ core yields $n_0 = 906$ with $L_{45} = 30.642135624$, directly adjacent to the high-deficit target interval $908-911$.

3. **Multi-Block Composition**:
   Wrapping $D=27, w=6$ ($n_0 = 494$) with an upright L-border of thickness $t = 8$ produces:
   $$n = 494 + 2(22)(8) + 8^2 = 494 + 352 + 64 = 910.$$
   This offers an alternative structural path to $n = 910$ that does not share the same fault-line coordinates as Pell-17.
