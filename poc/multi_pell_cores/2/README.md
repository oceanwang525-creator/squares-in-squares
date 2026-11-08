# POC 2: Asymmetric Even-D Corrugated Cores ($D = 10, 34, 58$)

Synthetic out-of-phase pinch-column corrugated staircase-band generator with asymmetric end staircases ($b_1 = b, b_2 = b + 1$), exact Diophantine geometry, and parametric upright L-border wrapping.

## Mechanism & Geometry

- **Parity Invariant**:
  With asymmetric ends ($b_1 = b, b_2 = b + 1$), $b_1 + b_2 = 2b + 1$ is odd.
  The flank sum $a_1 + a_2 = D + (b_1 + b_2) = D + 2b + 1$ is odd if and only if **$D$ is even**.
  This guarantees out-of-phase tooth parity:
  $$\Delta u_{\text{phase}} \equiv \frac{1}{\sqrt{2}} \pmod{\sqrt{2}}.$$
- **Longitudinal Span & Slack**:
  $$\ell = w + \left\lfloor \frac{D}{\sqrt{2}} \right\rfloor, \qquad \text{slack}(D) = \left\{\frac{D}{\sqrt{2}}\right\}.$$
  - $D = 10$: $\ell = w + 7$, $\text{slack} \approx 0.071068$ ($10^2 - 2 \cdot 7^2 = +2$)
  - $D = 34$: $\ell = w + 24$, $\text{slack} \approx 0.041631$ (Double Pell $2 \times 17$)
  - $D = 58$: $\ell = w + 41$, $\text{slack} \approx 0.012193$ ($58^2 - 2 \cdot 41^2 = +2$)
- **Asymmetric Wall Clearance**:
  Right container wall gap satisfies $\text{gap}_{\text{right}} = \{w/\sqrt{2}\} - 0.5$.
  Non-collision requires $\{w/\sqrt{2}\} > 0.50$ (practically $\{w/\sqrt{2}\} \in (0.55, 0.95)$), unlocking widths:
  $$w \in \{1, 4, 7, 8, 11, 14, 15, 18, 21, 22, 25\}.$$
  This directly complements symmetric odd-$D$ cores ($D = 17, 3, 27$) which require $\{w/\sqrt{2}\} \in (0.05, 0.42)$.

## Target Cores

| $D$ | Width $w$ | End $(b_1, b_2)$ | Flank $(a_1, a_2)$ | Strip $\ell$ | Core $n_0$ | $k_0$ | $L_{45}$ | Wrap $t$ | Wrapped $n$ |
|---|---|---|---|---|---|---|---|---|---|
| $10$ | $4$ | `(2, 3)` | `(8, 7)` | $11$ | $117$ | $11$ | `11.328427` | $0$ | $117$ |
| $10$ | $11$ | `(7, 8)` | `(13, 12)` | $18$ | $431$ | $21$ | `21.278175` | $8$ | $831$ |
| $34$ | $4$ | `(2, 3)` | `(20, 19)` | $28$ | $521$ | $23$ | `23.328427` | $0$ | $521$ |
| $34$ | $8$ | `(5, 6)` | `(23, 22)` | $32$ | $821$ | $29$ | `29.156854` | $1$ | $880$ |
| $58$ | $1$ | `(0, 1)` | `(30, 29)` | $42$ | $943$ | $31$ | `31.207107` | $0$ | $943$ |

## Incumbent Comparison Ledger (merged.csv)

| Target n | Candidate L | Our Incumbent L (merged.csv) | Delta L | Gate Status |
|:---|---:|---:|---:|:---|
| 117 | 11.32842700 | 11.00000000 | +0.32842700 | Falsified |
| 431 | 21.27817500 | 21.00000000 | +0.27817500 | Falsified |
| 831 | 29.27817500 | 29.00000000 | +0.27817500 | Falsified |
| 521 | 23.32842700 | 23.00000000 | +0.32842700 | Falsified |
| 821 | 29.15685400 | 29.00000000 | +0.15685400 | Falsified |
| 880 | 30.15685400 | 30.00000000 | +0.15685400 | Falsified |
| 943 | 31.20710700 | 31.00000000 | +0.20710700 | Falsified |

## Usage

```sh
# Run specific asymmetric core generator
python3 poc/multi_pell_cores/2/poc/asymmetric_even_d_cores.py --d 10 --w 4

# Run all target evaluations
python3 poc/multi_pell_cores/2/poc/asymmetric_even_d_cores.py --all

# Run pytest test suite
pytest poc/multi_pell_cores/2/poc/test_asymmetric_even_d_cores.py
```

Detailed derivations, proofs, and limits are in [docs/asymmetric_even_d_cores.md](docs/asymmetric_even_d_cores.md).
