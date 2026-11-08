# Multi-Pell Cores POC 3: Negative/Near-Zero Slack Pell Cores ($D=7, 41$)

## Status: Falsified at the Gate

Investigates semi-convergent Pell cores where $\ell = w + \lceil D/\sqrt{2} \rceil$ introduces longitudinal interference along the diagonal band:
- **$D = 7$**: $7/\sqrt{2} = 4.949747 \implies \ell = w + 5$, longitudinal slack $\sigma = -0.050253$.
- **$D = 41$**: $41/\sqrt{2} = 28.991378 \implies \ell = w + 29$, longitudinal slack $\sigma = -0.008622$.

## Summary of Results

| Parameter | $D=7$ Seed | $D=41$ Seed | Reference $D=17$ (Pell-17) |
|---|---:|---:|---:|
| Width $w$ | 4 | 8 | 13 |
| Length $\ell$ | 9 | 37 | 25 |
| Corner staircase $b$ | 2 | 5 | 9 |
| Flank staircases $(a_1, a_2)$ | $(6, 5)$ | $(23, 23)$ | $(18, 17)$ |
| Count $n_0$ | 78 | 878 | 739 |
| Diagonal Slack $\sigma$ | **$-0.050253$** | **$-0.008622$** | **$+0.020815$** |
| Mean Column Pitch | $0.9948$ (< 1.0) | $0.9998$ (< 1.0) | $1.0009$ (> 1.0) |
| Nominal $L_{45}$ | $9.328427$ | $29.656854$ | $27.692388$ |
| HiGHS LP Status at $L_{45}+0.15$ | **Infeasible** | **Infeasible** | Optimal |
| Corrugation Gain $\Delta L$ | N/A ($+0.1815$ with dilation) | N/A (Jammed) | **$-0.001467$** |
| Gate Outcome | **FALSIFIED** | **FALSIFIED** | **CERTIFIED WIN** ($n=910$) |

## Why Negative Slack Fails

1. **Longitudinal Pitch Deficit**: Placing $\ell = w + \lceil D/\sqrt{2} \rceil$ columns between rigid corner staircases $b_1, b_2$ constrains the average column pitch $\Delta u$ along axis $(x+y)/\sqrt{2}$ to:
   $$\overline{\Delta u} = 1 + \frac{\sigma}{\ell - 1} < 1.0000$$
2. **SAT Incompatibility**: Adjacent unit squares in adjacent columns require a longitudinal separation of at least $1.0000$. With pitch $< 1.0$, adjacent columns overlap by order $-0.006$ to $-0.050$.
3. **Impotence of Transverse Tooth Penetration**: Transverse flank tooth penetration shifts column ends along $v$, but does not compress the longitudinal metric along $u$. Consequently, the 4-sign-piece epigraph LP cannot resolve the non-convex inter-column collision without expanding the container.
4. **No Shrinkage**: When container constraints are relaxed, $L$ expands above $L_{45}$ ($\Delta L = +0.1815$ for $D=7$), completely falsifying the hypothesis that negative-slack cores can achieve corrugation shrinkage.

## Directory Structure

- `poc/multi_pell_cores_3.py`: Implementation of parameter models, column pitch measurement, and falsification gate evaluation.
- `poc/test_multi_pell_cores_3.py`: Pytest suite verifying negative slack algebra, column pitch compression, LP infeasibility, and dilation behavior.
- `docs/negative_slack_pell_cores.md`: Comprehensive mathematical analysis, proofs, and comparative tables.
- `README.md`: This executive report.
