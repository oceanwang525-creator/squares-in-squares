# Measured results

Input was the upstream `b57e78b` incumbent, SHA-256 `1f9c0fd3b7a88e5a0f6c44b567ff2ef354872812c6b0ad8e37807bd1a64e965c`. The single-process seed sweep tested a 23-by-23 cut grid with three settings for copy overlap and shift clearance. The historical controls each added the published cut coordinates as one extra cut pair. Each family optimized its ten best seed candidates, using up to 480 L-BFGS-B iterations at the target side minus `0.00005`, then 18 SLP rounds and one fixed-angle polish.

| Donor → target | Incumbent L | Cut variants | Unique target seeds | Trials | Smallest trial L | SAT gap at smallest L | Smallest trial L with nonnegative float gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 157 → 184 | 14 | 1,587 | 363 | 10 | 14.023736960 | −1.9820e−3 | — |
| 183 → 212 | 15 | 1,587 | 528 | 10 | 15.023893844 | −1.4450e−4 | 15.026671250 |
| 211 → 242 | 16 | 1,587 | 576 | 10 | 16.036978130 | −1.1898e−3 | 16.048585242 |
| 109 → 132 | 11.976048384 | 1,590 | 132 | 10 | 12.003704953 | −5.0624e−3 | — |
| 132 → 157 | 12.993302611 | 1,590 | 223 | 10 | 13.023469145 | −2.5286e−3 | — |

The smallest trial side in each family exceeded its incumbent and still had a collision. All 50 trials missed the side prefilter by more than its `1e-9` margin, so the official candidate evaluator was not reached. The historical controls confirm that layer insertion attained the right counts. The SciPy optimizer did not recover the known control sides. [Per-trial scan](../data/scan.json)

## Nonlinear continuation at n=184

CasADi 3.8.2 supplied IPOPT. Two reconstructed exact-count seeds each had two solves, capped at 180 iterations and 12 CPU seconds per solve. Each model assigned one separating axis per nearby pair, enforced the four corner inequalities, and refreshed that assignment before the next solve.

| Seed cuts (x, y) | Round | IPOPT status | Iterations | Bounding L | Minimum SAT gap |
| --- | ---: | --- | ---: | ---: | ---: |
| 3.062419, 3.553023 | 1 | Solve_Succeeded | 52 | 14.043857982 | −1.6322e−10 |
| 3.062419, 3.553023 | 2 | Maximum_CpuTime_Exceeded | 76 | 14.043466917 | −8.4108e−6 |
| 3.062419, 4.534233 | 1 | Solve_Succeeded | 47 | 14.043539848 | −1.5642e−10 |
| 3.062419, 4.534233 | 2 | Solve_Succeeded | 81 | 14.043394355 | −1.5709e−10 |

The completed nonlinear solves reduced penetration to near floating-point tolerance at sides above 14. The result covers the selected seed layouts and solve budgets. No candidate CSV was produced. [IPOPT receipt](../data/ipopt.json)
