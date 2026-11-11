# family_reverse_fit: root-plus-border reverse fit of non-(K^2+4, 10, 11) families. Result: negative, nothing built

Estimated gain: 0. No packing beats ours; no CSV written, merged.csv untouched, no `squares score` run (nothing feasible to score).
Script: `poc/relax442.py` (incumbent n -> shrink to leader L, penalty L-BFGS from `k2p10_fill2/poc/a_overlap.py`, 3 rounds, jitter restarts). Data: `leader_curve_analysis/docs/per_n_comparison.csv`.

## Gap mass by family (403 losing n, total sum(gap/sqrt n) = 0.1364)
| family | gap mass | n lost | max K | note |
|---|---|---|---|---|
| k(k+1)-n = 9 (cuts across D=K-9..) | 0.0095 | 17 | 31 | overlaps D families, not independent |
| k(k+1)-n = 8 / 10 | 0.0078 / 0.0075 | 18 / 15 | 31 | same |
| D=9 | 0.0064 | 14 | 31 | cousin of D=10/11 (45deg band + 1/K law) |
| D=7 / D=6 / D=5 / D=8 | 0.0056 / 0.0048 / 0.0044 / 0.0043 | 14/14/13/14 | 31 | band family, same geometry as D=10 |
| D=1 | 0.0052 | 12 | 31 | clustered, 7 distinct roots |
| D=4 | 0.0039 | 12 | 31 | |
| D=3 / D=2 | 0.0024 / 0.0016 | 11 / 6 | 31 | |

## D=1 fit (leader c = L-K; ours = 0.535534 = 5/sqrt2 - 3 until K=24, then 0.520815 for K>=25)
Leader c by K: 20 .526143 | 21,22 .524648 | 23 .521059 | 24 .520682 | 25-28 .513435 | 29 .509892 | 30,31 .506591. Leader equals ours (0.535534) for every K<=19 (n=65..362), so there is no small root: the family only departs at n=401.
Deficit from 0.535534: 0.0094 (K20), 0.0109 (21), 0.0145 (23), 0.0149 (24), 0.0221 (25), 0.0257 (29), 0.0289 (30) -> roughly linear, ~-0.0027 per K beyond K~19.5. Distinct roots (the rest are upright L-borders, t=1..3, 0 extra squares): 401, 442(->485), 530, 577, 626(->677,730,785), 842, 901(->962).
Guess: the 45deg-band-free "K^2+1" 0.5355 core (L = K-3+5/sqrt2, a sliding chain) gains a K-proportional saving once the chain is long enough (K>=20), i.e. a tilted/partially relaxed chain whose gain grows ~0.0027*K. Our own jump at K=25 (0.5208) is a different core, whose gap to the leader is the constant 0.0074 for 4 n.
D=2: leader c .5305 (K26-28) falling to .5226 (K31); ours stock .5355; same structure, roots start at K=26 (n=678). D=3: 12/17-type constants .565 (444,487) and .590 (259 + 5 borders): roots 259/228/292 are K=15-17, n=228..292.

## Root cost ranking (root n, how far from a directly searchable n<=120)
| family / cluster | root n | members gained | gap mass of cluster | difficulty |
|---|---|---|---|---|
| 0.590016 (D=3..6, 259) | 259 | 6 | ~0.001 | lowest root n but gap only 0.0014, ours already within 1e-3 |
| 0.596043 (228, D=3) | 228 | 2 | ~0.0007 | lowest n among lost clusters with real gap (0.0086) |
| 0.724373 (202) | 202 | 2 | ~0.0013 | |
| 0.599535 (328,365) | 328 | 2 | ~0.0017 | |
| 0.524648 (442,485) | 442 | 2 | 0.0109/sqrt442*2 = 0.0010 | D=1 chain, n=442 |
| 0.513435 (626..785) | 626 | 4 | 0.0074*(1/25+1/26+1/27+1/28)=0.0011 | D=1 |
| 0.564693 / 0.565624 (444, 680) | 444 / 680 | 2 / 2 | 0.0007 / 0.0005 | |
| 0.506591 (901,962) | 901 | 2 | 0.0009 | |

No lost cluster has root n <= 120 (smallest lost-cluster root is 202 for gap-bearing clusters; cluster 28 root n=70 has gap 4e-4 only). So nothing is reachable by the repo's exact-packing methods. By (mass/difficulty) the D=1 chains (442, 626) and 228 are the best, all at n >= 228; mass is spread (every cluster ~0.001), so a root solution is worth ~1e-3 at best.

## Attempt (measured)
Shrink-and-relax from the incumbent (300-1300 variables, 4-6 jittered restarts each, penalty L-BFGS, `poc/relax442.py`). Feasible means max overlap/wall violation < 1e-9.
| n | target L (leader) | our L | best residual violation | result |
|---|---|---|---|---|
| 442 | 21.524648 | 21.535534 | 8.5e-4 | infeasible |
| 626 | 25.513435 | 25.520815 | 6.2e-4 | infeasible |
| 401 | 20.526143 | 20.535534 | 9.4e-4 | infeasible |
Measured L achieved by us: none below incumbent (incumbent L unchanged). Leader L is a published value, not measured by us. A 45deg/Pell structured search was not run: with load ~135 and the earlier negatives (leader_poc_k2p4, k2p10_curve_geometry: free-angle repolish rigid, rigid bands cost ~0.15) a local relax is the only affordable check, and it shows the incumbent cannot absorb a 0.007-0.011 squeeze, so the leader's roots are structurally different packings.

## Verdict
Negative. The cheapest-to-reconstruct root is not cheap: the D=1 family's first departure is n=401 (K=20) and every lost-cluster root is n >= 202. Next step would need a global search at n~230-450 for a different core (K-proportional saving, ~0.0027*K for D=1), not a repolish of ours.
