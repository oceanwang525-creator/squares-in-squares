# k2p10_curve_geometry: what the leader's K^2+D (D=8..13) curve implies. Result: negative, no packing built

No packing was built or certified, no CSV was written, merged.csv is untouched. Estimated score gain from this POC: 0.
Data: `../leader_curve_analysis/data/perpuzzle_best.csv` (50-digit L, Oct 29 snapshot), `../leader_curve_analysis/docs/per_n_comparison.csv`, incumbent `../pell_corrugation/data/border-scan/merged.csv`.
Scripts in `poc/`, full per-D tables (L-K and waste W=L^2-n for ours and leader, K=10..31) in `docs/D_tables.txt`.

## Fitted law (leader, D=8..13, K>=22)
- Waste is linear in K: W = w1*K + w0. D=10: leader W = 0.80K + 5.3, ours 1.04K - 0.2. D=8..13 leader slopes 0.76-0.89 (ours 0.90-1.04). So L-K ~ a + b/K with a = w1/2 ~ 0.38-0.44, b = (w0+D)/2 ~ 7-8.
  Fits (K>=24, rms 0.003-0.005, a+b/K, a+b/sqrt(K), a+bK are indistinguishable): D=10 leader a=0.41,b=7.3; D=11 a=0.37,b=8.8; D=9 a=0.38,b=7.7; ours a~0.41-0.55 (plateau-dominated).
- Our curve is a staircase of stock constants (0.70711 = a+1+1/sqrt2 for D=8 K=19-22, D=9 K=22-25, D=10 K=24-26, D=11 K=25-26 at 0.72792); the 45deg band holds one extra square per ~2.5 K at fixed L-K, which is why D of the plateau grows with K.
  The leader departs from the 0.70711 plateau as soon as it is first reached (K=25 for D=10: 0.69874, then -0.01/K).
- Implied waste per unit diagonal band length: ours ~1.0/(sqrt2 K) * K ~ 0.7 (two 45deg sawtooth edges, 0.354 each); leader ~0.57. About 20% less edge waste per unit band length, or a band ~20% shorter. Both are consistent with a partially filled/corrugated 45deg edge (Pell-17 style) rather than a different angle.
- Leader constants are not closed forms. 537 and 739 share 0.68414360 (differ 9e-11), 908 and 969 share 0.60588451; 854 is 12/17 + 7e-11. None of the D=8..13 values (K>=17) is a root of a low-height quadratic (search over (m + a*sqrt r)/c, r<=197, c<=60, tol 3e-9) except stock 45deg (0.70711) and 854 (rational 12/17). So most leader values are numerically optimised, non-rigid; the K-dependence is the 1/K law above, not integer jumps of one formula.

## Hypotheses tested
- Rational-slope single band (slope p/q, tilted unit squares aligned to the band, width W rows) between two upright staircases: L = 2 + (T1+T2+W*sqrt(p^2+q^2))/(p+q); exact lattice count (`poc/band_count.py`, `scan3.py`; sanity: reproduces 635 at 45deg T1=T2=23). For K=25 target 635 with L<25.7072, p,q<14..40, W<=6: no template reaches 635 except the incumbent 45deg one. So no pure rational-angle rigid band explains 0.69874.
- Template sweep for D=10, K=14..26 (`poc/scan4.py`): the estimator is worse than ours for K<=24 (0.83-0.91 vs ours 0.74-0.86), i.e. the incumbent already uses richer structure (corrugation, L borders); the estimator is not a faithful model of the leader and only excludes the pure rigid-band explanation.
- Existing negatives still apply: free-angle repolish of the 45deg w=1 chain is first-order rigid (leader_poc_k2p10), and rigid-phase multi-angle bands cost ~0.15 in L (multi_angle_band).

## Inferred geometry (not built)
A 45deg band whose flank sawteeth are partially filled/relaxed in a K-dependent way (more band periods gives more savings), with ~0.57 instead of ~0.7 waste per unit band length. Gain at n=635 would be 0.0084 in L (~3.3e-4 in score-ratio units); K=26..31 gaps 0.02-0.04 each. Not achievable here without a per-n optimiser over the band interior; the generic lattice MILP is too rigid.

Targets (not met): n=635 L<25.7071 (leader 25.69874); n=686 26.68672 (ours 26.70711); n=687 26.70641 (ours 26.72792). Measured L: none.
