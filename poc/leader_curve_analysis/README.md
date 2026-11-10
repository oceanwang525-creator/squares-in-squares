# Leader curve analysis (Cats and boxes vs incumbent)

Data: `data/perpuzzle_best.csv` (snapshot, read-only). Our L from `poc/pell_corrugation/data/border-scan/merged.csv`
(exact rotated bounding box). Score = sum(L/sqrt n) - 1000 (reproduces 22.2038887). Gap mass = sum (L_ours - L_lb)/sqrt n.
Scripts: `poc/compare.py` (per-n csv), `analyze.py`, `families.py`, `waste.py`. Outputs in `docs/`
(`per_n_comparison.csv`, `by_D.csv`, `by_K.csv`, `by_offset.csv`, `top_gaps.csv`, `D_tables.txt`, `gap_fits.csv`, `waste_exponents.csv`).

## Totals
| team | n beating us | gap mass (score) |
|---|---|---|
| Cats and boxes | 349 | 0.1208 |
| Chris Deotte | 16 | 0.0052 |
| Achal Jain | 12 | 0.0033 |
| all others (11 teams) | 26 | 0.0071 |
| total | 403 | 0.1364 (0.61% of our score) |

## Findings
- No gap for K<=11. Cats first wins at K=12 and the gap grows steadily: mean (L_ours - L_lb) is ~0.004 at K=16, 0.008 at K=24, 0.012-0.013 at K=30-31
  (gap/(L-K) rises 0.5% -> 1.8%). Gap in wasted area W=L^2-n (=2K*gap) rises ~0.14 (K=16) -> ~0.8 (K=31).
  Per-K mass: K=28..31 each ~0.0125-0.0156, i.e. ~45% of everything sits at K>=28 (n>=784).
- Their L-K is not a family of shared constants: 328 distinct L-K values over 349 n (only 13 values reused; 6 coincide with values in our set).
  So most of their wins look like per-n bespoke optimisation of stock-like templates, not one closed form.
- Wasted area W=L^2-n scales ~K^1.0 for BOTH curves (global fit: ours K^1.014, theirs K^0.990; per-D with D<=15: ours ~0.9-1.0, theirs ~0.87-0.97).
  Erdos-Graham gives W=O(L^(7/11)=L^0.636), Roth-Vaughan lower bound ~ L^(1/2)-ish; neither of us (nor they) shows that decay at K<=31,
  so n<=1000 is in the pre-asymptotic, template regime. They are ~1 unit of W better at K=30 (D<=15: W 29.2-29.7 vs our 30.3-31.2) with a slightly smaller exponent.
- Per-D fits of L-K vs K: a+b/K, a+b/sqrt K, a+b/K^(4/11), a+b/K^(1/4) all fit with rms 0.012-0.014 (ours and theirs); the plateau/step structure
  dominates, so form cannot be discriminated. Stock steps in our data (0.5355 = 5*sqrt2/2-3, 0.70711, 0.72792, 0.63703) repeat over many K with no decay; theirs fall below them.
- k^2+1 plateau (ours 0.5355 for K<=24, 0.5208 for K>=25): Cats matches 0.5355 up to K=19, then K-dependent:
  K=20 0.52614, 21/22 0.52465, 23 0.52106, 24 0.52068, 25-28 0.51343, 29 0.50989, 30/31 0.50659. They keep stepping down with K where we stay flat
  (gap 0.007-0.015 at K=20..31, mass 0.0052). k^2+2: ours 0.5355 flat K=17..31; theirs breaks it at K=26 (0.53054) down to 0.52262 at K=31 (mass 0.0016).
- Stock-constant n (ours at 0.707107 / 0.727922 / 0.637029): 28 / 6 / 8 n, of which 10 / 5 / 8 are beaten (mass 0.0053 / 0.0027 / 0.0034).
  Their L-K there is 0.686-0.702 (45deg case), 0.706-0.725 (0.7279 case), 0.606-0.636 (0.637 case) and decreases with K (e.g. 0.637029 case: K=22 0.6362 -> K=30 0.6059).
  So stock 45deg constructions have no K-decay; theirs add K-dependence.
- By structure: D in 9..13 (n=K^2+9..13) carries 0.0293 of Cats mass (65 n); offset k(k+1)-n in 6..10 carries 0.0356 (76 n); both overlap.

## Recommended families
**A. n = K^2+10, K^2+11 with K=24..31 (decaying curve, biggest per-n gaps).** n = 586..971/972. Our L-K is stuck on the 45deg stock 0.70711 (K<=26), then 0.69->0.676;
theirs falls roughly 0.01/K: D=10: K=25 0.69874, 26 0.68672, 27 0.68414, 28 0.67381, 29 0.66265, 30 0.64688, 31 0.63679 (ours 0.70711,0.70711,0.69092,0.68414,0.68414,0.67915,0.67680).
D=11: K=26 0.70641, 29 0.68215, 30 0.66092, 31 0.65140 (ours 0.72792 ... 0.68406). Gap mass (D=10,11, K=25..31, Cats) ~0.0124; this includes 971, 911, 910, 972.
Guess: the 45deg corrugated band (our 0.70711 / Pell-17 band) with more band periods / a longer band whose tilted strip is partially relaxed, giving extra length-dependent saving
(the lb L-K approaching ~0.64 at K=31 means W~ 2*31*0.637 - 10 ~ 29.5; 1 unit of W saved per 4-5 K).
Fast targets: **n=686 (K=26,D=10): 26.68672 vs ours 26.70711 (gap 0.0204)**, **n=687: 26.70641 vs 26.72792 (0.0215)**, n=635: 25.69874 vs 25.70711 (0.0084, smaller K).
Note 686 and 687 sit at the exact 45deg stock constants, so the lead's 45deg tilted-band code is the starting point; the saved amount per n is ~0.02 in L.

**B. n = K(K+1)-9 (offset 9), K=16..30 (flat, K-independent-looking template).** 263, 297, 333, 371, 411, 453, 497, 543, 591, 641, 693, 747, 803, 861, 921.
Cats L-K: 411 0.77414, 453 0.77798, 497 0.78029, 543 0.78727, 591 0.79108, 641 0.79390, 693 0.79254, 747 0.79376, 803 0.79394, 861 0.79541 (ours 0.787->0.824).
Flat ~0.793-0.795 for K=25..29 vs our rising 0.80->0.82: our curve keeps growing while theirs saturates. Mass 0.0095 (offset 9 incl. neighbours 8,10: 0.025). Gap 0.0128-0.0284.
Guess: same grid-plus-tilted-strip template as ours, but with the strip rotated by a K-dependent non-45deg angle (free-angle polish of the strip) so L-K saturates.
Fast targets: **n=411: 20.77414 (ours 20.78694), n=453: 21.77798 (ours 21.79140), n=497: 22.78029 (ours 22.79709)**.

**Small-n, shared-constant bonus:** n=328 and n=365 (K^2+4, K=18,19) both reach L-K=0.59954 (ours 0.61437, 0.61286; gap 0.0148/0.0133) - a single reusable template, cheapest to reverse-engineer
(targets 18.59954, 19.59954); n=367 (19.67972 vs 19.69239) similar scale.

## Caveats
- Snapshot is Oct 29; leaders may have moved. Team name field contains emoji; matched after ASCII strip.
- Fits are descriptive only (K<=31, heavy step structure); decay exponents are not statistically separable.
