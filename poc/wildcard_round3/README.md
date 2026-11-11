# wildcard_round3: phase-0.5 corrugation lead, global count scan (all n). Result: negative, nothing built

Estimated score gain: 0. No CSV written, `merged.csv` untouched, nothing sent to `squares score`. Measured L: none.

## Phase class by arithmetic
Rigid 45deg core (w,a1,a2,b1,b2,t): L = (a1+a2)/2 + 1 + w/sqrt2 + t. The two staircase tooth lattices are corner-anchored, so their u-phase differs by 0.5 step iff s=a1+a2 is odd, else 0 (incumbent 635: s even). For w=1, s odd gives L = K+0.207, i.e. n ~ K^2 - 0.6K, never the K^2+10 family. So the w=1 phase-0.5 band the corrugated_flanks README left open does not exist near any gapped n (area alone rules out n=635, 686).
Phase-0.5 cores with w>=2 do exist (e.g. w=13, 10, 6, 3, 2).

## Scan (`poc/scan_phase.py`, `docs/scan_phase.json`, `poc/slack.py`)
Exhaustive over w<=39, t<29, s<64, a1,a2,b1,b2, exact `enum45.count`, L<=32.1, every n in 1..1000. Sanity: even-s cores never beat our L (0 n). For each n, lowest-L odd-s core holding >= n squares vs our L. Excess = L_odd - L_ours (needs a corrugation shave > excess; dL = dH/sqrt2, dH measured 0.01 at best for w>=2):

| n | w | count | excess over ours | longitudinal slack / length | leader L - ours |
|---|---|---|---|---|---|
| 739 | 13 | 739 | 0.0015 | 0.0009 | -0.0068 |
| 297, 332 | 6 | 297, 332 | 0.0022 | 0.0012 | -1e-5 |
| 907 | 3 | 907 | 0.0030 | 0.0016 | -0.0160 |
| 849, 908 | 20 | same | 0.0051 | 0 | -0.010, -0.031 |
| 494 | 6 | 494 | 0.0065 | 0.0038 | -0.0016 |
| 328 | 3 | 328 | 0.0070 | 0.0044 | -0.0148 |
| 365 | 3 | 365 | 0.0085 | 0.0029 | -0.0133 |
| 202 | 6 | 202 | 0.0147 | 0.0059 | -0.0036 |
| (~45 more, excess 0.006-0.015, all w>=2) | | | | | |

Counts equal n at L_odd (rigid, exact); at L < L_odd no tested config holds n. n with excess 0 (26, 85, 367, 368, 407, 448, 626, 677, 730, 785, 842, 901, 962) are already our odd-s cores, no gain available.

## Why corrugation cannot close the gap (`poc/rate.py`, `docs/rate_w3.jsonl`)
The shave needed everywhere along the band is dH = sqrt2*excess (0.002-0.02), paid out of the longitudinal slack (band length - floor), which is only 0.02-0.1 over 17-40 units (<= 0.6% per length; 0 for many). Cell relaxation, w=3, dphi=0.5, step 0.002:
| k,m | slack/length | dH |
|---|---|---|
| 5,7 | 0.0100 | -0.010 |
| 12,16 | 0.0572 | -0.008 |
| 17,24 | 0.0017 | 0.000 (no gain) |
So gain is ~0.01 at slack >= 1%/length and vanishes at the 0.1-0.4% slack these cores have. Required dH exceeds what the slack can pay in every row of the table (best case n=739/297/332 need dH 0.002-0.003 with slack 0.0009-0.0012/length, where the cell shows nothing). w=1 (rate ~2x higher) has no applicable core.

## Conclusion
Negative. The phase-0.5 corrugation lead is closed by arithmetic (w=1 odd-s lives at K+0.207) plus slack (w>=2 cores have <0.6% slack/length, cell gain 0 there). Not tried: concentrating slack by giving up a square in rows (loses a count), or cell runs for w=6,13 at exact slack (w=3 saturates at 0.01, assumed similar).
