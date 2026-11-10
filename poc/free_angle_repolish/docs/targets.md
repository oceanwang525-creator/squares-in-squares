# Target list

Source: `poc/pell_corrugation/data/border-scan/gallery-repair-provenance.json`, `wins` dict (194 entries).

Selection: rank by `gain_score = (old_L - new_L)/sqrt(n)`, the per-n score already recovered by the fixed-angle repair. This is the quantity that matters for the portfolio and it reproduces the brief's named top two, n=490 and n=535. Note that the provenance field named `box_growth` is `new_L - gallery_L` (the repaired exact layout's side above the gallery's rounded side label), not the incumbent reduction; ranking on that raw field puts n=707 first. The weighted ceiling below still uses the `box_growth` field as instructed.

| n | old_L | new_L | gain_score | tilted squares | unique angles |
|---:|---|---|---:|---:|---:|
| 490 | 22.656854249494646 | 22.637091576117623 | 8.927866e-4 | 130 | 32 |
| 535 | 23.656854249492497 | 23.637091577217602 | 8.544148e-4 | 196 | 58 |
| 631 | 25.627416997969686 | 25.608797900984577 | 7.412145e-4 | 186 | 22 |
| 293 | 17.633982502188317 | 17.623081863285827 | 6.368221e-4 | 56 | 13 |
| 932 | 30.960209721993144 | 30.940888589736805 | 6.328847e-4 | 335 | 49 |

The n=293 row has the smallest n and is cheap; the others span the medium and high-n repaired layouts. The five chosen are the top five by `gain_score`; n=932 is fifth and n=996 is eighth.

## Ceiling arithmetic

```
sum_{n in wins} box_growth(n) / sqrt(n) = 0.000723788
```

using the provenance `box_growth` field (`new_L - gallery_L`) for all 194 wins. The target gap is `22.216763458415837 - 22.19 = 0.0267634584`. The ceiling is 2.70% of the gap, so a free-angle method that only recovers rounding cannot approach the target.
