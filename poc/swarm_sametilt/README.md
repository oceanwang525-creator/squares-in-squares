# `poc/swarm_sametilt` — Hypothesis H14: Same-Tilt Dual-Crystal Construction (`~65° + ~65°`)

## Hypothesis H14
GuestData's `n=910` (`L = 30.65750541071413945`, PR #21) beats `swarm_homology`'s `n=910` (`30.67702432583`) by `dL = -0.019519` (`-1.95e-2`) because **both tilted crystal arms share the SAME orientation family (`~65°` mod 90°)**:
1. `LEFT_65`: 141 squares (`x in [0.65, 10.03], y in [10.42, 29.26]`, pitch `(0.915, -0.411)` & `(0.015, 1.10)`).
2. `BOT_65`: 99 squares (`x in [12.43, 26.25], y in [0.59, 8.99]`, pitch `(0.418, 0.912)` & `(1.104, -0.002)`).
3. Surrounded by upright `BL` (`110` squares), `TR` (`420` squares), `BR` (`36` squares), `TL_strip` (`27` squares), and seam/transition squares.

## Tracks
- **Track A (`poc/trackA_surgery.py`)**: Direct neighbor surgery around `n=909/910` (`n=905..915`) and other same-tilt packings (`920`, `793`) via void insertion / high-slack deletion + C++ masked melt + joint `(x, y, theta, L)` SLP.
- **Track B (`poc/trackB_homology.py`)**: Homologous `m -> m +/- k` border extension (`+61` L-strip to `m=31`: `970..976`) and border peel (`m=29..24`: `851, 850, 794, 793, 739, 686, 635, 586` + cascading peels & `+k` expansions).
- **Track C (`poc/trackC_conversion.py`)**: Synthetic same-tilt conversion (`~25° + ~65° -> ~65° + ~65°` or `~25° + ~25°`) on other two-arm opposite-tilt incumbents.

## Certification
Every candidate is certified in BOTH `literal` and `pandas` transport modes via `poc/auto_union/poc/union.py` (`check`) against the full frontier (`origin/main` + PR #21 + PR #22).
Only strictly winning candidates (`dL <= -1e-9`) are saved to `poc/swarm_sametilt/data/win_n*.json` and compiled into `poc/swarm_sametilt/final.csv`.

## Results (batch 1, certified literal==pandas vs origin/main 252fe8b)
| n | L | dL | source |
|---|---|---|---|
| 793 | 28.659432082141936751 | -1.780745e-2 (vs guestdata_atlas 28.677239531) | `peel2_910_BL_dropMinProt_reloc` (910 -> peel 2 border layers) |
| 850 | 29.649077415119136501 | -7.776834e-3 (vs merged 29.656854249) | `peel1_909_BL_v1` |
| 971 | 31.657506410714138034 | -5.474715e-4 (vs merged 31.658053882) | `ext61_910_raw` (+61 border strip on GuestData 910) |

Certify: `PYTHONPATH=. python poc/swarm_sametilt/poc/certify_final.py origin/main`.
