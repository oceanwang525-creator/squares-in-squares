# Pinwheel layer growth

## Idea

Grow an existing interlocked pinwheel across two orthogonal cuts. The first target is `157 → 184`, where the incumbent for 184 is the side-14 grid. The preceding historical steps `109 → 132 → 157` added 23 and 25 squares, so a subsequent 27-square step is a natural candidate. The secondary targets are the next two counts above the final non-grid incumbent in each family, `183 → 212` and `211 → 242`.

## Method

Keep all donor squares and insert one layer at each cut, using displacement `1/cos(theta)` normal to the cut for a tilted square. The layer copies are greedy and must clear the finite interlock within a bounded separating-axis penetration. A fixed-box L-BFGS-B settle and a joint center/angle sequential LP with true SAT acceptance then try to close the new box. The pinned input is upstream `b57e78b89784a87904f0793ec388276741d9f8e8`, `poc/pell_corrugation/data/border-scan/merged.csv`, SHA-256 `1f9c0fd3b7a88e5a0f6c44b567ff2ef354872812c6b0ad8e37807bd1a64e965c`.

Joost de Winter published the successful two-cut growth of Samanyu-dev’s 109-square pinwheel to 132 and then 157 squares. His workflow used fixed-side LP, IPOPT and high-precision contact polishing. The search here uses SciPy and starts from the incumbent coordinates. It also sweeps more cuts and gives each candidate a separate feasibility and compaction budget. [Published method](https://github.com/jlevy/squares/blob/main/packing/resources/web/de-winter-method-132-157-2026-11-12/source/method_n132_n157/README.md)

The cached `sqk.grow` tried unit slab copies and fixed-angle LP repair. Here tilted copies use their own pitch, nearby copies are screened before compaction, and center and angle changes are both searched. The workspace’s earlier `constructnext_sweep.py` and `constructnext_rotate.py` perform cropping and enclosing-box rotation. The team’s flat-arm clipping and fresh block-split seeds addressed a different problem and lost the finite corner interlocks.

## Result

Zero strict wins. Each of the three proposed extensions reached its target square count in the seed sweep. The bounded SciPy search polished ten seeds per target and ten per historical control. The nonlinear continuation completed four IPOPT rounds on two `n=184` seeds. Its smallest solver-success side was `14.043394355300`, against the incumbent `14`. The two historical controls reached their expected counts but did not reproduce the published sides. [Run details](docs/results.md)

## Reproduce

From the repository root with NumPy, SciPy, Numba and the baseline package’s metric dependencies installed:

```sh
python poc/pinwheel_layer_growth/poc/search.py \
  --incumbent poc/pell_corrugation/data/border-scan/merged.csv \
  --out poc/pinwheel_layer_growth/data \
  --family 157:184 183:212 211:242 109:132 132:157
```

The output is a bounded search log. Any emitted replacement groups must pass the official metric on literal and pandas-round-tripped decimals and be strictly shorter on both. Full submission validation remains a separate gate.

With CasADi installed, the optional nonlinear continuation reconstructs the two strongest exact-count seeds from that log. It constrains all four corners against each chosen separating axis, allows the axis to rotate with its owner, and selects the axes again before the second bounded IPOPT solve.

```sh
python poc/pinwheel_layer_growth/poc/ipopt.py \
  --incumbent poc/pell_corrugation/data/border-scan/merged.csv \
  --scan poc/pinwheel_layer_growth/data/scan.json \
  --out poc/pinwheel_layer_growth/data
```
