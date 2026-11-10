# Free-angle re-polish of repaired gallery packings

## Idea

The gallery repair in `poc/gallery_repair` froze every angle at the gallery's two-decimal-degree value and moved only centers inside a +/-0.03 trust region. That leaves rotation as the one untested local degree of freedom in the 194 repaired layouts now in `merged.csv`. This POC re-implements the same fixed-topology separating-axis LP, then adds coordinate descent on angles: try a small angle increment on one tilted square, rebuild the pair axes and sign pattern, re-solve the LP, and keep the move only when the box side strictly decreases.

## Why free angles could help

The repair recovered the rounding loss in center coordinates, but the gallery's angle labels are still quantized to 0.01 degree. Changing an angle changes a square's wall extent (`0.5*(|cos|+|sin|)`) and the contact-axis radii, so it can relieve a jammed contact that no center translation can fix. The question is whether that residual freedom is worth anything.

The arithmetic is unfavorable. Per-n score contribution is `L/sqrt(n) - 1`, so reducing `L` by `dL` gains `dL/sqrt(n)`. Over the 194 repaired wins, the sum of the provenance `box_growth` values weighted by `1/sqrt(n)` is **0.000723788**, i.e. under 0.001 score. The current gap to the 22.19 target is 0.026763. Recovering only the gallery rounding loss is therefore at most about 2.7% of the gap. The only reason to continue this line is evidence that free angles beat the rounded gallery layout itself, not merely that they tidy up rounding.

## Measured evidence

Targets were the five largest per-n repair gains (`gain_score = (old_L - new_L)/sqrt(n)`), which includes the two named in the brief, n=490 and n=535.

| n | incumbent L | candidate literal L | candidate pandas L | literal cert | pandas cert | dL | per-n score delta |
|---:|---|---|---|---|---|---:|---:|
| 490 | 22.63709157611762284978063385354474235100497025996447 | 22.63706688789393827013928617618534045638694073348123 | 22.63706688789393800000000000000000000000000000000000 | better | better | 2.468822368e-5 | 1.115300260e-6 |
| 535 | 23.63709157721760161633892992627536955296818632632494 | 23.63706688496243967041281515173112254870357455961379 | 23.63706688496244000000000000000000000000000000000000 | better | better | 2.469225516e-5 | 1.067539283e-6 |
| 293 | 17.62308186328582660435408798749179482001636642962694 | 17.62308186327792550667164158773534564748119068976849 | 17.62308186327792600000000000000000000000000000000000 | better | better | 7.9e-12 | 4.62e-13 |
| 631 | 25.60879790098457715709805811998833746656600851565599 | 25.60879790098457736071216491653696328174276049858582 | 25.60879790098457800000000000000000000000000000000000 | worse | worse | -2.0e-16 | -8.11e-18 |
| 932 | 30.94088858973680488523063730754714129034255165606737 | 30.94088858973680868567633943427176595873907252403763 | 30.94088858973681000000000000000000000000000000000000 | worse | worse | -3.8e-15 | -1.24e-16 |

Strict wins on both literal and pandas transport: **3 of 5**. Total measured score delta: **2.182840005e-6**. Only n=490 and n=535 are genuine free-angle gains (one square moved by +0.005 rad in each). The n=293 "win" has no accepted angle move; it is a 7.9e-12 L difference from re-solving the LP and emitting with the safety dilation, i.e. numerical dust, not evidence for free angles. n=631 and n=932 found no improving single-square move and their re-solved candidates are microscopically worse after dilation.

## Limits and risk

- This is a local one-square coordinate descent, not a global search. It ran 3 passes over tilted squares with the ladder +/-0.002, +/-0.005, +/-0.01, +/-0.02 rad. No annealer or parameter sweep was run.
- The topology is fixed by the same active-axis rule and 1.6 center-pair radius as `repair_gallery.py`; a different topology is out of scope.
- The +/-0.03 center trust region is held relative to the merged coordinates, so this cannot make a large layout change.
- Candidate CSVs are per-n files under `data/`. They are not merged into the portfolio and no whole-file 1,000-puzzle check was run here; the lead must merge and validate.
- The two real wins are about 1e-6 score each, three orders of magnitude below the 0.0268 target gap. The main risk is that the remaining gallery angle rounding is mostly not recoverable by coordinate descent, and that free angles only tidy rounding rather than beat the rounded gallery layout.

## Reproduce

```sh
python3 poc/free_angle_repolish/poc/free_angle_repolish.py
# or a subset / different budget
python3 poc/free_angle_repolish/poc/free_angle_repolish.py --targets 490,535 --passes 3
```

Outputs: `data/candidate-{n}.csv`, `data/candidate-{n}-pandas.csv`, `data/results.json`, `data/measured.csv`. The run is deterministic (fixed RNG seed per target). It needs only the packages already used by `gallery_repair` (`numpy`, `scipy`, `pandas`, `mpmath`) and the shared official metric.
