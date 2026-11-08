# Slope Band (POC 1)

Exact 1:2 rational slope band generator ($\theta = \arctan(1/2) \approx 26.565^\circ$) for square packings.

## Overview

Single-band $45^\circ$ constructions fix $\theta = \pi/4$, inflating bounding boxes diagonally by $h(\pi/4) = \sqrt{2} \approx 1.414214$.
Along a rational slope of $1:2$ ($\theta = \arctan(1/2) \approx 0.463648\text{ rad}$), the bounding extent of rotated squares drops to:
$$h(\arctan(1/2)) = \cos\theta + \sin\theta = \frac{3}{\sqrt{5}} \approx 1.341641$$
This yields a direct transverse inflation saving of $\Delta L_{\text{adv}} = \sqrt{2} - 3/\sqrt{5} \approx 0.072573$ per square.

Integer Cartesian steps of $\Delta x = 2, \Delta y = 1$ advance exactly along the band tangent $\mathbf{t} = (2/\sqrt{5}, 1/\sqrt{5})$ with step length $\sqrt{5} \approx 2.236068$ and zero normal displacement ($\Delta \mathbf{r} \cdot \mathbf{n} = 0$). Commensurate band lengths match continued fraction convergents of $\sqrt{5}$ ($38/17$, $9/4$), with residual vernier mismatch absorbed by 4-sign-piece epigraph LP relaxation.

## Incumbent Comparison Ledger (merged.csv)

| Target n | Candidate L | Our Incumbent L (merged.csv) | Delta L | Gate Status |
|:---|---:|---:|---:|:---|
| 908 | 30.63702956 | 30.63702956 | 0.00000000 | Tie |

## Structure

- `poc/slope_band.py`: Generator, periodic $2 \times 1$ tooth staircase builder, asymmetric corner buffers, 4-sign-piece LP relaxation, and falsification gate test.
- `poc/test_slope_band.py`: Unit tests verifying geometric invariants, $2 \times 1$ tooth alignment, LP feasibility, and falsification gate.

## Usage

```sh
python poc/slope_band/poc/slope_band.py --help
python poc/slope_band/poc/slope_band.py --falsification-test
python poc/slope_band/poc/slope_band.py --prototype --polish
