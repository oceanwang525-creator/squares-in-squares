# Variable-Slope Band Optimizer (slope_band POC 2)

Continuous variable-slope band optimizer centered around $\theta = 24.1004^\circ$ ($0.420631$ rad, matching rational lattice slope 17:38 and 1:2 rational staircases).

Implements the 4-sign-piece epigraph LP formulation to prevent stall at the non-differentiable $r=0$ Separating Axis Theorem (SAT) support kink, supporting heterogeneous angular degrees of freedom (free tilted band squares vs pinned $\theta = 0$ upright squares).

## Structure

- `poc/variable_slope_optimizer.py`: Core LP polisher, 4-sign-piece epigraph constraints, SAT clearance checker, and pocket layout builder.
- `poc/slope_band.py`: Top-level CLI and driver.
- `poc/test_variable_slope_optimizer.py`: Unit tests and geometric invariants.
- `docs/variable_slope_optimizer.md`: Detailed mathematical formulation and pocket anatomy.

## Usage

Run from the repository root:

```sh
python poc/slope_band/poc/slope_band.py --help
python poc/slope_band/poc/slope_band.py --n 908 --iterations 30
