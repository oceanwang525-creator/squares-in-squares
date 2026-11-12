# Lipschitz-safe free-angle SLP (corrected)

`poc/free_angle_slp` linearised the support function `0.5*(|cos|+|sin|)` at its
kinks with a zero subgradient and (independently) flipped the sign of every
pair row. The resulting LP was infeasible at the incumbent for every tested
`n`; its recorded "0 wins" therefore said nothing about the packings.

This POC replaces the linearisation with a conservative, provably safe one.
For each nearby pair the support growth is bounded by

```
rad(theta + dtheta) <= rad(theta) + LIP*(t_i + t_j),   t_i >= |dtheta_i|
```

with `LIP = 1/sqrt(2)`. Wall rows use the same bound. Consequently every
accepted LP step is exactly overlap-free, and the unchanged official metric is
used only to confirm the decimal result.

## Formulation

Variables per square: `dx, dy, dtheta, t`; one global `dL`.

- pair row: `a.(dc_i - dc_j) + LIP*(t_i + t_j) <= slack`
- walls: `dx + LIP*t - dL <= L - x - e`, and the three mirrored rows
- `t >= dtheta`, `t >= -dtheta`, `t in [0, ta]`
- objective: `min dL`

The LP is rebuilt each round; trust regions grow after accepted steps and
halve on LP infeasibility.

## Measured portfolio result

Full 1000-group run at 32 processes, 8 rounds:

| metric | value |
|---|---:|
| groups | 1000 |
| strict wins | **1** |
| win | n=740, `27.706646513415677... -> 27.706646513373385...` |
| certified reduction | `1.554695213290798756474198831e-12` |
| wall time | 426.8 s |

Whole-file certification on the expanded top-five team union: literal
`22.173824282917252`, actual pandas `22.173824282917252`, both 1000/1000. Reports in `docs/`.

So the portfolio is a local optimum under this safe free-angle model as well;
the single win is a 4.2e-11 side improvement at n=740.

Script: `poc/lip_slp.py`. Log: `poc/loop_log.md` Loop 23.
