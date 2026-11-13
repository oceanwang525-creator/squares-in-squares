# Notes

`fillfast` reports `n + 1` capacity at the current side for:

```text
n=627, 678, 843, 902, 963
```

The surplus is one upright square. A direct refresh needs a small dilation
because many contacts are exactly touching. The certified side after dilation
does not fall below the incumbent.

`n=627` was measured explicitly:

- current folded level: `25.5355339...`
- refill at current side: 628 squares
- first literal+pandas-feasible candidate needs a `5e-10` dilation
- resulting side: `25.5355339181007675`, above the incumbent

Small box reductions preserve a 628-square capacity in the float conflict graph
but fail exact certification; larger reductions violate the tilted-wall
contacts. No target produces a strict two-transport win.

The useful conclusion is structural: the extra square is numerical slack in
the fillfast model, not recoverable free area. Shrinking this family requires a
different tilted pocket, not another upright MIS.
