# Failure note: the 5/5/26 shortfall

The prior donor-slice arithmetic recorded a maximum composite capacity of 640,
920 and 961 at sides strictly below the three incumbents. That is a shortfall of
5, 5 and 26 squares. This implementation's broader donor and cut enumeration
finds 644, 923 and 961, so the measured shortfall is 1, 2 and 26. The difference
is +4 at `n=645` and +3 at `n=925`; `n=987` is unchanged. Both versions lose.

## What the shortfall means

A composite is feasible at side `L` only when

```
c + floor(L - w) * floor(L) + floor(w) * floor(L - h) >= n
```

where `c` is the crop capacity and `w x h` is its tight fractional box. Below
the incumbent, the grid term is quantized. It does not grow continuously; it
jumps only when `L` crosses `w + i`, `h + j`, or an integer.

### `n=645`: one square short

The best crop below `L=25.870318123` is 644 squares with tight box
`25.869769561`. The remaining width and height are both below one grid pitch,
so `floor(L - w) = floor(L - h) = 0` and the grid adds nothing. The next crop
with 645 squares has tight box `25.881109226`, which is `0.010791104` above the
incumbent. The integer cut grid cannot add the missing square without crossing
the incumbent side. The prior 640 result used a narrower donor/cut set; the
mechanism is the same.

### `n=925`: two squares short

The best crop below `L=30.863218373` is 923 squares with tight box
`30.859361960`. Again the remaining `0.003856413` is below one grid pitch, so
`G = 0`. A 925-square crop has tight box `30.875854187`, which is
`0.012635814` above the incumbent. The prior 920 result is 3 squares lower; the
gap is still a donor-cut gap, not a repairable overlap.

### `n=987`: 26 squares short

The best below-incumbent composite is a 22x22 crop from donor 999 (484 squares)
plus a `9 x 31 + 22 x 9 = 477` upright grid, total 961, at side `31.000000100`.
Below side 32 the grid counts are fixed by the integer floors. The missing 26
squares require another grid column and another grid row, which first fit at
side 32. The capacity-matched composite is therefore above the incumbent: a
956-square crop from donor 988 plus a 31-square grid at side `31.875854287`,
`0.009549022` above `31.866305265`. This is the structural case: the grid
quantum, not the crop, sets the 26-square shortfall.

## Conclusion

No capacity-matched composite exists at a side below any of the three
incumbents. The emitted packings are valid and certified, but all three sides
are larger than the incumbents. The donor-slice construction is a negative
result; the 5/5/26 prior shortfall and the 1/2/26 measured shortfall are both
symptoms of the same integer-cut and upright-grid quantization.
