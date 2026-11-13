# Notes

## Seed

The search starts from the 969-square `n=908` tilted core plus exact upright
refill at side `31.66431822515163`. The current `n=971` incumbent side is about
`31.676795`, so adding both missing squares would be a material structural win,
not numerical dust.

## Method

For each large rasterized void, the script:

1. expands the void bounding box by a configurable margin;
2. generates dense centre and angle candidates only inside that box;
3. discards candidates that collide with any square fixed outside the window;
4. builds the exact unit-square conflict graph for the local seed and
   candidates; and
5. solves the local maximum independent set with HiGHS.

A window solve is accepted when it increases the local selection count.

## Runs

Starting from the certified 969-square `n=908` core refill:

```text
centre step   angle step   window margin   windows   best
0.1 deg grid  0.5 deg      0.6             30        969
0.2 deg grid  1.0 deg      2.5              2        969
```

The first run found zero candidate square centres that were separated from all
fixed squares in every one of the 30 largest voids. The wider run could replace
40 local seed squares but the exact MIS still selected only 40, so the total
stayed at 969. This confirms the missing two squares require a larger topology
change, not a finer placement inside the existing 969 contact graph.
