# Tilted cores with upright corner pockets

Keep only the tilted squares from each nearby donor. Rotate and translate that
rigid core, then fill remaining space with upright squares from lattices
anchored at each of the four container corners. Four greedy orders prioritize
different corner pockets when their lattice sites conflict.

The scan evaluated 585,900 core poses across 16 targets, at a side one
millionth below the `2c1e6a6` incumbent. None reached the required count. The
best count for n=147 was 131; for n=199 it was 182. The results constrain the
sampled poses and greedy upright fills only. See [scan data](data/scan.json).

```sh
python poc/midrange_diamond_pockets/poc/search.py \
  poc/pell_corrugation/data/border-scan/merged.csv /tmp/diamond-pockets
```

Requires NumPy and Numba. No replacement coordinates were generated.
