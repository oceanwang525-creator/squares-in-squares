# Hypothesis H126 (`band3ext`): Next-Gen `band3` Corner-Melted 2-Brick Pinwheel Sweep

## Summary
H126 extends `poc/swarm_band3/` (`band3`, H94) by harvesting corner bricks (`g = 3..14`, `f <= 0.86`) from `origin/main` `merged.csv` and recent certified PR packs (`vstrip492`, `vstrip852`, `peelstep`, `pocketpeel`, `thinband8`, `b7xplant`, `watchxplant`, `gstep`, `liftpolish3`), combining them with the existing external brick pool (`1,820` deduplicated shapes + transposes), and sweeping zero-overlap (`ov=0: ct=0` or corner-touch `ct=1`) 2-brick pinwheels with `del_k = g_1 + g_2 - r in {0, 1, 2}`, masked C++ corner melt, `pslp.py` (with 4-corner static block reduction + `shake_squeeze`), and `lpslack` across `m = 21..31, r = +9..+17`.

## Certified Wins (PR #1 vs `origin/main` `20fe263`, `literal == pandas`, `2/2` wins, `total_ds_L2n = -5.918436e-04`)
- **`n=910`** (`m=30, r=+10`, `B5_20x10_h63_walk_T` `10x20(g=5)` + `B5_20x10_h63_walk` `20x10(g=5)`, `flipx_flipy` corner-melted pinwheel):
  - `origin/main` (`20fe263` `merged.csv` / `swarm_ratchetall`): `30.65634879873266828533838404377454675006427374198247`
  - `band3ext`: **`30.65174623555355154346701594172274005167897156141426`** (**`dL = -4.602563e-03`**, `ds_L2n = -3.100817e-04`)
- **`n=587`** (`m=24, r=+11`, `B5_16x9_n492_TL_mw_vs1_B6_9x17` `9x17(g=6)` from `vstrip492` + `H5_7x15_n450_BRT_T` `15x7(g=5)`, `flipx_id` corner-melted pinwheel):
  - `origin/main` (`20fe263` `merged.csv` / `external`): `24.74183578099088723764676283008914692615831252787778`
  - `band3ext`: **`24.73849315453727506557097165057468562513908703083146`** (**`dL = -3.342626e-03`**, `ds_L2n = -2.817619e-04`)

## Verification
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest poc/swarm_band3ext/tests/test_band3ext.py -v
```
