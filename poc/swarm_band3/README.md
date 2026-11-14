# Hypothesis H94 (`band3`): High-`r` Asymmetric / Corner-Melted $K=2$ Pinwheel Rebuild

## Summary
H94 targets `F3B` high-remainder packings (`r = +13..+22`, `m = 20..31`) by:
1. Harvesting minimal-perimeter high-`g` bricks (`g >= 6`) from `merged.csv` + external brick libraries (`swarm_gstep`, `swarm_cleanpeel2`, `swarm_stepdown`, `swarm_peelstep`).
2. Synthesizing 1-layer peeled & `vstrip`/`hcol`-spliced bricks (`poc/swarm_band3/poc/synth_and_peel.py`) to turn `ct=1` (`w1+w2=m, h1+h2=m`) pairs into zero-overlap (`ov=0, ct=0`) or thinner corner-touching bricks.
3. Assembling corner-touching (`w_1 + w_2 = m, h_1 + h_2 = m`) pinwheels, melting the inner corner with widened `corner_sm` + L-BFGS / full-melt rescue (`poc/swarm_band3/poc/run_scale.py`), and polishing via kink-safe full-row SLP (`poc/row_ratchet_980/poc/pslp.py`) + `lpslack` (`poc/swarm_band3/poc/pslp_polish_and_sweep.py`).

## Certified Wins
### PR #2 (vs `origin/main` `00d4305`, `literal == pandas`, `3/3` strict wins, total `ds = -3.489688e-04`, `ds_L2n = -7.092225e-04`)
- **`n=974`** (`m=31, r=+13`, `B6_9x21_peel_left + B7_10x22_vstrip_T` `ct=1` corner melt + `pslp`):
  - `origin/main` (`00d4305`): `31.700186957068922...` -> **`31.69258193382914020216233915998624843367026953288747`** (**`dL = -7.605023e-03`**, `ds_L2n = -4.949728e-04`)
- **`n=743`** (`m=27, r=+14`, `B7_8x19_n916_TL + B7_19x8` `ct=1` corner melt + `pslp`):
  - `origin/main` (`00d4305`): `27.753852959175397...` -> **`27.75219911014502018473946740730354496024073426404537`** (**`dL = -1.653849e-03`**, `ds_L2n = -1.235513e-04`)
- **`n=914`** (`m=30, r=+14`, `9x21(g7) + 21x9(g7)` `ct=1` corner melt + `pslp`):
  - `origin/main` (`00d4305`): `30.731116268614116...` -> **`30.72976746967652015854075232919969068075917531526521`** (**`dL = -1.348799e-03`**, `ds_L2n = -9.069844e-05`)

### PR #1 (Merged in `d9ce487`)
- **`n=975`** (`m=31, r=+14`, `B7_9x22_peel_left` + `B7_22x9` corner melt): `31.718773696...` -> **`31.71656693256243245199...`** (`dL = -2.206764e-3` vs `9abec6b`, `-4.973070e-3` vs `8c623fc`). (Omitted from PR #2 `final.csv` as already superseded in `origin/main` `00d4305`.)

## Verification
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 poc/baseline/poc/.venv/bin/pytest poc/swarm_band3/tests/test_band3.py -v
```
