# Hypothesis H94 (`band3`): High-`r` Asymmetric / Corner-Melted $K=2$ Pinwheel Rebuild

## Summary (PR #1)
H94 targets `F3B` high-remainder packings (`r = +14..+21`, `m = 20..31`) by harvesting minimal-perimeter high-`g` bricks (`g >= 6`) from `merged.csv` + external brick libraries (`B7_9x22_peel_left`, `B7_10x22_vstrip`, etc.) and assembling corner-touching (`w_1 + w_2 = m, h_1 + h_2 = m`) and asymmetric $K=2$ pinwheels followed by C++ seam melt (`cpp_batch_melt`) and `joint_xytheta_slp_tight`.

### Certified Gate Win (vs `origin/main` `9abec6b86e60c1fbdfcc6791460d79795768770a`)
- **`n=975`** (`m=31, r=+14`, `B7_9x22_peel_left` `9x22, g=7, f=0.71584` + `B7_22x9` `22x9, g=7, f=0.71584` corner-melted pinwheel):
  - `origin/main` (`9abec6b8`) incumbent: `31.71877369634442420730106835126991248827712132963533`
  - `band3` certified (`literal == pandas`): `31.71656693256243245199169310243547960426919336722686`
  - **`dL = -2.206764e-3`** (and `-4.973070e-3` vs `8c623fc`), `ds = -7.0674e-5` (`L/sqrt(n)-1`).

## Reproduction & Verification
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 poc/baseline/poc/.venv/bin/python poc/swarm_band3/poc/run_gate_probe.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 poc/baseline/poc/.venv/bin/pytest poc/swarm_band3/tests/test_band3.py -v
```
