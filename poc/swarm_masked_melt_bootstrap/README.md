# H128 second-generation masked-pocket melt

## Idea

The H13 masked-melt runs changed several certified rows and, with them, the
local SAT contact graph. Reusing the original contact stresses would therefore
miss any slack created by the first melt.

This POC applies a second relaxation generation to the already-improved merge
rows. It keeps the H13 masked void / tight-contact / upright-tilted-seam
starts, but uses a fresh RNG stream, expands the initial box by `0.036` instead
of `0.032`, widens the local pocket radius from `4.5..7.2` to `4.8..8.0`, and
modestly widens the coordinate and angle perturbations. Each start is polished
with L-BFGS, joint `xytheta` SLP, and fixed-angle LP before canonical literal
and pandas certification.

## Baseline

- `upstream/main`: `6bfe2c48f33bc3b645dba8489297c4720f9d0fe8`
- `upstream/merge`: `7c93cb082e10ba9125156a9f22653f6131552ec8`
- Artifact: `merged-22.15277788725.csv`
- Artifact and tracked `merged.csv` SHA256:
  `1d623a7b76da602fb955346d815dc6e22760c150a9be5b526224372f64984767`

## Result

Four complete groups pass both the official literal certificate and the actual
`pandas.read_csv` / `to_csv` transport:

| n | seed | before L | after L | dL |
|---:|---|---:|---:|---:|
| 634 | `mpk_void_9` | 25.688381146233741 | 25.688376088157854 | -5.058076e-6 |
| 695 | `mpk_hot_1` | 26.831780320347433 | 26.831690049211594 | -9.027114e-5 |
| 861 | `mpk_seam_3` | 29.817033759741868 | 29.816317663506829 | -7.160962e-4 |
| 926 | `mpk_seam_4` | 30.870628504827978 | 30.870349580557340 | -2.789243e-4 |

The resulting official score is approximately `22.152776298129506`, a
reduction of `1.589120493861e-6` from the merge artifact.

The quick gate also tested `n=459` (`+8.817e-6`) and the extension tested
`978` (no valid polished candidate), `499` (`+1.011e-5`), `804`
(`+5.171e-6`), and `920` (`+5.816e-6`). Only the four strict groups above are
exported.

## Run

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python -u poc/swarm_masked_melt_bootstrap/poc/run_gate.py \
  --targets 459 634 695 --threads 4 --top-k 6 \
  --out-json poc/swarm_masked_melt_bootstrap/docs/gate_results.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python -u poc/swarm_masked_melt_bootstrap/poc/run_gate.py \
  --targets 459 634 695 861 978 499 804 920 926 --threads 4 --top-k 6 \
  --out-json poc/swarm_masked_melt_bootstrap/docs/scale_results.json
```

Verify the exported final independently:

```bash
python poc/swarm_masked_melt_bootstrap/poc/verify_export.py \
  > poc/swarm_masked_melt_bootstrap/docs/verification.json
python .github/scripts/check_final_csv.py
```

## Files

- `poc/run_gate.py`: checkpointed gate/scaling runner and canonical exporter.
- `poc/pocket_seeds.py`: second-generation masked starts.
- `poc/verify_export.py`: re-reads `final.csv` and checks both transports.
- `tests/`: seed-shape and final-schema tests.
- `docs/ckpt/`, `docs/*_results.json`, `docs/*_stdout.log`: checkpoints and raw
  run records.
- `data/win_n*.csv`, `final.csv`: complete certified winning groups only.
