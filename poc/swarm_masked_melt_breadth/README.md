# H132 breadth-first masked-pocket melt

## Idea

H13 showed that masked void, tight-contact, and upright/tilted-seam pockets can
re-route local SAT contacts without destroying the incumbent bulk. H128 then
reapplied that mechanism to its own winners. This hypothesis applies the same
proven three-family melt mechanism breadth-first to high-gap rows that were not
in either target log.

The quick-fail gate is `n = 367, 902, 530`. Each incumbent gets a fresh,
deterministic pocket radius, expansion, coordinate/angle perturbation, C++ seed,
and melt temperature chosen for its density. A real win is a complete group
strictly below the newest upstream merge artifact in both the literal and
actual pandas CSV transports.

## Baseline

- `upstream/main`: `6bfe2c48f33bc3b645dba8489297c4720f9d0fe8`
- `upstream/merge`: `7c93cb082e10ba9125156a9f22653f6131552ec8`
- Artifact: `merged-22.15277788725.csv`
- Tracked baseline SHA256:
  `1d623a7b76da602fb955346d815dc6e22760c150a9be5b526224372f64984767`

## Result

The quick gate found one certified strict win:

| n | seed | before L | after L | dL |
|---:|---|---:|---:|---:|
| 530 | `mpk_seam_5` | 23.532367090941507798 | 23.525757668744916601 | -6.609422196591e-3 |

Both the official literal transport and the actual `pandas.read_csv` /
`to_csv` transport pass. `final.csv` contains only this complete group.

The remaining gate rows did not certify a strict win: `n=367` missed by
`+1.839733e-7`, while `n=902` had no valid polished candidate (minimum final
penetration `1.59e-4`). A first breadth extension then missed at `n=791`
(`+2.405968e-5`) and `n=908` (`+1.854286e-5`) before the work was redirected
to nearby rows.

## Run

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
/Users/Guestly/Documents/Codex/2026-11-12/https-www-kaggle-com-competitions-squares/squares-team/.venv/bin/python \
  -u poc/swarm_masked_melt_breadth/poc/run_gate.py \
  --targets 367 902 530 --threads 4 --top-k 6 \
  --out-json poc/swarm_masked_melt_breadth/docs/gate_results.json
```

Verify an export with:

```bash
python poc/swarm_masked_melt_breadth/poc/verify_export.py
python .github/scripts/check_final_csv.py
```

## Files

- `poc/run_gate.py`: checkpointed masked-melt runner and strict exporter.
- `poc/pocket_seeds.py`: H13 seed families with H132 target presets.
- `poc/verify_export.py`: independent literal and pandas transport verifier.
- `tests/`: seed-shape, preset, and final-schema tests.
