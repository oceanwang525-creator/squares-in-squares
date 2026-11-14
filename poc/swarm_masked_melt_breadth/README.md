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

- `upstream/main`: `331b307fbf67fc721609b121b9606b399b2359b9`
- `upstream/merge`: `2554d5b96b64b6d633daf922446331d8f35044da`
- Artifact: `merged-22.15258085489.csv`
- Tracked baseline SHA256:
  `c9e5398775e31b536cbfa1519ab9c6066c95f98edf1d7bc9bec3a54a85cafd9e`

## Result

The quick gate found one certified strict win, and the neighboring D=1 family
produced three more:

| n | seed | before L | after L | dL |
|---:|---|---:|---:|---:|
| 442 | `mpk_seam_w0_5` | 21.532729070625025365 | 21.525794787811988426 | -6.934282813037e-3 |
| 485 | `mpk_hot_w0_2` | 22.532422372157410469 | 22.528854267098848927 | -3.568105058562e-3 |
| 530 | `mpk_seam_5` | 23.532365554880189139 | 23.525757668744916601 | -6.607886135273e-3 |
| 577 | `mpk_hot_5` | 24.532365557227169901 | 24.526057369540529144 | -6.308187686641e-3 |

Both the official literal transport and the actual `pandas.read_csv` /
`to_csv` transport pass against the refreshed merge artifact.

The remaining gate rows did not certify a strict win: `n=367` missed by
`+1.839733e-7`, while `n=902` had no valid polished candidate (minimum final
penetration `1.59e-4`). A first breadth extension then missed at `n=791`
(`+2.405968e-5`) and `n=908` (`+1.854286e-5`) before the work was redirected
to nearby rows. In the adjacent family, `528`, `529`, `531`, `532`, `533`,
`576`, `578`, `842`, `900`, and `901` produced certified candidates that were
not strict wins, or failed to polish; the exact per-`n` values remain in
`docs/`. The remaining D=1 batch missed at `962` (`+3.367623e-6`), `786`
(`+2.724334e-7`), and `785` (`+2.730591e-7`). A fresh 48-start `842` sweep at
expansions `0.044` and `0.050` produced no valid polished candidate (minimum
penetration `1.46e-4`).

## Run

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
/Users/Guestly/Documents/Codex/2026-11-12/https-www-kaggle-com-competitions-squares/squares-team/.venv/bin/python \
  -u poc/swarm_masked_melt_breadth/poc/run_gate.py \
  --targets 367 902 530 --threads 4 --top-k 6 \
  --out-json poc/swarm_masked_melt_breadth/docs/gate_results.json
```

The 48-start `n=842` expansion sweep is reproducible with:

```bash
python -u poc/swarm_masked_melt_breadth/poc/run_gate.py \
  --targets 530 577 842 --threads 4 --top-k 6 \
  --seed-offset 84200 --seed-waves 2 --l-expand-sweep 0.044 0.050
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
