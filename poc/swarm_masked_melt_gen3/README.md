# H131 third-generation masked melt

## Idea

H128 changed the SAT contact graph at `n=634,695,861,926` and produced four
strict upstream wins. This POC starts from those H128 child geometries and
performs one further masked void/contact/seam melt generation. It uses fresh
deterministic RNG streams, expands the seed box by `0.042`, widens the local
radius to `5.0..8.7`, increases the seed mix to `12 void + 10 contact + 8
seam`, and uses a five-stage box retreat.

Every candidate is polished with L-BFGS, joint `xytheta` SLP, and fixed-angle
LP. A candidate is exported only when the canonical CSV certifies strictly
below both the required upstream artifact and H128 in the official literal
check and the actual `pandas.read_csv` / `to_csv` transport.

## Baselines

- `upstream/main`: `6bfe2c48f33bc3b645dba8489297c4720f9d0fe8`
- `upstream/merge`: `7c93cb082e10ba9125156a9f22653f6131552ec8`
- Required upstream artifact: `merged-22.15277788725.csv`
- H128 child POC: `poc/swarm_masked_melt_bootstrap/`

## Gate

Run `n=861,926,695` first. Extend to `n=634` only if the gate remains
productive.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python -u poc/swarm_masked_melt_gen3/poc/run_gate.py \
  --targets 861 926 695 --threads 4 --top-k 8 \
  --out-json poc/swarm_masked_melt_gen3/docs/gate_results.json
```

Verify the exported groups against both baselines:

```bash
python poc/swarm_masked_melt_gen3/poc/verify_export.py \
  > poc/swarm_masked_melt_gen3/docs/verification.json
python .github/scripts/check_final_csv.py
```

## Result

Two complete groups strictly beat both required baselines in the official
literal check and the actual pandas roundtrip:

| n | start label | upstream L | H128 L | H131 L | dL vs upstream | dL vs H128 |
|---:|---|---:|---:|---:|---:|---:|
| 695 | `mpk_seam_7` | 26.831780320347433 | 26.831690049211594 | 26.831633017584601 | `-1.473028e-4` | `-5.703163e-5` |
| 861 | `mpk_void_8` | 29.817033759741868 | 29.816317663506829 | 29.816302955140713 | `-7.308046e-4` | `-1.470837e-5` |

The gate also tested `n=926`; its best candidate was `2.119e-6` above H128,
so it was not exported. The permitted one-target extension tested `n=634`;
its best candidate was `4.683e-6` above H128 and was likewise not exported.

`final.csv` therefore contains only the complete `n=695` and `n=861` groups.
Independent verification is recorded in `docs/verification.json`.
