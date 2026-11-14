# H131 third-generation masked melt

## Idea

H128 changed the SAT contact graph at `n=634,695,861,926` and produced four
strict upstream wins. This POC starts from those H128 child geometries and
performs further masked void/contact/seam melt generations. The first pass
uses fresh deterministic RNG streams, expands the seed box by `0.042`,
widens the local radius to `5.0..8.7`, increases the seed mix to `12 void +
10 contact + 8 seam`, and uses a five-stage box retreat. A second pass on
adjacent rows uses another stream pair and a `0.046` expansion.

Every candidate is polished with L-BFGS, joint `xytheta` SLP, and fixed-angle
LP. A candidate is exported only when the canonical CSV certifies strictly
below both the required upstream artifact and H128 in the official literal
check and the actual `pandas.read_csv` / `to_csv` transport.

## Baselines

- `upstream/main`: `331b307fbf67fc721609b121b9606b399b2359b9`
- `upstream/merge`: `2554d5b96b64b6d633daf922446331d8f35044da`
- Required upstream artifact: `merged-22.15258085489.csv`
- Tracking file SHA256:
  `c9e5398775e31b536cbfa1519ab9c6066c95f98edf1d7bc9bec3a54a85cafd9e`
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

python -u poc/swarm_masked_melt_gen3/poc/run_adjacent.py \
  --targets 633 634 635 696 697 860 862 863 --threads 4 --top-k 8 \
  --out-json poc/swarm_masked_melt_gen3/docs/adjacent_results.json
```

Verify the exported groups against both baselines:

```bash
python poc/swarm_masked_melt_gen3/poc/verify_export.py \
  > poc/swarm_masked_melt_gen3/docs/verification.json
python .github/scripts/check_final_csv.py
```

The runner loads its seed functions from this POC by absolute path; the test
also uses a unique importlib module name so combined pytest collection cannot
be redirected by another POC's `pocket_seeds.py`.

## Result

Three complete groups strictly beat both required baselines in the official
literal check and the actual pandas roundtrip:

| n | start label | upstream L | H128 L | H131 L | dL vs upstream | dL vs H128 |
|---:|---|---:|---:|---:|---:|---:|
| 634 | `mpk_void_11` | 25.688381146233741 | 25.688376088157854 | 25.688369972665937 | `-1.117357e-5` | `-6.115492e-6` |
| 695 | `mpk_seam_7` | 26.831780320347433 | 26.831690049211594 | 26.831633017584601 | `-1.473028e-4` | `-5.703163e-5` |
| 861 | `mpk_void_8` | 29.817033759741868 | 29.816317663506829 | 29.816302955140713 | `-7.308046e-4` | `-1.470837e-5` |

The gate also tested `n=926`; its best candidate was `2.119e-6` above H128,
so it was not exported. The first `n=634` extension missed H128, but the
adjacent-row pass found `mpk_void_11` and exported it. The other adjacent
targets (`633,635,696,697,860,862,863`) did not beat their current rows.

`final.csv` therefore contains only the complete `n=634`, `n=695`, and
`n=861` groups. All three remain strict wins against
`merged-22.15258085489.csv`; no row was superseded by the new merge artifact.
Independent verification is recorded in `docs/verification.json`.

## Files

- `poc/run_gate.py`: initial H131 gate and strict dual-baseline exporter.
- `poc/run_adjacent.py`: fresh-stream adjacent-row pass preserving incumbents.
- `poc/pocket_seeds.py`: third-generation masked void/contact/seam starts.
- `poc/verify_export.py`: pins and verifies the current merge artifact hash in
  both certification transports.
- `tests/test_masked_melt_gen3.py`: isolated seed-import, shape, and schema
  checks.
