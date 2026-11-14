# Hypothesis H107: Diagonal `(m+1, r+k) -> (m, r)` Arm-Tip + L-Border Peel Cascade (`swarm_corrcasc`)

## Summary (Gate PR1 — `3/8` Big-Move Certified Wins)

Peeling constant-`r` `(m+1, r) -> (m, r)` (`dn = 2m+1`) often fails because the donor's tilted arm extends into the outer 1-unit L-border, leaving `k in {1, 2, 3}` protruding tilted squares (`2m+1+k` total intersecting squares). Mapping **diagonally `(m+1, r+k) -> (m, r)`** (`dn = 2m+1+k`) removes the outer 1-unit L-border **plus** the protruding arm-tip square(s), then applies C++ masked penalty melt (`batch_masked_melt_solve`), `joint_xytheta_slp`, 4-round SVD `mechwalk_compress`, and `lpslack`.

### Gate Results (`poc/swarm_corrcasc/docs/gate_results.json`, vs `origin/main` `772dfc2`)

| Target `n` <- Donor `n_d` | `(m, r, k)` | `L_inc` (`merged.csv`) | `L_new` | `dL` | Method |
|---:|---:|:---:|:---:|:---:|:---|
| **`687 <- 741`** | `(26, 11, 1)` | `26.717912747968` | **`26.710374831985`** | **`-7.537916e-03`** | `peel1_741_BR_k0.985_dropProt_clamp` + `mechwalk` + `lpslack` |
| **`758 <- 814`** | `(27, 29, 1)` | `27.958641416318` | **`27.957108739080`** | **`-1.532677e-03`** | `diag_814_L_TL_bbox_raw` + `mechwalk` + `lpslack` |
| **`876 <- 936`** | `(29, 35, 1)` | `29.997559465372` | **`29.996909189847`** | **`-6.502755e-04`** | `peel1_936_BR_k0.985_dropProt_clamp` + `mechwalk` + `lpslack` |
| `383 <- 423` | `(19, 22, 1)` | `20.000000000000` | `20.000008952094` | `+8.952e-06` | — |
| `650 <- 702` | `(25, 25, 1)` | `25.932813788883` | `25.939683680827` | `+6.870e-03` | — |
| `931 <- 993` | `(30, 31, 1)` | `30.931531120167` | `30.942119967037` | `+1.059e-02` | — |
| `602 <- 652` | `(24, 26, 1)` | `24.970896866569` | `24.982871525501` | `+1.197e-02` | — |
| `420 <- 462` | `(20, 20, 1)` | `20.960692680488` | `20.974624089275` | `+1.393e-02` | — |

### Audit (`poc/swarm_corrcasc/poc/audit_diagonal_pairs.py`)
- Scanned `4,473` 1-step `(m+1, r+k)->(m, r)` (`k=1,2,3`) and 2-step `(m+2, r+k)->(m, r)` (`k=2,3,4`) pairs across `m = 12..30`.
- `2,358` pairs satisfy `dL_raw <= +0.005`; `2,267` also have boundary/seam protrusion/overlap `score <= 0.12`, including **`224` pairs with `dL_raw < -1e-4`**.

### Verification
- All 3 wins in `poc/swarm_corrcasc/final.csv` (`687, 758, 876`) pass dual certification (`literal == pandas == kaggle_roundtrip`).
- Structure & tests: `.github/scripts/check_final_csv.py` PASS; `pytest poc/swarm_corrcasc/tests/test_corrcasc.py -v` PASS.
