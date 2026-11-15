# Hypothesis H107: Diagonal `(m+1, r+k) -> (m, r)` Arm-Tip + L-Border Peel Cascade (`swarm_corrcasc`)

## Summary

Peeling constant-`r` `(m+1, r) -> (m, r)` (`dn = 2m+1`) often fails because the donor's tilted arm extends into the outer 1-unit L-border, leaving `k in {1, 2, 3}` protruding tilted squares (`2m+1+k` total intersecting squares). Mapping **diagonally `(m+1, r+k) -> (m, r)`** (`dn = 2m+1+k`) removes the outer 1-unit L-border **plus** the protruding arm-tip square(s), then applies C++ masked penalty melt (`batch_masked_melt_solve`), `joint_xytheta_slp`, SVD `mechwalk_compress`, and `lpslack`.

### PR3 Scale-Up Result (`1` Big-Move Certified Win vs `origin/main` `b3ac0d7`)

| Target `n` <- Donor `n_d` | `(m, r, k)` | `L_inc` (`merged.csv`) | `L_new` (`literal == pandas`) | `dL` | Method |
|---:|---:|:---:|:---:|:---:|:---|
| **`815 <- 873`** | `(28, 31, 1)` | `28.965965683761` | **`28.964263069216`** | **`-1.702615e-03`** | `diag_873_L_TL_bbox_raw_R8.4_s0` + `mechwalk_compress` + `lpslack` |

- PR2 (`n=934 <- 996`, `30.962413358186`, `dL = -1.729791e-03`) merged in PR #96 (`9bfcaf3`) and is preserved strictly tighter in `origin/main:merged.csv` (`30.962413301783`); `win_n934.json` was removed from PR3 so `poc/swarm_corrcasc/final.csv` exports strictly `n=815`.
- PR1 (`687`, `758`, `876`) merged in PR #89 (`1f337f9`) and is preserved in `origin/main:merged.csv`.

### PR2 Scale-Up Result (`9bfcaf3`, PR #96)

| Target `n` <- Donor `n_d` | `(m, r, k)` | `L_inc` (`cc58c7e`) | `L_new` (`literal == pandas`) | `dL` | Method |
|---:|---:|:---:|:---:|:---:|:---|
| **`934 <- 996`** | `(30, 34, 1)` | `30.964143149166` | **`30.962413358186`** | **`-1.729791e-03`** | `diag_996_L_BR_bbox_raw` basin + 3-round `mechwalk_compress` + `lpslack` |

### PR1 Gate Results (`poc/swarm_corrcasc/docs/gate_results.json`, vs `origin/main` `772dfc2`)

| Target `n` <- Donor `n_d` | `(m, r, k)` | `L_inc` (`772dfc2`) | `L_new` | `dL` | Method |
|---:|---:|:---:|:---:|:---:|:---|
| **`687 <- 741`** | `(26, 11, 1)` | `26.717912747968` | **`26.710374831985`** | **`-7.537916e-03`** | `peel1_741_BR_k0.985_dropProt_clamp` + `mechwalk` + `lpslack` |
| **`758 <- 814`** | `(27, 29, 1)` | `27.958641416318` | **`27.957108739080`** | **`-1.532677e-03`** | `diag_814_L_TL_bbox_raw` + `mechwalk` + `lpslack` |
| **`876 <- 936`** | `(29, 35, 1)` | `29.997559465372` | **`29.996909189847`** | **`-6.502755e-04`** | `peel1_936_BR_k0.985_dropProt_clamp` + `mechwalk` + `lpslack` |
| `383 <- 423` | `(19, 22, 1)` | `20.000000000000` | `20.000008952094` | `+8.952e-06` | — |
| `650 <- 702` | `(25, 25, 1)` | `25.932813788883` | `25.939683680827` | `+6.870e-03` | — |
| `931 <- 993` | `(30, 31, 1)` | `30.931531120167` | `30.942119967037` | `+1.059e-02` | — |
| `602 <- 652` | `(24, 26, 1)` | `24.970896866569` | `24.982871525501` | `+1.197e-02` | — |
| `420 <- 462` | `(20, 20, 1)` | `20.960692680488` | `20.974624089275` | `+1.393e-02` | — |

### Scale-Up Sweeps (`run_scale.py`)
- `docs/scale_results.json` (`2` pairs): `934 <- 996` (**`-1.729791e-03`** win), `936 <- 998` (`+4.479e-3`).
- `docs/scale2_results.json` (`6` pairs): `874, 873, 875, 870, 929, 814` (`+4.657e-3` to `+1.450e-2`).
- `docs/scale3_results.json` (`15` pairs): `762 <- 876` (`s=2`, `+2e-6`), `556, 555, 604, 811, 423, 597, 603, 703, 759, 757, 690, 755, 816, 857` (`+4.043e-3` to `+2.346e-2`).
- `docs/scale4_results.json` (`20` pairs): `815 <- 873` (**`-1.702615e-03`** win), `274 <- 344` (`s=2`, `+0.000000`), `308 <- 382` (`s=2`, `+2.851e-3`), `460, 381, 866, 596, 818, 601, 465, 505, 241, 868, 810, 463, 553, 552, 930, 506, 812`.

### Audit (`poc/swarm_corrcasc/poc/audit_diagonal_pairs.py`)
- Scanned `4,473` 1-step `(m+1, r+k)->(m, r)` (`k=1,2,3`) and 2-step `(m+2, r+k)->(m, r)` (`k=2,3,4`) pairs across `m = 12..30`.
- `2,358` pairs satisfy `dL_raw <= +0.005`; `2,267` also have boundary/seam protrusion/overlap `score <= 0.12`, including **`224` pairs with `dL_raw < -1e-4`**.

### Verification
- `poc/swarm_corrcasc/final.csv` (`n=815`) passes dual certification (`literal == pandas == kaggle_roundtrip`).
- Structure & tests: `.github/scripts/check_final_csv.py` PASS; `pytest poc/swarm_corrcasc/tests/test_corrcasc.py -v` PASS.
