# Hypothesis H79 (`gstep`): Cross-Remainder Staircase Surgery, $K = 2$ Diagonal Pinwheels & Round-3 `H117 + H119` Ratchet

## Mechanism
- **Round 1 & 2 (`PR #70`, `PR #75`)**:
  - Cross-remainder `g`-step staircase surgery (`vstrip` / peel) + $K = 2$ diagonal pinwheels (`BL + TR`, 0-overlap and 1-corner `rot180` seam melt + `lpslack`), landing `n=974` (`31.705887`), `n=975` (`31.718774`), `n=912` (`30.692586`, `dL = -6.350e-03`), `n=853` (`29.705882`).
- **Round 3 (`H117 + H119` Ratchet, `poc/swarm_gstep/poc/run_ratchet_r3.py`)**:
  - Combines **Stage-0 `pslp2.squeeze` + `row980.py` reduced active-arm model** (`poc/row_ratchet_980/poc/pslp2.py`, kink-safe trust-scaled free-angle HiGHS SLP) with **alternating `mechwalk` mode-walk / C++ masked-melt (`poc/swarm_mechwalk`) -> `pslp2` ratchet** (passing even near-feasible `mechwalk` candidates into `pslp2`) + `lpslack` polish.
- **Round 3 PR4 (`H117 + H119` Ratchet Scale-Up, `docs/r3_audit_9856cdf.json`)**:
  - Full `mechwalk <-> pslp2/row980` ratchet (`run_ratchet_r3.py`) across Priority-2 targets plus remaining un-packaged Stage-0 wins achieved `24` strict wins vs `origin/main` (`total ds_L2n = -2.107632e-05`), including `-1.752e-04` deeper on `n=973` and `-1.304e-04` deeper on `n=793`.
  - To keep `poc/swarm_gstep/final.csv` `< 1 MB` (`958,660 B`), the top `9` strict wins (`99.95%` of total `ds_L2n`, `total ds = -1.036235e-05`, `total ds_L2n = -2.106520e-05`) are packaged in `final.csv`.

## Round 3 PR4 Certified Wins vs `origin/main` (`089a9d9`)
Verified in BOTH `literal` and `pandas` modes via `poc/auto_union/poc/union.py` (`literal == pandas`):

| `n` | Incumbent `L` (`089a9d9`) | Incumbent Source | New Certified `L` (`lit == pan`) | `dL` | `ds_L2n = d(L^2/n)` | Method |
|---:|---:|:---|---:|---:|---:|:---|
| `973` | `31.67348444420338412...` | `merged.csv` | `31.67330922358059414...` | **`-1.752206e-04`** | **`-1.140767e-05`** | `c0_rejam_r1_b0_m2_t1e-03_p_0.008+pslp2` |
| `793` | `28.65655523699503000...` | `merged.csv` | `28.65642481081007284...` | **`-1.304262e-04`** | **`-9.426372e-06`** | `c0_rejam_r1_b0_m0_t2e-04_p_0.008+pslp2` |
| `582` | `24.63394768804568668...` | `poc/swarm_upwardcasc/final.csv` | `24.63394629421225629...` | **`-1.393833e-06`** | **`-1.179918e-07`** | `stage0_pslp2` |
| `974` | `31.69258193382914020...` | `merged.csv` | `31.69258085281143506...` | **`-1.081018e-06`** | **`-7.034957e-08`** | `stage0_pslp2` |
| `799` | `28.75429480467201661...` | `merged.csv` | `28.75429445690715222...` | **`-3.477649e-07`** | **`-2.503062e-08`** | `stage0_pslp2` |
| `863` | `29.84268556024288615...` | `merged.csv` | `29.84268546278836008...` | **`-9.745453e-08`** | **`-6.739988e-09`** | `stage0_pslp2` |
| `934` | `30.96241335818630972...` | `poc/swarm_corrcasc/final.csv` | `30.96241329582811218...` | **`-6.235820e-08`** | **`-4.134390e-09`** | `stage0_pslp2` |
| `599` | `24.91681698849889529...` | `merged.csv` | `24.91681693920973313...` | **`-4.928916e-08`** | **`-4.100598e-09`** | `stage0_pslp2` |
| `687` | `26.71037483198452602...` | `merged.csv` | `26.71037479582138648...` | **`-3.616314e-08`** | **`-2.812026e-09`** | `stage0_pslp2` |

- **Round 3 PR4 Packaged Total (`9/9` strict wins)**: **`ds = -1.036235e-05`**, **`ds_L2n = -2.106520e-05`**, `0` ties, `0` losses (`final.csv` = `958,660 B`).
- Prior Round 1/2/3 wins (`n=748, 795, 853, 854, 910, 912, 924, 975`) are merged and preserved in `origin/main` (`089a9d9`) at equal or strictly tighter `L`.

## Verification
- `poc/baseline/poc/.venv/bin/pytest poc/swarm_gstep/tests/test_gstep.py -v`
- `BASE_SHA=$(git rev-parse origin/main) HEAD_SHA=$(git rev-parse HEAD) poc/baseline/poc/.venv/bin/python .github/scripts/check_final_csv.py`
- `poc/auto_union/poc/union.py` dual `literal == pandas` certification on `poc/swarm_gstep/final.csv`
