# Hypothesis H79 (`gstep`): Cross-Remainder Staircase Surgery, $K = 2$ Diagonal Pinwheels & Round-3 `H117 + H119` Ratchet

## Mechanism
- **Round 1 & 2 (`PR #70`, `PR #75`)**:
  - Cross-remainder `g`-step staircase surgery (`vstrip` / peel) + $K = 2$ diagonal pinwheels (`BL + TR`, 0-overlap and 1-corner `rot180` seam melt + `lpslack`), landing `n=974` (`31.705887`), `n=975` (`31.718774`), `n=912` (`30.692586`, `dL = -6.350e-03`), `n=853` (`29.705882`).
- **Round 3 (`H117 + H119` Ratchet, `poc/swarm_gstep/poc/run_ratchet_r3.py`)**:
  - Combines **Stage-0 `pslp2.squeeze` + `row980.py` reduced active-arm model** (`poc/row_ratchet_980/poc/pslp2.py`, kink-safe trust-scaled free-angle HiGHS SLP) with **alternating `mechwalk` mode-walk / C++ masked-melt (`poc/swarm_mechwalk`) -> `pslp2` ratchet** (passing even near-feasible `mechwalk` candidates into `pslp2`) + `lpslack` polish.
  - Across `origin/main` (`0e74cfc`), `37` targets strictly improved (`total ds_L2n = -5.580675e-05`, archived in `docs/r3_audit_0e74cfc.json`). To keep `poc/swarm_gstep/final.csv` compact (`933,873 B < 1 MB`), the top `8` wins (`99.30%` of total `ds_L2n`, `total ds_L2n = -5.541742e-05`, `total ds = -2.727683e-05`) are packaged in `final.csv`.

## Round 3 Certified Wins vs `origin/main` (`0e74cfc`)
Verified in BOTH `literal` and `pandas` modes via `poc/auto_union/poc/union.py` (`literal == pandas`):

| `n` | Incumbent `L` (`0e74cfc`) | Incumbent Source | New Certified `L` (`lit == pan`) | `dL` | `ds_L2n = d(L^2/n)` | Method |
|---:|---:|:---|---:|---:|---:|:---|
| `973` | `31.67387279556432697...` | `poc/swarm_peelstep/final.csv` | `31.67348444420338412...` | **`-3.883514e-04`** | **`-2.528369e-05`** | `stage0_pslp2` |
| `854` | `29.71947506011512395...` | `poc/swarm_peelstep/final.csv` | `29.71927262972981038...` | **`-2.024304e-04`** | **`-1.408924e-05`** | `stage0_pslp2` |
| `924` | `30.85445749417608715...` | `merged.csv` | `30.85428944045241211...` | **`-1.680537e-04`** | **`-1.122336e-05`** | `stage0_pslp2` |
| `975` | `31.71587849865607013...` | `merged.csv` | `31.71584144380572014...` | **`-3.705485e-05`** | **`-2.410721e-06`** | `c1_mmelt_r6_b0_m2_t4e-04_p_0.008+pslp2` |
| `748` | `27.82678685346119222...` | `merged.csv` | `27.82676346600943321...` | **`-2.338745e-05`** | **`-1.740100e-06`** | `c0_rejam_r2_b0_m3_t2e-04_p_0.008+pslp2` |
| `795` | `28.69977978646164908...` | `poc/swarm_peelstep/final.csv` | `28.69977555982119058...` | **`-4.226640e-06`** | **`-3.051664e-07`** | `stage0_pslp2` |
| `793` | `28.65655786244277352...` | `merged.csv` | `28.65655523699503001...` | **`-2.625448e-06`** | **`-1.897511e-07`** | `stage0_pslp2` |
| `910` | `30.65635249739734839...` | `merged.csv` | `30.65634989417666023...` | **`-2.603221e-06`** | **`-1.753961e-07`** | `stage0_pslp2` |

- **Round 3 Packaged Total (`8/8` strict wins)**: **`ds = -2.727683e-05`**, **`ds_L2n = -5.541742e-05`**, `0` ties, `0` losses (`final.csv` = `933,873 B`).
- Prior Round 1/2 wins (`n=853, 912, 974`) are superseded in `origin/main` (`0e74cfc`) at strictly tighter `L`.

## Verification
- `poc/baseline/poc/.venv/bin/pytest poc/swarm_gstep/tests/test_gstep.py -v`
- `BASE_SHA=$(git rev-parse origin/main) HEAD_SHA=$(git rev-parse HEAD) poc/baseline/poc/.venv/bin/python .github/scripts/check_final_csv.py`
- `poc/auto_union/poc/union.py` dual `literal == pandas` certification on `poc/swarm_gstep/final.csv`
