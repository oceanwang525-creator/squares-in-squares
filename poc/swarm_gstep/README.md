# Hypothesis H79 (`gstep`): Cross-Remainder Staircase Surgery + $K = 2$ Diagonal Pinwheel Re-Assembly

## Mechanism
- **Cross-remainder `g`-step staircase surgery** splices 1 periodic tilted strip (and matching border AA squares) into an ultra-thin `g = +6` brick (`B6_10x21_mw972_walk`, `f = 0.656513`; `B6_10x20_mw972_top`, `f = 0.686486`; `B6_9x21_peel_left`, `f = 0.692574`) to increment `(w, h, g) -> (w, h+1, g+1)`:
  - `B7_10x22_vstrip` (`10x22, g = +7`): `f = 0.688507` (`-0.033033` thinner than prior best `B7_19x11_n854` `0.721540`).
  - `B7_10x21_vs_k5` (`10x21, g = +7`): `f = 0.718637`.
  - `B7_9x22_vs_k3` (`9x22, g = +7`): `f = 0.715826` (`peel_left` of `B7_10x22_vstrip`: `0.715844`).
- **$K = 2$ Diagonal Pinwheel Re-Assembly (`BL` + `TR`)**:
  - **Zero-overlap (`w1+w2+h1+h2 <= 2m-1`)**: `B6_9x18_n796_TL` (`9x18, g=+6, f=0.705887`) + `B7_10x22_vstrip^T` (`22x10, g=+7, f=0.688507`) yields a direct zero-overlap packing for `n=974` (`m=31, r=+13`).
  - **1-corner-overlap seam melt (`w1+w2 == m, h1+h2 == m`)**: `B7_9x22_vs_k3` (`9x22, g=+7, f=0.715826`) + `B7_9x22_vs_k3^T` (`22x9, g=+7`) + `cpp_batch_melt` + `joint_xytheta_slp` resolves the single corner overlap for `n=975` (`m=31, r=+14`).

## Certified Wins vs `origin/main` (`8c623fc2`)
Verified in BOTH `literal` and `pandas` modes via `poc/auto_union/poc/union.py` (`literal == pandas`):

| `n` | `(m, r)` | Incumbent `L` | New Certified `L` (`lit == pan`) | `dL` | `ds = d(L^2/n)` | Construction |
|---:|:---:|---:|---:|---:|---:|:---|
| `974` | `(31, +13)` | `31.707106781186799...` (`merged.csv`) | `31.70588674578704613...` | **`-1.220035e-03`** | **`-7.943131e-05`** | `B6_9x18_n796_TL + B7_10x22_vstrip^T` |
| `975` | `(31, +14)` | `31.7215400022594658...` (`poc/dual-pocket/final.csv`) | `31.71877369634442420...` | **`-2.766306e-03`** | **`-1.799952e-04`** | `B7_9x22_vs_k3 + B7_9x22_vs_k3^T` (`id+flipx` corner melt) |

- **Total competition `ds`**: **`-2.594265e-04`** (`2/2` strict wins, `0` ties, `0` losses).

## Verification
- `poc/baseline/poc/.venv/bin/pytest poc/swarm_gstep/tests/test_gstep.py -v`
- `BASE_SHA=$(git rev-parse origin/main) HEAD_SHA=$(git rev-parse HEAD) python3 .github/scripts/check_final_csv.py`
- `poc/auto_union/poc/union.py` dual `literal == pandas` certification on `poc/swarm_gstep/final.csv`
