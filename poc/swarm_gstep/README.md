# Hypothesis H79 (`gstep`): Cross-Remainder Staircase Surgery + $K = 2$ Diagonal Pinwheel Re-Assembly

## Mechanism
- **Cross-remainder `g`-step staircase surgery** splices 1 periodic tilted strip (and matching border AA squares) into an ultra-thin `g = +6` brick (`B6_10x21_mw972_walk`, `f = 0.656513`; `B6_10x20_mw972_top`, `f = 0.686486`; `B6_9x21_peel_left`, `f = 0.692574`) and peels/splices `g = +7, +8` derivatives:
  - `B7_10x22_vstrip` (`10x22, g = +7`): `f = 0.688507` (`-0.033033` thinner than prior `B7_19x11_n854` `0.721540`).
  - `B7_10x21_vs_k5` (`10x21, g = +7`): `f = 0.718637`.
  - `B7_9x22_vs_k3` (`9x22, g = +7`): `f = 0.715826` (`peel_left` of `B7_10x22_vstrip`: `0.715844`).
  - `B7_9x21_vs_bottom` (`9x21, g = +7`): `f = 0.724370`.
  - `B8_11x20_vs_k4` (`11x20, g = +8`): `f = 0.733991`.
  - `B6_9x21_peel_left` (`9x21, g = +6`): `f = 0.692574`.
  - `B6_23x8_left` (`23x8, g = +6`): `f = 0.686630`.
- **$K = 2$ Diagonal Pinwheel Re-Assembly (`BL` + `TR`)**:
  - **Round 1 (merged in PR #70)**: `n=974` (`31.705886746`, `dL = -1.220035e-03`) via zero-overlap `B6_9x18_n796_TL + B7_10x22_vstrip^T`; `n=975` (`31.718773696`, `dL = -2.766306e-03`) via 1-corner seam melt `B7_9x22_vs_k3 + B7_9x22_vs_k3^T`.
  - **Round 2 (this branch, vs `origin/main` `cf838aa`)**:
    - `n=912` (`m=30, r=+12`): `B6_9x21_peel_left` (`9x21, g=+6, f=0.692574`) + `B6_9x21_peel_left^T` (`21x9, g=+6`) via `rot180_id` 1-corner seam melt + `joint_xytheta_slp` + `lpslack`.
    - `n=853` (`m=29, r=+12`): `B6_9x18_n796_cut` (`9x18, g=+6, f=0.705882`) + `B6_10x20_mw972_top^T` (`20x10, g=+6, f=0.686486`) zero-overlap direct + `lpslack`.

## Round 2 Certified Wins vs `origin/main` (`cf838aa`)
Verified in BOTH `literal` and `pandas` modes via `poc/auto_union/poc/union.py` (`literal == pandas`):

| `n` | `(m, r)` | Incumbent `L` (`cf838aa`) | New Certified `L` (`lit == pan`) | `dL` | `ds_L2n = d(L^2/n)` | Construction |
|---:|:---:|---:|---:|---:|---:|:---|
| `912` | `(30, +12)` | `30.69893632287430157...` | `30.69258631912348312...` | **`-6.350004e-03`** | **`-4.274522e-04`** | `B6_9x21_peel_left + B6_9x21_peel_left^T` (`rot180_id` corner melt) |
| `853` | `(29, +12)` | `29.70588237679204204...` | `29.70588237369941039...` | **`-3.092632e-09`** | **`-2.154029e-10`** | `B6_9x18_n796_cut + B6_10x20_mw972_top^T` (zero-overlap + `lpslack`) |

- **Round 2 Total `ds_L2n`**: **`-4.274524e-04`** (`2/2` strict wins, `0` ties, `0` losses).

## Verification
- `poc/baseline/poc/.venv/bin/pytest poc/swarm_gstep/tests/test_gstep.py -v`
- `BASE_SHA=$(git rev-parse origin/main) HEAD_SHA=$(git rev-parse HEAD) python3 .github/scripts/check_final_csv.py`
- `poc/auto_union/poc/union.py` dual `literal == pandas` certification on `poc/swarm_gstep/final.csv`
