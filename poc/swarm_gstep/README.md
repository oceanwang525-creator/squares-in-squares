# Hypothesis H79 (`gstep`): Cross-Remainder Staircase Surgery, $K = 2$ Diagonal Pinwheels & Round-3 `H117 + H119` Ratchet

## Mechanism
- **Round 1 & 2 (`PR #70`, `PR #75`)**:
  - Cross-remainder `g`-step staircase surgery (`vstrip` / peel) + $K = 2$ diagonal pinwheels (`BL + TR`, 0-overlap and 1-corner `rot180` seam melt + `lpslack`), landing `n=974` (`31.705887`), `n=975` (`31.718774`), `n=912` (`30.692586`, `dL = -6.350e-03`), `n=853` (`29.705882`).
- **Round 3 (`H117 + H119` Ratchet, `poc/swarm_gstep/poc/run_ratchet_r3.py`)**:
  - Combines **Stage-0 `pslp2.squeeze` + `row980.py` reduced active-arm model** (`poc/row_ratchet_980/poc/pslp2.py`, kink-safe trust-scaled free-angle HiGHS SLP) with **alternating `mechwalk` mode-walk / C++ masked-melt (`poc/swarm_mechwalk`) -> `pslp2` ratchet** (passing even near-feasible `mechwalk` candidates into `pslp2`) + `lpslack` polish.
- **Round 3 PR5 (Final Ratchet on Newly Merged Breakthroughs, `docs/r3_audit_10a15e8.json`)**:
  - Applying `run_ratchet_r3.py` (`H117 + H119`) to newly merged `origin/main` breakthroughs achieved **`32` strict wins** vs `origin/main` (`10a15e8`).
  - To keep `poc/swarm_gstep/final.csv` `< 1 MB` (`940,184 B`), the top **`9`** strict wins (`total ds = -3.771754e-05`, `total ds_L2n = -7.665788e-05`) are packaged in `final.csv`.

## Summary: What Worked vs Failed (Rounds 1–3, Retiring per `#717`/`#725`)
- **WORKED**:
  - Cross-remainder `g`-step staircase surgery + $K = 2$ diagonal pinwheels + `lpslack` (Rounds 1–2, `PR #70`, `PR #75`, `n=912` `dL=-6.350e-03`, `974, 975, 853`).
  - Alternating `mechwalk` (incl. near-feasible mode-walk/masked-melt seeds) -> kink-safe `pslp2` / `row980` free-angle SLP squeeze + `lpslack` polish (Round 3 PR3 `PR #107`, PR4 `PR #119`, PR5). Noticeably strong whenever freshly merged pinwheel/pocket packings (`857, 861, 971, 973, 793, 924, 911`) still had `1e-4..7e-4` un-relaxed angular slack.
- **FAILED**:
  - `g`-step staircase surgery without breaking brick count (`B6/B7_synth`) stalled against newer multi-angle/asymmetric `vstrip`/`band3ext` rebuilds once simple `g = +5/+6` corridors saturated.
  - Micro-polish ratcheting on already-saturated `GROUP_B`/`GROUP_C` targets (`r3_p2_w2`: `0/14` strict wins across `908, 969, 791, 847, 872, 813, 818, 761, 650, 595, 550, 859, 858, 856`) yielded `< 1e-10` crumbs; per `#717` course correction, `gstep` retires after PR5 so slots can move to Big-Move family builders (`H127–H132`).

## Round 3 PR5 Certified Wins vs `origin/main` (`10a15e8`)
Verified in BOTH `literal` and `pandas` modes via `poc/auto_union/poc/union.py` (`literal == pandas`):

| `n` | Incumbent `L` (`10a15e8`) | Incumbent Source | New Certified `L` (`lit == pan`) | `dL` | `ds_L2n = d(L^2/n)` | Method |
|---:|---:|:---|---:|---:|---:|:---|
| `857` | `29.76046248159143723...` | `merged.csv` | `29.76009453346141528...` | **`-3.679481e-04`** | **`-2.555482e-05`** | `stage0_pslp2` |
| `861` | `29.81608964380292025...` | `merged.csv` | `29.81576455894061614...` | **`-3.250849e-04`** | **`-2.251500e-05`** | `stage0_pslp2` |
| `971` | `31.63866488650004477...` | `merged.csv` | `31.63843149871709008...` | **`-2.333878e-04`** | **`-1.520917e-05`** | `stage0_pslp2` |
| `924` | `30.85421229571784210...` | `merged.csv` | `30.85413579354389399...` | **`-7.650217e-05`** | **`-5.109116e-06`** | `stage0_pslp2` |
| `911` | `30.67573013467941021...` | `merged.csv` | `30.67566774641112542...` | **`-6.238827e-05`** | **`-4.201545e-06`** | `stage0_pslp2` |
| `748` | `27.82675316760123620...` | `poc/swarm_highdpocket/final.csv` | `27.82672883904211313...` | **`-2.432856e-05`** | **`-1.810119e-06`** | `stage0_pslp2` |
| `795` | `28.69887184145230144...` | `merged.csv` | `28.69885564779231656...` | **`-1.619366e-05`** | **`-1.169156e-06`** | `stage0_pslp2` |
| `530` | `23.52536776674761989...` | `poc/swarm_liftpolish3/final.csv` | `23.52535947615029019...` | **`-8.290597e-06`** | **`-7.359974e-07`** | `stage0_pslp2` |
| `447` | `21.65634899177884959...` | `poc/swarm_liftpolish3/final.csv` | `21.65634534903779009...` | **`-3.642741e-06`** | **`-3.529685e-07`** | `stage0_pslp2` |

- **Round 3 PR5 Packaged Total (`9/9` strict wins)**: **`ds = -3.771754e-05`**, **`ds_L2n = -7.665788e-05`**, `0` ties, `0` losses (`final.csv` = `940,184 B`).
- Prior Round 1/2/3 wins (incl. PR4 `[582, 599, 687, 793, 799, 863, 934, 973, 974]`) are merged and preserved in `origin/main` (`10a15e8`) at equal or strictly tighter `L`.

## Verification
- `poc/baseline/poc/.venv/bin/pytest poc/swarm_gstep/tests/test_gstep.py -v`
- `BASE_SHA=$(git rev-parse origin/main) HEAD_SHA=$(git rev-parse HEAD) poc/baseline/poc/.venv/bin/python .github/scripts/check_final_csv.py`
- `poc/auto_union/poc/union.py` dual `literal == pandas` certification on `poc/swarm_gstep/final.csv`
