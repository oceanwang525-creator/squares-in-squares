# Swarm `brickslp`: H17 (Pinwheel Sub-Brick Joint SLP Compression & Multi-`n` Broadcast) + H21 (Asymmetric Cross-Family Slack Sharing)

This folder implements and evaluates **Hypothesis H17** and **Hypothesis H21** on square packings in `poc/swarm_brickslp/`:

1. **Rectangular Sub-Brick Extraction & Direct SLP + Wall-Melt Compression (`brick_slp.py`)**:
   - Extracts `(w x h, +g)` rectangular tilted sub-bricks from frontier pinwheels (plus `n=910` `(10x21, +6)` and `(19x9, +4)` orientations) and compresses their fractional thickness `f` via:
     - exact fixed-angle LP tightening + multi-stage `joint_xytheta_slp_rect` (`scipy.sparse` + `linprog(highs-ds)`),
     - Idea-6-style boundary wall/void masked C++ melt (`run_cpp_masked_melt` around sub-brick corner/seam pockets) followed by `joint_xytheta_slp_rect`.
2. **Multi-`n` Broadcast & Asymmetric Cross-Family Pinwheel Assembly (`H17` + `H21`)**:
   - Re-assembles `(W1+f1) x (H1+f1)` and `(H2+f2) x (W2+f2)` tilted bricks with an `m x m` unit grid into 5-region pinwheels (`n = m^2 + s1 + s2`), including cross-family pairs where `f1 < f2`.
   - Applies 28-seed pinwheel seam & slack-sharing melts (`mpk_void`, `mpk_hot`, `pw_seam`, `pw_slack`) + `joint_xytheta_slp_tight`.
3. **Exact Portfolio Certification (`frontier_io.py`)**:
   - Loads the true frontier across `poc/pell_corrugation/data/border-scan/merged.csv` (`angle` in radians) and all `poc/*/final.csv` (`deg` in degrees), and certifies every candidate via `poc/auto_union/poc/union.py` in BOTH `literal` and `pandas` modes (`dL <= -1e-9` strict REAL WINs only).

---

## Certified PR #33 REAL WINs (vs `origin/main` `46b17b3`)

All **7** packings in `poc/swarm_brickslp/final.csv` are strictly feasible in `union.check` (`literal == pandas`) and pass `.github/scripts/check_final_csv.py`:

| `n` | Incumbent `L` (`46b17b3`) | Certified `L_lit == L_pd` | `dL` | Method / Seed |
|---:|---:|---:|---:|---|
| **538** | `23.705915748949026...` | `23.705903334892446...` | **`-1.241406e-05`** | `bcast_18x9+6_and_5x13+3_pw_slack_1` |
| **740** | `27.705913740145120...` | `27.705904466463028...` | **`-9.273682e-06`** | `bcast_18x9+6_and_9x17+5_pw_seam_0` |
| **796** | `28.705910015470158...` | `28.705905701929437...` | **`-4.313541e-06`** | `bcast_18x9+6_and_9x18+6_pw_seam_6` |
| **910** | `30.657505410714139...` | `30.657098274364220...` | **`-4.071363e-04`** | `bcast_10x21+6_and_19x9+4_mpk_void_1` |
| **911** | `30.686956998826932...` | `30.684154306663058...` | **`-2.802692e-03`** | `bcast_10x21+6_and_17x9+5_pw_seam_4` |
| **914** | `30.731219928218065...` | `30.731213866641378...` | **`-6.061577e-06`** | `bcast_18x9+6_and_12x20+8_direct` |
| **971** | `31.657505420714138...` | `31.656834521894880...` | **`-6.708988e-04`** | `bcast_10x21+6_and_16x10+4_pw_seam_0` |

*Note:* The 15-minute gate also won `n=639` (`L = 25.776043454897531...`, `dL = -6.753145e-05` via `pw_seam_4` relative to `4478e04`); `n=639` was superseded by PR #32 (`swarm-mechwalk`, `L = 25.774501349...`) and removed from `final.csv` after rebasing on `46b17b3`.

---

## Experimental Log (Wins, Ties, and Misses)

### 1. 15-Minute Falsification Gate (`poc/run_gate.py`, `docs/gate_report.json`)
- **Result: PASS** (`dt = 1019.7s`):
  - **Track 1 (Pinwheel seam melt + SLP)**: `n=639` **WIN** (`dL = -6.753145e-05`, `pw_seam_4`); ties (`|dL| < 1e-12`) on `n=638` (`-5.0e-16`), `n=293` (`-4.0e-16`), `n=691` (`-9.36e-16`), `n=918` (`+1.20e-15`), `n=972` (`0.0`); near-ties on `n=300` (`+3.53e-07`), `n=301` (`+2.70e-06`).
  - **Track 2 (Standalone sub-brick compression)**:
    - `B6_24x8` (`24x8, +6`): `0.678889347 -> 0.678213214` (`df = -6.761334e-04`, `rect_melt_6`).
    - `B10_22x11` (`22x11, +10`): `0.756006961 -> 0.755882519` (`df = -1.244420e-04`, `rect_direct_0`).
    - Ties (`df = 0.0`) on `B8_18x8`, `B7_17x8`, `B6_14x7`, `B5_22x7`.
  - **Track 3 (Naive asymmetric pinwheel assembly without pre-shifted seam)**: `n=973` feasible (`+2.13e-02`), others had seam overlap `4e-16 .. 2.5e-15` (pre-shift needed).

### 2. Phase A: Standalone Rectangular Sub-Brick Compression (`poc/run_scale.py`, `docs/scale_ckpt.json`)
32 multi-seed wall/void masked-melt + `joint_xytheta_slp_rect` trials across 14 sub-bricks:
- **6 compressed sub-bricks**:
  - `B4_19x9` (`19x9, +4`): `0.652100203 -> 0.645697446` (`df = -6.402756e-03`, `rect_wall_17`)
  - `B6_24x8` (`24x8, +6`): `0.678889347 -> 0.676473328` (`df = -2.416019e-03`, `rect_wall_14`)
  - `B10_20x12` (`20x12, +8`): `0.731743779 -> 0.731213864` (`df = -5.299152e-04`, `rect_wall_23`)
  - `B6_10x21` (`10x21, +6`): `0.657505411 -> 0.657124329` (`df = -3.810821e-04`, `rect_wall_20`)
  - `B10_22x11` (`22x11, +10`): `0.756006961 -> 0.755882519` (`df = -1.244420e-04`, `rect_direct_0`)
  - `B6_18x9` (`18x9, +6`): `0.705985624 -> 0.705908383` (`df = -7.724096e-05`, `rect_direct_0`)
- **8 ties (`df = 0.0`)**: `B8_18x8`, `B7_17x8`, `B5_22x7`, `B5_15x5`, `B9_19x11`, `B4_16x11`, `B4_16x10`, `B3_15x9`.

### 3. Phase A2: Multi-`n` Broadcast + Seam Melt (`docs/scale_ckpt.json`)
- **7 WINs**: `538, 740, 796, 910, 911, 914, 971` (detailed above).
- **19 non-winning broadcasts** (mostly because the other pinwheel arm had larger `f`, or `B6_24x8` / `B4_19x9` had an asymmetric partner):
  - Near-ties (`dL ~ 2e-9 .. 3e-9`): `n=744` (`+2.48e-09`), `n=854` (`+2.85e-09`), `n=918` (`+2.27e-09`), `n=972` (`+3.05e-09`).
  - Misses (`dL > 0`): `n=587` (`+1.56e-02`), `n=588` (`+4.07e-02`), `n=685` (`+1.46e-02`), `n=687` (`+3.96e-02`), `n=691` (`+6.72e-02`), `n=742` (`+2.72e-02`), `n=743` (`+2.22e-02`), `n=794` (`+2.18e-02`), `n=851` (`+2.18e-02`), `n=855` (`+4.21e-02`), `n=912` (`+2.64e-03`), `n=973` (`+9.30e-03`), `n=975` (`+9.67e-03`), `n=977` (`+1.03e-02`), `n=979` (`+9.57e-03`).

### 4. Phase B: Initial Pinwheel Seam Melt (`docs/scale_ckpt.json`)
- First 4 targets tested before restart tied at `~1e-15`: `n=691` (`-9.36e-16`), `n=745` (`-3.65e-15`), `n=858` (`+6.74e-16`), `n=918` (`+1.20e-15`).

---

## Reproducing & Testing

```bash
uv run --project poc/baseline/poc --with scipy --with numba --with pytest --with requests \
  python3 -m pytest -q poc/baseline/poc/tests/ poc/auto_union/poc/ .github/scripts/ poc/swarm_brickslp/tests/
```
