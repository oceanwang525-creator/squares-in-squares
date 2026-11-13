# swarm_highd — High-Excess (`D = 12..27`, `f = 0.768..0.947`) Packing Search

Disjoint target pool (`14` targets; never touch `861` or any non-owned `n`):
- Starter pilot (`< 10 min`): `269, 918, 545`
- Extended pool: `237, 919, 862, 805, 921, 698, 703, 639, 981, 860, 339`

## Pre-Test Audit (`origin/main` `a6fbef7` `merged.csv` vs `per_n_gap.csv`)

1. **`237 (15,12)` & `269 (16,13)`**: Already match `L_frontier` (`15.893805581` vs `15.893805585`; `16.889730074` vs `16.889730079`), using a `~42°` diagonal band (`non45 = 43, 45`). However, on the constant `k - D = 3` line, `n=867 (29,26)` achieves `f = 0.883544` (`-0.006186` lower than `269`'s `0.889730`!) via `~19°-27°` 3-quadrant L-corridors (`non45 = 313`), showing the `42°` band on `269` can still be beaten by wide non-45° L-corridors or multi-step surplus peels (`502/547 -> 269`).
2. **Remaining 12 targets (`339, 545, 639, 698, 703, 805, 860, 862, 918, 919, 921, 981`)**: Sit `dL = 0.0088..0.0161` above `L_frontier` (`f = 0.768..0.947`) with non-45° L-corridors (`18°..27°`) wrapping an upright block (`222..681` upright squares), strong wall-contact asymmetry (`918: [26,19,27,29]`, `698: [25,20,24,14]`, `860: [26,17,19,16]`, `981: [29,26,31,22]`, `339: [17,14,17,10]`), and `max_clr = 0.306..0.396` voids.
3. **Direct Multi-Step Surplus Peels Already Identified in `merged.csv`**:
   - `979 (31,18) wall(1,0,1,0) -> 918` (`surplus = 0`, `oh_rem = 0.0662`, `L_box = 30.766537`, `dL = -0.017612` below `L_inc = 30.784148` and below `L_frontier = 30.768043`!)
   - `592 (24,16) wall(0,1,1,0) -> 545` (`surplus = 0`, `oh_rem = 0.1125`, `L_box = 23.799034`, `dL = -0.026139` below `L_inc = 23.825172` and below `L_frontier = 23.812764`!)
   - `641 (25,16) wall(0,2,1,1) -> 545` (`surplus = 0`, `oh_rem = 0.1125`, `L_box = 23.799034`)
   - `756 (27,27) wall(0,1,0,1) -> 703` (`surplus = 0`, `oh_rem = 0.0530`, `L_box = 26.921416`, `dL = -0.025600`)
   - `983 (31,22) wall(0,1,1,0) -> 921` (`surplus = 3`, `oh_rem = 0.0616`, `L_box = 30.816790`)
   - `808 (28,24) wall(1,1,1,1) -> 698` (`surplus = 2`, `oh_rem = 0.0800`, `L_box = 26.873578`)
   - `919 (30,19) wall(0,1,1,0) -> 860` (`surplus = 2`, `oh_rem = 0.0762`, `L_box = 29.796506`)

---

## Idea 1 (Logged pre-test): Multi-Track High-`D` Basin Transfer + Pocket Melt + Joint `(x, y, dtheta)` SLP

- **Files**: `poc/highd_ils.cpp`, `poc/utils.py`, `poc/joint_slp.py`, `poc/idea1_highd_pilot.py`
- **Hypothesis**:
  1. **Track A — Multi-Step Surplus Center-Peels (`c = 1..8` wall splits & interior upright cross-cuts)**: Peeling `c` rows/cols from lower-`f` donors (`979 -> 918`, `592/641/639 -> 545`, `502/547 -> 269`) gives exact (`surplus=0`) or near-exact (`surplus=1..4`) sub-packings whose `L_box` sits `-0.008..-0.044` below `L_inc` with `oh_rem = 0.066..0.14`. Soft-core `a`-melt (`0.84 -> 1.00`) + C++ cavity-relocate ILS + HiGHS `joint_xytheta_slp` + multi-margin `fixed_angle_lp_tighten` will absorb the `0.066..0.14` overhang below `L_inc`.
  2. **Track B — High-Dual `(n+1)->n` & `(n+2)->n` Pocket-Square Deletion**: Deleting 1–2 highest-contact / boundary-pinning tilted or seam squares from `merged[n+1]` and `merged[n+2]` (`270/271 -> 269`, `919/920 -> 918`, `546/547 -> 545`) and melting an `R=4.8` pocket around the deleted sites + `a`-melt + `joint_xytheta_slp`.
  3. **Track C — Incumbent 6-Mode Wall-Unbalance, Void-Collapse, Seam-Unlock & Upright-Block Shift Melts**: Exploits wall-contact asymmetry (`cL, cR, cB, cT`) and `0.25..0.36` voids via pin-teleport, wall-pin pocket melts, upright/tilted seam melts, void-collapse pulls, wall-slack drift, and rigid upright-block shifts toward weak walls.
  4. **Track D — Multi-Angle Sheared L-Corridor Synthesis (for `269` & general high-`D`)**: Tests `(k-P) x (k-P)` upright core + `W = P + f` L-corridor at tilted angles `18°..27°` (`P = 5..7`) + `(n-1)+1` void insertions melted from `a_start = 0.78 -> 1.00`.
- **Pilot Targets (`< 10 min`, max 4 CPU threads)**: `269, 918, 545`.
- **Status**: Completed (`poc/idea1_highd_pilot.py --quick --targets 269 918 545 --threads 4`, results in `docs/idea1_results.json`).
- **Results**:
  - `n=269` (`22.9s`): `raw_inc_slp` = `16.889730074` (`dL=0`), 0 wins (`C++ ILS` 70 starts: `feas=0, min_pen=0.00074`; `inc_melt` re-converged at `+0.00001`).
  - `n=918` (`453.2s`): `raw_inc_slp` = `30.784148398` (`dL=0`), 0 wins (`C++ ILS` 66 starts: `feas=2, best_Lf=30.8158, min_pen=0.00036`; best melt `inc_melt_13` = `30.784490`, `+0.00034`).
  - `n=545` (`86.97s`): `raw_inc_slp` = `23.825172336` (`dL=0`), 0 wins (`C++ ILS` 70 starts: `feas=4, best_Lf=23.8502, min_pen=0.00061`; best melt `inc_melt_3` = `23.825186`, `+0.000014`).
- **Diagnosis**:
  1. **Wall-split peels (`979->918`, `592->545`, `756->703`)**: Global `a_start=0.82` shatters the crystal (`30.7957`), while gentle `a_start=0.95` overhang melts leave `min_pen >= 0.00138` because the wall cut slices through tilted border strips while the interior upright crystal stays rigid.
  2. **Global `a_start=0.82` `inc_melt` vs Local Melt**: Shrinking ALL `n` squares at `L0 = L_inc + 0.025` disturbs the already-tight `~20 deg` L-corridor, so it re-freezes slightly worse (`+0.00034` on `918`) instead of collapsing local voids/wall imbalances.
  3. **SLP runtime bottleneck**: `joint_xytheta_slp` ran 18 rounds without early-stopping on 25 candidates on `n=918` (`~414s` of the `453s`). Adding early stopping when `bbox_c` stalls `< 1e-8` for 2 rounds and pruning candidates cuts per-`n` time ~3-4x.

---

## Idea 2 (Logged pre-test): Fast Early-Stop `joint_xytheta_slp` + Dual-Regime (`a=0.78` Global & `a=1.00` Local-Only) Pocket Melts Across Highest-Asymmetry Targets

- **Files**: `poc/joint_slp.py`, `poc/idea2_highd_sweep.py`
- **Hypothesis**:
  1. **Early-stopping `joint_xytheta_slp` (`rounds=10` default, upgrade to `16` when promising)**: Stops after 2 consecutive stagnant rounds (`< 1e-8`) without losing LP polish precision, dropping per-`n` runtime by ~3-4x.
  2. **Track A — Deep Wall-Unbalance / Void-Collapse Melts (`a_start=0.78`, `L0 = L_inc + 0.04`, 32 seeds)**: Matches the exact `a_start=0.78` melt schedule that won `694` (`dL=-2.54e-4`), targeting our un-tested highest-asymmetry / highest-void `n`: `698` (`walls=[25,20,24,14], clr=0.381`), `860` (`[26,17,19,16], clr=0.395`), `339` (`[17,14,17,10], clr=0.306`), `921` (`[19,28,25,22]`), and subsequently `981, 703, 805, 919, 639, 862`.
  3. **Track B — Pure Local-Only Pocket Melts (`a_start=1.00`, `L_sched=[L_inc+0.003, L_inc-1.5e-4, L_inc-0.001, L_inc-0.0035]`, 28 seeds)**: Keeps the global lattice at `a=1.0` while perturbing only a local pocket around weak walls, voids, or bottleneck overlaps (`R=4..7`), allowing local rearrangements without shattering the crystal.
  4. **Track C — `(n+1)->n` & Top-3 `surplus_peel` Donors**: Same early-stop SLP polish on top `k_keep=12` diverse low-penalty candidates.
- **Pilot Targets (`< 6 min`, max 4 CPU threads)**: `698, 860, 339, 921`.
- **Status**: Planned (`poc/idea2_highd_sweep.py`).
