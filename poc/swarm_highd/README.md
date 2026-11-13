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
- **Status**: Fast-failed at `~5m40s` (`task-187` terminated per `#80` `< 5 min` rule).
- **Results**:
  - `n=698`: `[Batch78] N=41 feas=5 best_Lf=26.9206 min_pen=5.90e-4`; `[BatchLoc] N=27 feas=0 min_pen=3.29e-4`; `BEST VALID n=698: inc78_13 L_lit=26.8811088295203 (inc=26.881104490, dL=+4.340e-06, win=False)`.
  - `n=860`: `[Batch78] N=41 feas=2 best_Lf=29.8418 min_pen=6.92e-4`; `[BatchLoc] N=27 feas=0 min_pen=3.84e-4`; stalled in 24-round HiGHS SLP polish across 12 `n=860` candidates (`>2m40s` in SLP alone).
- **Diagnosis**: Calling `joint_xytheta_slp(rounds=10)` PLUS a second `rounds=14` upgrade whenever `L_slp < L_inc + 0.002` caused `~24` HiGHS LP solves on `n=860` across 12 candidates (`> 3 min` per `n`). Dropping the cap to `k_keep = 6`, `rounds = 8`, and only upgrading if `L_slp < L_inc + 0.0005` cuts SLP polish to `< 35s` per `n`. Also `cascades` (`ff463df`) just won `n=904` and `n=640` directly via `joint_xytheta_slp` on `merged.csv` (`Track A`), which we have not yet run on our other 11 targets!

---

## Idea 3 (Logged pre-test): Full 14-Target Direct Joint `(x, y, dtheta)` SLP Audit + Ultra-Fast Dual-Regime (`a=0.78` & `a=0.92`) Pocket Melts

- **Files**: `poc/idea3_direct_and_fastmelt.py`
- **Hypothesis**:
  1. **Track A — Direct Joint `(x, y, dtheta)` SLP on ALL 14 Owned Targets**: Run `joint_xytheta_slp(rounds=18, tr_th0=0.025, tr_xy0=0.10)` + multi-margin `fixed_angle_lp_tighten` directly on `merged.csv[n]` across all 14 targets (`~30-45s` total on 4 cores) to harvest any immediate local `(x, y, theta)` optimality slack (the exact mechanism behind `cascades`'s `640` and `904` wins).
  2. **Track B — Ultra-Fast Pocket Melts (`a=0.78` & gentler `a=0.92`, `k_keep=6`, `rounds=8`)**: Keeps SLP polish `< 35s` per `n` so 4 targets finish cleanly in `< 3-4 min`, testing `{860, 339, 921, 981}` and `{703, 805, 919, 639, 862}`.
- **Status (`Track A` Direct Multi-Jitter SLP)**: Completed (`poc/idea3_direct_and_fastmelt.py --mode direct --threads 4`, `582.2s`, `docs/idea3_direct_results.json`).
  - **CERTIFIED WIN `n=862`**: `jit0_m1e-07` (`142.65s`): `L_lit = L_pd = 29.82713653946183845476838727052095377039066254099632` vs `inc_L = 29.82713734035603629491603846202282427686716636335656` (`dL = -8.008941954074089e-07`), written to `poc/swarm_highd/final.csv` (`data/win_n862.csv`).
  - Remaining 13 targets: `237, 269, 339, 545, 703, 805, 919, 921, 981` (`dL=0.0`), `698, 860, 918` (`dL=-3.55e-15`, float tie), `639` (`infeasible: Squares overlap` at `m=1e-7`, needs `m=1e-6` or L-BFGS repair).
- **Status (`Track B` Fast Melt)**: Partial pilot (`339`: `inc92_2` `+1.871e-6` in `19.3s`; `860`: `inc92_13` `+3.424e-6` in `205.1s`, interrupted by server restart). Notice `inc92` (`a=0.92`) consistently beat `inc78` (`a=0.78`), motivating **Idea 4 (Frozen-Exterior / Masked-Pocket Melt)**.

---

## Idea 4 (Logged pre-test): Frozen-Exterior Masked-Pocket C++ `a`-Melt + Void/Weak-Wall Re-Packing + Fine-Margin SLP

- **Files**: `poc/highd_ils.cpp` (add `seam_mask` frozen-exterior support or masked L-BFGS), `poc/idea4_masked_pocket.py`
- **Hypothesis**:
  1. Why did `a=0.78` and even `a=0.92` global melts re-freeze `+2e-6..+3e-6` above `L_inc`? Because melting all `n` squares shrinks/perturbs the already-optimal `~20°` L-corridor everywhere instead of concentrating rearrangement where voids (`0.30..0.40` clearance) and weak walls live.
  2. Freezing all squares outside a local pocket (`R = 4.5..7.0` around max-clearance voids, weakest-wall defects, or tilted/upright seams — `~35-95` free squares) while running C++ L-BFGS `a`-melt (`a: 0.85 -> 1.00`) + cavity relocation ONLY on the active pocket lets 1-2 squares rotate/slide into adjacent voids without disturbing the rest of the crystal, followed by full-box fine `joint_xytheta_slp` + `[1e-7, 2e-8, 5e-9]` LP tighten.
  3. Also test fine-margin (`1e-6` repair check on `n=639` + `1e-4` micro-jitter on `n=862` iterated once more / other top `n`: `860, 698, 918, 921, 981, 339, 545`).
- **Status**: Completed (`poc/idea4_masked_pocket.py --targets 639 698 918 545 --threads 4`, `253.8s`, `docs/idea4_results.json`, 0 wins).
- **Results**:
  - `n=639` (`55.8s`): `mvoid_5` `25.7811432274468` (`+6.706e-06`).
  - `n=698` (`71.7s`): `mpin_4` `26.8811072313141` (`+2.742e-06`).
  - `n=918` (`101.0s`): `mseam_9` `30.7841523417815` (`+3.944e-06`).
  - `n=545` (`24.0s`): `mvoid_9` `23.8251747009582` (`+2.365e-06`).
- **Diagnosis & `win_n862.csv` Structural Diff**:
  1. Diffing `win_n862.csv` vs `merged[862]` revealed how `jit0` broke through when 0-jitter (`fine`/`med`) stalled at `dL=0`: `189` squares moved (`dp > 1e-3`, `47 > 1e-2`, `max dp = 0.249` on loose upright `sq 637` and `0.240` on `sq 837`), `~15` tilted `~27°/71°` squares rotated up to `1.45°`, and wall contacts rebalanced `[28, 25, 29, 27] -> [28, 27, 26, 28]`.
  2. By contrast, `Idea 4` (`a=0.80` melt + `±0.11 / ±14°` perturbation even inside a frozen exterior) is still too violent and re-locks `+2.3e-6..+6.7e-6` above `L_inc`.

---

## Idea 5 (Logged pre-test): Iterative `n=862` Continuation + Loose-Square / Tilted-Pocket Coherent Jitter Directly into Joint `(x, y, dtheta)` SLP

- **Files**: `poc/idea5_coherent_slp.py`
- **Hypothesis**:
  1. **Continue `n=862` from `win_n862.csv`**: Since `jit0` was still moving `sq 637` by `0.249` and tilted strips by `1.45°` after 12 rounds, running another fine + micro-jitter SLP pass on `win_n862.csv` (`rounds=18`) can push `n=862` even lower!
  2. **No-Melt Loose-Square & Tilted-Pocket Perturbations -> Direct `joint_xytheta_slp`**: Instead of shrinking `a` to `0.80` (which breaks contacts), apply small **non-destructive** nudges (`1e-4..2e-3` pos, `0.02°..0.5°` angle) specifically on: (a) loose/rattling squares near max-clearance voids, (b) tilted corridor strips & upright-tilted seams, (c) weak-wall drift + `8` multi-scale micro-jitter seeds (`tr_th0=0.012..0.020, tr_xy0=0.04..0.08`), feeding directly into `joint_xytheta_slp` + `[1e-7, 2e-8, 5e-9]` LP tighten.
- **Status**: Completed (`poc/idea5_coherent_slp.py --targets 862 339 545 639 698 703 805 --threads 4`, `202.3s`, `docs/idea5_results.json`, 0 new wins; dropped per `#106`).

---

## Idea 6 (Logged pre-test): Masked-Pocket Cavity-Relocation (`L0 = L_inc + 0.032`, `a=0.78`) + Frozen-Core Border-Seam Surplus Peels (`1e-5..1e-2` Structural Track)

- **Files**: `poc/highd_ils.cpp`, `poc/idea6_structural_masked.py`
- **Hypothesis**:
  1. **Why Idea 4 (`+2.3e-6..+6.7e-6`) didn't trigger topological pocket swaps like `n=694` (`-2.88e-4`)**: Idea 4 started too tight (`L0 = L_base + 0.004`) and had no `cavity_relocate_orbit` inside `run_masked_melt_trajectory`, so squares inside the mask could not rotate `45°`/`22°`/`0°` or swap positions with voids.
  2. **Mechanism A — Masked-Pocket `cavity_relocate_orbit` + `+0.032` Expansion (`a: 0.78 -> 1.00`)**: Freeze `~90%` of the crystal outside `R = 4.8..7.2` around max-clearance voids, weakest walls (with wall-slack drift), and corridor junctions, while expanding the box to `L0 = L_base + 0.032` (`L_sched = [L_base+0.032, L_base+0.009, L_base-2e-4, L_base-0.002, L_base-0.005]`) and running masked `cavity_relocate_orbit` + `a=0.78 -> 1.00` melt ONLY inside the pocket, then unfreezing gently.
  3. **Mechanism B — Frozen-Core Border-Seam Surplus Peels (`592->545`, `756->703`, `808->698`, `919->860`, `979->918`, `983->921`)**: Keep the entire intact interior (`dist_from_cut > 3.2` and `oh == 0`) frozen (`active_mask = 0`), while melting & cavity-relocating ONLY the `~60-120` squares in the peeled overhang border strip (`a=0.78 -> 1.00`) so the `0.05..0.11` overhang re-packs without shattering the donor's core!
- **Pilot (`poc/idea6_structural_masked.py --targets 698 860 545 703 --threads 4`, `258.0s`, `docs/idea6_results.json`)**:
  - **CERTIFIED REAL WIN `n=698`**: `mpk_void_11` (`50.2s`): `L_lit = L_pd = 26.8810999586947...` vs `inc_L = 26.8811044897729...` (**`dL = -4.531e-06`**, `38` squares shifted `> 1e-2`, `max dp = 0.3908` around `(9.5, 22.5)` + right/top wall balance!).
  - `n=860` (`112.5s`): `mpk_wall_1` `29.80243677` (`+8.053e-06`).
  - `n=545` (`30.3s`): `mpk_seam_0` `23.82517747` (`+5.136e-06`).
  - `n=703` (`63.8s`): `mpk_wall_9` `26.94701946` (`+3.608e-06`).
- **Status**: **PASSED `#108` 15-min gate (`n=698` REAL WIN `-4.531e-06`)!** Scaling `mpk_void`/`mpk_wall`/`mpk_seam` across remaining pool targets (`918, 919, 921, 981, 805, 639, 339, 237, 269`) and re-running `698` continuation.
