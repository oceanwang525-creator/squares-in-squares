# H16 — Clean-Peel (`prot=0..2`, `tilt=0` 1-Layer L-Border Peels + Cavity Relocation) (`poc/swarm_cleanpeel`)

Tested Hypothesis **H16**: `dm = 1` homology projections `(n_donor = m_d^2 + r -> n_target = (m_d - 1)^2 + r)` where the peeled `2*m_d - 1` outer L-border consists of 100% axis-aligned unit squares (`tilt = 0`) and `0..2` remaining squares protrude past the `L_d - 1` box boundary.

## Summary of Results vs. `origin/main` (`4478e04d`)

- **3 Certified REAL WINs** (`literal == pandas` via `poc/auto_union/poc/union.py`, **total `ds = -2.869930e-05`** vs. `origin/main` `4478e04d`):
  - **`n=485`** (`peel_530_BL_clamp_rot45_direct`, `m=22, r=+1`, `prot=1` with `prot_amt=0.00766`):
    - `L = 22.5330402775912109878636418370519266726786852883551`
    - `dL = -4.329429e-05`, `ds = -1.965895e-06` vs `4478e04d` (`22.533083571885758`)
    - Also beats `origin/swarm-sametilt` PR #29 (`22.533079978339103103705`) by `dL = -3.970075e-05` (`ds = -1.802718e-06`)!
  - **`n=442`** (`peel_485_BL_clean0_raw`, `m=21, r=+1`, `prot=0, tilt=0` cascade from `n=485` win):
    - `L = 21.5330402831450951026491945625973010750792970855948`
    - `dL = -3.968519e-05`, `ds = -1.887632e-06` vs `4478e04d` (`21.533079968339098062363`)
  - **`n=401`** (`peel_442_BL_clamp_keepAng_from442raw`, `m=20, r=+1`, cascade from `n=442` win; initial win `20.5330798775` also came directly from `peel1_442_BL` `prot=0, tilt=0`):
    - `L = 20.5330293030942054381522428790826518657223353010434`
    - `dL = -5.008038e-04`, `ds = -2.500895e-05` vs `4478e04d` (`20.533530106866359326964`)

## 15-Minute Falsification Gate (3 `n`s)

Tested all 4 corners (`BL, TL, BR, TR`) with BOTH (a) direct boundary clamp (`keepAng, rot0, rot45`) + `repair_overlaps` + C++ masked melt (`run_cpp_masked_melt`) + `joint_xytheta_slp`, and (b) interior void relocation (`reloc` / `dropMinProt_reloc`) across both `Lt`-targeting (`L_goal = L_d - 1`) and soft `L_inc - 2e-4` targeting (`a_start = 0.72`, `R = 6.5..12.5` + full-box unfreeze):

| Target <- Donor | Corner (`tilt=0, prot=1`) | `L_inc` (`4478e04d`) | `L_d - 1` | Gap | Best Seen `L` | Best `dL` | Outcome |
|---|---|---:|---:|---:|---:|---:|---|
| `582 <- 631` | `TL` (`max_prot=0.2472`) | `24.633982502` | `24.608712578` | `+0.025270` | `24.666684470` | `+3.270e-02` | Falsified (`0/120`, `0/80` feas) |
| `692 <- 745` | `TL` (`max_prot=0.3232`) | `26.791643395` | `26.776110986` | `+0.015532` | `26.826898394` | `+3.525e-02` | Falsified (`0/120`, `0/80` feas) |
| `577 <- 626` | `BL/TL` (`max_prot=0.2604`) | `24.533050506` | `24.520815280` | `+0.012235` | `24.560331693` | `+2.728e-02` | Falsified (`0/120`, `0/80` feas) |

**Mechanism Diagnosis:** Large single-square protrusions (`max_prot = 0.25..0.32`) come from tilted crystal blocks/diamonds (`17x17` 45° core in `626`, `~27°` block in `631`/`745`) that penetrate into the peeled `1x1` border row. Interior voids have `clr <= 0.30` (requiring `>= 0.20` overlap relaxation), so neither clamping nor void relocation can pack inside `L_inc` without displacing the crystal by `+0.027..+0.035`. By contrast, **`prot = 0`** (`401<-442`, `442<-485_new`) and **micro-protrusions** (`prot = 0.00766` in `530->485`) compress cleanly under `repair_overlaps` + `joint_xytheta_slp`.

## Full Clean-Peel Sweep Log (`prot=1..2, tilt=0`)

| Target <- Donor | `L_inc` (`4478e04d`) | Best Seen `L` | Best `dL` | Outcome |
|---|---:|---:|---:|---|
| `485 <- 530` | `22.533083572` | `22.533040278` | `-4.329e-05` | **REAL WIN** (`peel_530_BL_clamp_rot45_direct`) |
| `442 <- 485_new` | `21.533079968` | `21.533040283` | `-3.969e-05` | **REAL WIN** (`peel_485_BL_clean0_raw`) |
| `401 <- 442_new` | `20.533530107` | `20.533029303` | `-5.008e-04` | **REAL WIN** (`peel_442_BL_clamp_keepAng_from442raw`) |
| `362 <- 401_new` | `19.535533906` | `19.542370018` | `+6.836e-03` | No win (`feas=0/40`) |
| `83 <- 102` | `9.634757649` | `9.654182815` | `+1.943e-02` | No win (`feas=0/40`) |
| `85 <- 104` | `9.742640687` | `9.778175380` | `+3.553e-02` | No win (`feas=0/48`) |
| `105 <- 126` | `10.789303784` | `10.917872902` | `+1.286e-01` | No win (`feas=0/48`) |
| `109 <- 130` | `10.949289581` | `11.000064026` | `+5.077e-02` | No win (`feas=0/30`) |
| `110 <- 131` | `10.996783397` | `11.000080356` | `+3.297e-03` | No win (`feas=0/30`) |
| `123 <- 146` | `11.591378145` | `11.626257353` | `+3.488e-02` | No win (`feas=0/48`) |
| `125 <- 148` | `11.707106781` | — | — | No win (`feas=0/36`) |
| `126 <- 149` | `11.742640687` | `11.899495053` | `+1.569e-01` | No win (`feas=0/48`) |
| `175 <- 202` | `13.764050052` | `13.843090379` | `+7.904e-02` | No win (`feas=0/48`) |
| `199 <- 228` | `14.614689853` | `14.622987012` | `+8.297e-03` | No win (`feas=0/48`) |
| `200 <- 229` | `14.656854249` | `14.661210705` | `+4.356e-03` | No win (`feas=0/48`) |
| `234 <- 265` | `15.822875656` | `16.000014487` | `+1.771e-01` | No win (`feas=0/36`) |
| `260 <- 293` | `16.633520950` | `16.674009892` | `+4.049e-02` | No win (`feas=0/40`) |
| `292 <- 327` | `17.591378145` | `17.618366891` | `+2.699e-02` | No win (`feas=0/48`) |
| `450 <- 493` | `21.733166008` | `22.000151996` | `+2.670e-01` | No win (`feas=0/48`) |
| `488 <- 533` | `22.591378145` | `22.638391350` | `+4.701e-02` | No win (`feas=0/48`) |
| `539 <- 586` | `23.732037287` | — | — | No win (`feas=0/36`) |
| `550 <- 597` | `23.911174766` | `23.938705434` | `+2.753e-02` | No win (`feas=0/30`) |
| `583 <- 632` | `24.656854249` | `24.698455230` | `+4.160e-02` | No win (`feas=0/30`) |
| `789 <- 846` | `28.591332243` | `28.638411156` | `+4.708e-02` | No win (`feas=0/36`) |
| `801 <- 858` | `28.781136521` | `28.855966798` | `+7.483e-02` | No win (`feas=0/36`) |
| `854 <- 913` | `29.721540003` | `30.000086886` | `+2.785e-01` | No win (`feas=0/30`) |

## Verification

- `poc/swarm_cleanpeel/final.csv`: 3 complete groups (`401, 442, 485`), `literal == pandas` via `poc/auto_union/poc/union.py`.
- `.github/scripts/check_final_csv.py` + `pytest` on `poc/baseline/poc/tests/`, `poc/auto_union/poc/`, `poc/swarm_cleanpeel/tests/`.
