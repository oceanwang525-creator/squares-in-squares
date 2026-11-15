# H112 (`thinband8`): Elongated `h in {7, 8, 9}` Sub-Brick Synthesis & Asymmetric Pinwheel Sweep

Hypothesis **H112** harvests existing `h in {7, 8, 9}` (and `h = 10`) tilted sub-bricks, synthesizes a full family of `w in {7, 8, 9}, h in {15..23}, g in {4..8}` bricks via outer unit-layer peeling, `vstrip` column addition/removal, `compress_brick_slp`, and `brick_mechwalk`, and sweeps `K = 2` pinwheels (`kx, ky in {-2..+1}`, `kx + ky <= 1`) across `m = 22..29, r = +9..+15` with C++ corner/seam melting + `joint_xytheta_slp` + `pslp` + `lpslack`.

## PR3 Certified Strict Win (`literal == pandas`, vs `origin/main` `221bc28`)

*(PR1 `{585, 634}` and PR2 `{539, 635}` are preserved in `origin/main` `merged.csv`; PR3's `final.csv` carries `{637}` cleanly.)*

| `n` | `(m, r)` | Incumbent (`merged.csv`) | `thinband8` `L` | `dL` | Construction |
|---:|---|---:|---:|---:|---|
| `637` | `(25, +12)` | `25.744394485251974801` | **`25.743493518942190900`** | **`-9.009663e-04`** | `p_n915_7x18_g6 + p_n915_7x18_g6_T` (`kx=0, ky=0`, `id_rot180`, melt + SLP + `lpslack`) |

## Prior Merged PR Wins (Preserved in `origin/main` `merged.csv`)

| PR | `n` | `(m, r)` | Prior Incumbent | `thinband8` `L` | `dL` | Construction |
|---|---:|---|---:|---:|---:|---|
| PR2 | `635` | `(25, +10)` | `25.707106781186697000` | `25.705882364271365669` | `-1.224417e-03` | `MW_7x16g4 + 18x9g6` (`kx=0, ky=0`) |
| PR2 | `539` | `(23, +10)` | `23.732037287059769743` | `23.731097834588543823` | `-9.394525e-04` | `5x14g4 + 18x9g6` (`kx=0, ky=0`) |
| PR1 | `634` | `(25, +9)` | `25.688369972665937362` | `25.685975744175304842` | `-2.394228e-03` | `MW_8x16g4 + MW_9x17g5_T` (`kx=0, ky=0`) |
| PR1 | `585` | `(24, +9)` | `24.705882352946071035` | `24.704470952419591608` | `-1.411401e-03` | `MW_8x15g4 + MW_9x16g5_T` (`kx=0, ky=0`) |

## Layout
- `poc/thinband8_lib.py` — Brick harvest, peel/`vstrip` synthesis, `compress_brick_slp`, `brick_mechwalk`, `build_pinwheel_kxky`, `solve_pinwheel_pair` (`pslp` + `lpslack`), and dual certification.
- `poc/run_synth_thin.py` — Synthesizes `w in {7, 8, 9}, h in {15..23}, g in {4..8}` brick catalog.
- `poc/run_gate_thin.py` — Gate sweep & brick loader.
- `poc/run_scale_thin.py` — Scale-up runner across `m = 22..29, r = +9..+15`.
- `tests/test_thinband8.py` — Unit tests + dual certification against live portfolio frontier.
- `final.csv`, `data/win_n*.json`, `docs/` — Certified winning packings and summaries.
