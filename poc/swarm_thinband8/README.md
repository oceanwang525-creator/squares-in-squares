# H112 (`thinband8`): Elongated `h in {7, 8, 9}` Sub-Brick Synthesis & Asymmetric Pinwheel Sweep

Hypothesis **H112** harvests existing `h in {7, 8, 9}` (and `h = 10`) tilted sub-bricks, synthesizes a full family of `w in {7, 8, 9}, h in {15..23}, g in {4..8}` bricks via outer unit-layer peeling, `vstrip` column addition/removal, `compress_brick_slp`, and `brick_mechwalk`, and sweeps `K = 2` pinwheels (`kx, ky in {-2..+1}`, `kx + ky <= 1`) across `m = 22..29, r = +9..+15` with C++ corner/seam melting + `joint_xytheta_slp` + `lpslack`.

## PR1 Certified Strict Wins (`literal == pandas`, vs `origin/main` `cc58c7e`)

| `n` | `(m, r)` | Incumbent (`merged.csv`) | `thinband8` `L` | `dL` | Construction |
|---:|---|---:|---:|---:|---|
| `634` | `(25, +9)` | `25.688369972665937362` | **`25.685975744175304842`** | **`-2.394228e-03`** | `MW_8x16g4 + MW_9x17g5_T` (`kx=0, ky=0`, `rot180_flipx`, melt + SLP + `lpslack`) |
| `585` | `(24, +9)` | `24.705882352946071035` | **`24.704470952419591608`** | **`-1.411401e-03`** | `MW_8x15g4 + MW_9x16g5_T` (`kx=0, ky=0`, `id_flipx`, melt + SLP + `lpslack`) |

## Layout
- `poc/thinband8_lib.py` — Brick harvest, peel/`vstrip` synthesis, `compress_brick_slp`, `brick_mechwalk`, `build_pinwheel_kxky`, `solve_pinwheel_pair`, and dual certification.
- `poc/run_synth_thin.py` — Synthesizes `w in {7, 8, 9}, h in {15..23}, g in {4..8}` brick catalog (102 canonical shapes filled; 164 harvested bricks).
- `poc/run_gate_thin.py` — Gate sweep runner.
- `poc/run_scale_thin.py` — Scale-up runner across `m = 22..29, r = +9..+15`.
- `tests/test_thinband8.py` — Unit tests + dual certification against live portfolio frontier.
- `final.csv`, `data/win_n*.json`, `docs/` — Certified winning packings and summaries.
