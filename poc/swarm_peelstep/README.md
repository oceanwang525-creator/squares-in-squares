# H109 (`peelstep`): Unified Cross-Donor Pinwheel & Masked C++ Melt

Pools sub-bricks from `cleanpeel2`, `stepdown`, `gstep`, `band3`, plus newly extracted and 1-row/1-column peeled+compressed (`compress_brick_slp` + `brick_mechwalk`) sub-bricks from `win_n974` (`31.700187`), `win_n912` (`30.692988` / `30.692586`), and `n=852` (`29.688527`), and sweeps 2-brick pinwheels across `kx, ky in {-1, 0, +1}` with masked C++ corner/seam melt, `joint_xytheta_slp_tight`, and `lpslack`.

## Scale-Up Result (PR2, vs `origin/main` `cc58c7e`)

| `n` | `m, r` | Incumbent (`origin/main`) | `peelstep` `L` | `dL` | Mechanism |
|---:|:---:|---:|---:|---:|---|
| **973** | `31, +12` | `31.679166610638385` (`merged.csv`) | **`31.673872795564327`** | **`-5.293815e-03`** | `B6_10x21_n972_BR` (`10x21+6`) + `B6_10x21_n972_BR_T` (`21x10+6`), `kx=0, ky=0`, `flipx_rot180`, masked C++ melt + SLP + `lpslack` |
| **795** | `28, +11` | `28.703403922913072` (`merged.csv`) | **`28.699779786461649`** | **`-3.624136e-03`** | `H60_B4_9x16_peel_vstrip_k3` (`9x16+5`) + `n852_BR_11x20_g6_peel_top_slp_walk_T` (`20x11+6`), `kx=0, ky=0`, `id_rot180`, masked C++ melt + SLP + `lpslack` |
| **854** | `29, +13` | `29.721540002259443` (`merged.csv`) | **`29.719475060115124`** | **`-2.064942e-03`** | `B6_9x19_n585_TL` (`9x19+6`) + `B7_10x22_vstrip_pt_T` (`22x10+7`), `kx=+1, ky=0`, `flipy_id`, masked C++ melt + SLP + `lpslack` |

(`n=913` from PR1 is preserved in `origin/main` `merged.csv`.)

## Gate Result (PR1, Merged)

| `n` | `m, r` | Incumbent (PR1 base) | `peelstep` `L` | `dL` | Mechanism |
|---:|:---:|---:|---:|---:|---|
| **913** | `30, +13` | `30.707106781206181` (`merged.csv`) | **`30.706062294652124`** | **`-1.044487e-03`** | `B6_9x20_n853_BRT` (`band3`, `9x20+6`, `f=0.705882`) + `B7_10x22_vstrip_T` (`gstep`, `22x10+7`, `f=0.688507`), `kx=+1, ky=0`, `flipy_rot180`, masked C++ corner melt + SLP + `lpslack` |

## Reproduction

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/highrpeel_venv/bin/python -m pytest poc/swarm_peelstep/tests/test_peelstep.py -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/highrpeel_venv/bin/python poc/swarm_peelstep/poc/run_gate.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/highrpeel_venv/bin/python poc/swarm_peelstep/poc/run_scale.py
```
