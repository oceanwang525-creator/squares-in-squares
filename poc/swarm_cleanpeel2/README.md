# H74 (`cleanpeel2`): Sub-Brick Row/Column Clean-Peel + Standalone `brick_mechwalk` + Zero-Overlap Pinwheels

## Mechanism
1. **Outer Unit Row/Column Clean-Peel**: Extract rectangular tilted bricks from champion packings (`H60_B5_20x11_n970s6`, `B6_10x21_mw972`, `B5_20x10_mw972`, `B6_9x18_n796`, `B5_19x10_n850`), peel 1 outer unit row/column along the rectangle boundary (`w x h -> w x (h-1)` or `(w-1) x h` preserving `g`), re-compress via `compress_brick_slp`, and run standalone `brick_mechwalk` (`d(i+-j)` soft modes).
2. **Zero-Overlap Pinwheel Assembly + `lpslack`**: Assemble non-overlapping 5-region pinwheels (`w1+w2 <= m, h1+h2 <= m`, `kx + ky < 0`), optimize via `joint_xytheta_slp_tight` + `polish_and_certify`, and polish with `poc/swarm_lpslack/poc/lpslack.py`.

## Gate Results (vs `origin/main` `9abec6b`, `literal == pandas` certified)
| `n` | `m, r` | `L_inc` (`origin/main`) | Best H74 `L` | `dL` | Outcome | Best Pair / Note |
| ---: | :---: | :--- | :--- | :---: | :---: | :--- |
| **972** | `31, +11` | `31.656512178590887` (`poc/carried_lpslack_wins/final.csv`) | **`31.656497298735282`** | **`-1.487986e-05`** | **WIN** | `B5_20x10_peel_top` (`f=0.642881`) + `B6_10x21_mw972_walk` (`f=0.656513`), `kx=-1, ky=0`, + `lpslack` |
| 971 | `31, +10` | `31.641871370426643` (`merged.csv`) | `31.642876347116610` | `+1.004977e-03` | No win | `B5_20x10_peel_top` (`f=0.642881`) + `B5_20x10_peel_top_T` |
| 910 | `30, +10` | `30.656420974091060` (`poc/carried_lpslack_wins/final.csv`) | `30.659396534586932` | `+2.975560e-03` | No win | `B5_20x10_peel_top` + `B5_10x19_n793` (`kx=0, ky=-1`); `kx=0, ky=0` melts expanded to `30.672376` |
| 911 | `30, +11` | `30.676983696661890` (`merged.csv`) | `30.684060764060315` | `+7.077067e-03` | No win | `B6_10x20_mw972_bottom` (`f=0.687458`) + `B5_10x19_n793_T` |
| 852 | `29, +11` | `29.699128044056555` (`poc/carried_lpslack_wins/final.csv`) | `29.705885731756364` | `+6.757688e-03` | No win | `B5_20x10_peel_top` + `B6_9x18_n796_TL` (`f=0.705887`) |
| 795 | `28, +11` | `28.703403922913072` (`merged.csv`) | `28.732459882292487` | `+2.905596e-02` | No win | `B6_9x17_n796_bottom` (`f=0.732460`) + `B5_10x19_n793_T` |

### Key Observation
Peeling 1 outer unit row/column from a corrugated brick breaks boundary interlock (`raw_f` jumps to `0.696..1.232`), and standalone SLP + `brick_mechwalk` recovers most of it (`B5_20x10_peel_top` `0.695982 -> 0.642881`, `B6_10x20_mw972_top` `0.943785 -> 0.686486`) while still leaving a net fractional thickness penalty on symmetric targets (`971, 911, 852, 795`). On **`n=972`**, pairing the walked `B6_10x21_mw972_walk` (`0.656516 -> 0.656513`) with the peeled `B5_20x10_peel_top` (`0.642881`) eliminates the `x`-seam collision (`kx=-1, ky=0`) and allows `joint_xytheta_slp_tight` + `lpslack` to break `n=972` down to `31.656497298735282` (`dL = -1.488e-5`).

## Verification
```bash
/tmp/highrpeel_venv/bin/python -m pytest poc/swarm_cleanpeel2/tests/test_cleanpeel2.py -v
```
