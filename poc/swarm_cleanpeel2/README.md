# H74 (`cleanpeel2`): Sub-Brick Row/Column Clean-Peel + Standalone `brick_mechwalk` + Zero/Corner-Melted Pinwheels

## Mechanism
1. **Outer Unit Row/Column Clean-Peel**: Extract rectangular tilted bricks from champion packings (`H60_B5_20x11_n970s6`, `B6_10x21_mw972`, `B5_20x10_mw972`, `B6_9x18_n796`, `B5_19x10_n850`, `B6_24x8`), peel 1+ outer unit row/column along the rectangle boundary (`w x h -> w x (h-1)` or `(w-1) x h` preserving `g`), re-compress via `compress_brick_slp`, and run standalone `brick_mechwalk` (`d(i+-j)` soft modes).
2. **Pinwheel Assembly + Seam Melt + `lpslack`**: Assemble 5-region pinwheels (`w1+w2 <= m, h1+h2 <= m`), optimize zero-overlap (`kx + ky < 0`) pairs via `joint_xytheta_slp_tight` + `polish_and_certify` and `kx=0, ky=0` corner-touching pairs via masked `cpp_batch_melt` + `joint_xytheta_slp_tight`, then polish with `poc/swarm_lpslack/poc/lpslack.py`.

## Certified Wins (`literal == pandas`)
- **PR #78 (merged in `origin/main` `793c1ec`, `n=972`)**: `B5_20x10_peel_top` (`f=0.642881`) + `B6_10x21_mw972_walk` (`f=0.656513`), `kx=-1, ky=0`, direct SLP + `lpslack`:
  `31.656512178590887` -> **`31.656497298735282`** (`dL = -1.487986e-05`, `ds = -4.772716e-07`, `ds_L2n = -9.692268e-07`). Preserved in `merged.csv`.
- **PR2 (`n=974`, current `final.csv`, vs `origin/main` `cf838aa`)**: `B6_9x21_peel_left` (`9x21+6, f=0.692574`) + `B7_10x22_k10_T` (`22x10+7, f=0.688791`), `kx=0, ky=0`, masked C++ melt + SLP + `lpslack`:
  `31.705886745787046` (`merged.csv`) -> **`31.700186957068923`** (**`dL = -5.699789e-03`**, `ds = -1.826330e-04`, `ds_L2n = -3.710485e-04`).

## Gate & Scale-Up Results (`docs/gate_direct_summary.json`, `docs/scale_summary.json`)
| `n` | `m, r` | `L_inc` (`origin/main`) | Best H74 `L` | `dL` | Outcome | Best Pair / Note |
| ---: | :---: | :--- | :--- | :---: | :---: | :--- |
| **974** | `31, +13` | `31.705886745787046` | **`31.700186957068923`** | **`-5.699789e-03`** | **WIN (PR2)** | `B6_9x21_peel_left + B7_10x22_k10_T` (`kx=0, ky=0`) + melt + `lpslack` |
| **972** | `31, +11` | `31.656512178590887` | **`31.656497298735282`** | **`-1.487986e-05`** | **WIN (PR #78)** | `B5_20x10_peel_top + B6_10x21_mw972_walk` (`kx=-1, ky=0`) + `lpslack` |
| 850 | `29, +9` | `29.642569706691871` | `29.642569708615873` | `+1.924002e-09` | Tie | `H60_B5_20x11_n970s6 + B4_9x16_n684` |
| 909 | `30, +9` | `30.638828599737297` | `30.638891818306161` | `+6.321857e-05` | No win | `H60_B5_20x11_n970s6 + B4_10x16_n737` |
| 910 | `30, +10` | `30.656420974091060` | `30.656510804545165` | `+8.983045e-05` | No win | `B6_10x21_mw972_walk + B4_9x16_n684_T` |
| 971 | `31, +10` | `31.641871370426643` | `31.642876347116610` | `+1.004977e-03` | No win | `B5_20x10_peel_top + B5_20x10_peel_top_T` |
| 973 | `31, +12` | `31.679166610638383` | `31.682453441261900` | `+3.286831e-03` | No win | `B6_10x20_mw972_top + B6_10x21_b4_T` |
| 852 | `29, +11` | `29.699128044056555` | `29.705885731756364` | `+6.757688e-03` | No win | `B5_20x10_peel_top + B6_9x18_n796_TL` |
| 911 | `30, +11` | `30.676983696661890` | `30.684060764060315` | `+7.077067e-03` | No win | `B6_10x20_mw972_bottom + B5_10x19_n793_T` |
| 851 | `29, +10` | `29.666031468227378` | `29.673983342246674` | `+7.951874e-03` | No win | `B6_11x20_from11x21_top + B4_17x9_n684` |
| 795 | `28, +11` | `28.703403922913072` | `28.732459882292487` | `+2.905596e-02` | No win | `B6_9x17_n796_bottom + B5_10x19_n793_T` |
| 687 | `26, +11` | `26.717912747967934` | `26.761309041453945` | `+4.339629e-02` | No win | `B7_17x8 + B4_17x9_n684_T` |
| 740 | `27, +11` | `27.705882358799239` | `27.761536404376820` | `+5.565405e-02` | No win | `B7_17x8 + B4_10x16_n737` |

## Verification
```bash
/tmp/highrpeel_venv/bin/python -m pytest poc/swarm_cleanpeel2/tests/test_cleanpeel2.py -v
```
