# swarm_ratchetall — H119: auto 4-corner reduced arm-row `pslp` + `mechwalk` ratchet

## Idea

`poc/row_ratchet_980` showed that, on n = 980, a free-angle kink-safe sequential-LP squeeze (`pslp`) on a
*reduced* model (bottom-left 0° grid block frozen, top-right block translating rigidly with the wall,
only the tilted arms + one seam layer free) is ~11x faster than a full-row squeeze, and that alternating
it with the team mechanism walk (`poc/swarm_mechwalk`) ratchets L down.

This POC generalizes that model to any two-arm / pinwheel packing:

- `poc/arm_model.py` — auto-partitions the packing. Squares within 2.6 of a strongly tilted (> 2°) square,
  or any square tilted > 0.2°, are active. Remaining 0° squares are split into connected components
  (centre distance < 1.15); each component of ≥ 4 squares touching exactly one corner wall pair becomes
  - `BL_static` (walls x=0, y=0): dx = dy = dθ = 0,
  - `TR_rigid` (x=L, y=L): translates by (dL, dL),
  - `TL_rigid` (x=0, y=L): translates by (0, dL),
  - `BR_rigid` (x=L, y=0): translates by (dL, 0).
  Components spanning opposite walls or no corner become active. Only rigid squares within 1.6 of an active
  square are kept in the LP (plus any rigid square within 1.8 of a different rigid block); the rest are
  shifted analytically. Falls back to the full row when the reduction would keep > 85 % of squares.
- `poc/pslp.py` — `row_ratchet_980/pslp2.py` with per-axis `move_x / move_y / frozen_x / frozen_y / frozen_th`
  masks so all four corner types are exact in the LP (moving columns are substituted by the `df` column).
- `poc/ratchet.py` — step 0: reduced `pslp` on the live incumbent, then a full-row `pslp` pass on the
  result; then rounds of `run_gate._run_round` (mechwalk mode-walk + C++ masked melt) → reduced `pslp`
  → full `pslp` on the best distinct candidates, beam 2, stopping after two no-gain rounds.
- `poc/portfolio.py` — live portfolio (exact `merged.csv` rows + every other `poc/*/final.csv`, both
  transport modes) and dual certification with tiny LP/expansion margins only if needed.
- `poc/verify_final.py` — certifies `final.csv` via `poc/auto_union/poc/union.py` (`load` → literal +
  pandas) against the live portfolio.

## Gate result (30 min, vs `origin/main` `f4fbb17`)

Step 0 alone (reduced + full `pslp` on the incumbent, no walk) already gives 8 strict wins out of the 12
reduced-model gate targets; `literal == pandas` for all.

| n | K | r | corner blocks | incumbent (`merged.csv`) | new L | dL |
|---|---|---|---|---|---|---:|
| 924 | 30 | +24 | TL 423 / BR 50 | 30.855059003278 | **30.854457494176** | **−6.015e-4** |
| 975 | 31 | +14 | BL 79 / TR 430 | 31.716566932562 | **31.716071475338** | **−4.955e-4** |
| 863 | 29 | +22 | TL 388 / BR 50 | 29.842805713654 | **29.842685560243** | **−1.202e-4** |
| 748 | 27 | +19 | TL 51 / BR 330 | 27.826809269178 | 27.826786853461 | −2.242e-5 |
| 912 | 30 | +12 | BL 78 / TR 391 | 30.692586319123 | 30.692576840848 | −9.478e-6 |
| 916 | 30 | +16 | BL 95 / TR 357 | 30.750239508235 | 30.750239441758 | −6.648e-8 |
| 913 | 30 | +13 | BL 86 / TR 385 | 30.706062294652 | 30.706062291590 | −3.062e-9 |
| 974 | 31 | +13 | BL 90 / TR 411 | 31.700186957069 | 31.700186954542 | −2.527e-9 |

Total `Σ (L_new² − L_old²)/n = −8.303e-5`. Misses (already LP-stationary): 919, 979, 981, 976.

Geometric reason it works: the TL/BR pinwheels (863, 924, 748) were never covered by the BL/TR-only
`row980` model, and the full-row SLP stalls on them at its trust radius; with the frozen/rigid corner
blocks removed from the LP, the arms get a feasible collective rotation+shear that the full LP did not
find within its iteration budget.

## Reproduce

```
OMP_NUM_THREADS=1 python poc/swarm_ratchetall/poc/ratchet.py --pslp-only --targets 975 924 863 748 912 919 974 913 979 981 976 916 --tag gate_step0
python poc/swarm_ratchetall/poc/verify_final.py
pytest -v poc/swarm_ratchetall/tests/test_ratchetall.py
```
