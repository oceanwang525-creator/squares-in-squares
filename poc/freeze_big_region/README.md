# freeze_big_region

Big free region (all squares within R of the tilted structure, plus right/top wall bands; or everything except three 2x2 corner blocks), jittered start (sigma 0.02-0.1 position, 1-3 deg angle, seed-derived), L-BFGS penalty relaxation with an annealed container (start at L0+0.3, step down adaptively, on failure back off with a kick, so L moves up and down), then exact rigid repair (centre dilation 1+1e-10..1e-6, official `squares.core.evaluate`).

**Estimated score gain: 0. No win.** No run reached a penalty-free state with L below the incumbent, so no repair, no CSV, no `squares score`. merged.csv untouched.

## Evidence (33 runs, `docs/run_all.log`; 3 n x [R=3,6,10 one seed, 8 s each + "all but corners" 8 seeds, 22 s each])
Best penalty-feasible L reached vs incumbent L0 (all were ABOVE L0; measured official L: none, nothing passed repair below L0):

| n | L0 (ours) | leader | R=3 / 6 / 10 | all-but-corners, best of 8 seeds |
|---|---|---|---|---|
| 634 | 25.707107 | 25.682309 | stuck at jitter, E~2e-8 at L0+0.3 | 25.707248 (+1.4e-4) |
| 911 | 30.695622 | 30.660916 | stuck | 30.697498 (+1.9e-3) |
| 738 | 27.679381 | n/a here | stuck | 27.681881 (+2.5e-3) |

## Where it stalls
- R=3/6/10: the jittered start cannot even be repaired to E=0 at L0+0.3 (E~2e-8, a strict local minimum). Frozen lattice outside R pins the jammed tilted/lattice interface, so the free ring has no slack. Larger frozen set = no motion at all.
- All-but-corners: jitter relaxes to E=0 at loose L, but the descent stalls 1e-4 to 3e-3 above the incumbent L0 for every seed; it does not even recover the incumbent. The incumbents are rigid contact networks; random jitter + penalty descent lands in worse jammed states, not better.
- L-BFGS convergence on this nonsmooth penalty is slow near E~1e-9, which also limits the finest step (2e-6 min).

## Not tried / caveats
- ~20 CPU min total, short per-run budgets (8-22 s); machine load was ~17 on 8 cores. Longer runs, momentum/GD instead of L-BFGS, and swap moves untested. 738 leader value not looked up.

Run: `poc/freeze_big_region/poc/run_all.sh` from repo root (uses `poc/k2p10_fill2/poc/a_overlap.py`).
