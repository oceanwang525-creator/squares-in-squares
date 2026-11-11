# global_search_midn: global stochastic search at mid n (negative)

Targets (gap to leader in L, from perpuzzle_best.csv): n=328 (ours 18.61437, leader 18.59954), n=497 (22.79709 / 22.78029), n=492 (22.70711 / 22.69054). Target L = ours - 0.001, so any overlap-free hit would be a strict win.
Code: poc/gsearch.py (one process per n, 330 s each, single-threaded, machine load ~60, so ~16 CPU min total). Logs/results: data/.

Starts, cycled: (a) incumbent dilated 2%, half the angles re-randomised; (b) square lattice + random tilted band + jitter; (c) random uniform positions/angles, Metropolis (cell-list local energy; translate/rotate moves, container 3% oversize, T 0.05 -> 5e-4) then L-BFGS. Then optional Metropolis polish, L-BFGS on the soft penalty at the target L, and hardening (scale positions until zero overlap, measured with numpy SAT). 6-7 starts per mode per n.
The "swap into largest hole" move was NOT implemented.

## Measured (hardened, overlap-free, all 1000 squares counts n)
| n | ours | leader | best (a) | best (b) | (c) |
|---|---|---|---|---|---|
| 328 | 18.6144 | 18.5995 | 18.726 | 19.083 | no zero-overlap state (min E 1.75) |
| 497 | 22.7971 | 22.7803 | 23.281 | none (min E 0.11) | none (min E 3.2) |
| 492 | 22.7071 | 22.6905 | 22.890 | 23.270 | none (min E 2.8) |
Hardened L are all worse than ours, and mostly above ceil(sqrt n) (23 for 497/492, 19 for 328 in (b)). Best soft-penalty energies at target L: (a) 1.7e-3 (328), 2.4e-2 (497), 3.2e-3 (492); i.e. even the incumbent-derived starts never reach E=0 at ours-0.001.
Angle structure of best hardened results (mod 90 deg folded, bins 0-1.1, 1.1-5.7, 5.7-17, 17-46 deg) is in data/; (a) results stay mostly near-axis aligned with a few tilted blocks inherited from the incumbent; (b) stays two-angle (axis + band angle); (c) is amorphous, high energy.

## Verdict
No promise at mid n. No structure unlike the incumbent emerged; random starts (c) jam far from feasibility (same finding as poc/soft_to_rigid), band starts (b) are worse than the grid fallback-ish, and perturbed incumbents (a) relax back to worse variants of the incumbent. Estimated gain: 0 (no win; official `squares score` not run since nothing beat ours). Compute needed to be competitive is orders of magnitude beyond a numpy Metropolis (~5k moves/s).
