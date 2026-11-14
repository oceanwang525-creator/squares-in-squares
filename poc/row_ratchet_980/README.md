# Row ratchet on n = 980

## Idea

n = 980 (31² + 19) is two tilted "arms" around two rigid grid blocks. Two tools are combined: the team's contact-mechanism walk (`poc/swarm_mechwalk`), which moves the packing into a different contact arrangement, and a sequential-LP squeeze of the whole row (`poc/pslp.py`, from `poc/pocket_squeeze`), which then removes the slack that arrangement has. Either alone stalls; alternating them ratchets.

## Method

1. `poc/pslp.py` – full-row SLP squeeze (free angles, kink-safe rows, trust-scaled LP, final violation about 1e-15).
2. `poc/mechwalk_run.py` – runs `poc/swarm_mechwalk/poc/run_gate.solve_target` on one n from an external start, writing only to a scratch directory.
3. `poc/ratchet.py` – mechanism walk → squeeze → repeat while it improves.
4. `poc/row980.py` – reduced model for arm rows: only squares near the arms are active, the bottom-left grid block is static and the top-right block translates with the wall (546 squares instead of 980; 5 s per squeeze instead of 55 s; exact on the full packing).

## Result

| step | L (n = 980) | change |
|---|---|---:|
| main `772dfc2` | 31.779288329480 | |
| full-row squeeze, repeated to convergence | 31.779288320374 | −9.1e-9 |
| team mechanism walk from that start (round 3, `mmelt_r3_b1_m1_t5e-05_p_0.008`) | 31.779075417504 | −2.13e-4 |
| full-row squeeze of the walk result | 31.777807354190 | −1.27e-3 |
| second walk + squeeze cycle | **31.777719043335** | −8.8e-5 |
| third walk | no win | |

`final.csv` holds the last row. Certified from its exact bytes with the team gate functions, literal and pandas are both **31.777719043397**, a strict win of **1.569e-3** in L (5.0e-5 in score) against main `772dfc2`. The full 1000-group union gate with `poc/public_watch_nov14/final.csv` scores 22.147423335171034 literal / 22.147423335171037 pandas, 1000/1000 accepted, no regressions (`../public_watch_nov14/docs/full_gate.json`).

## Negative results on the same row (Nov 13–14)

- Team tools run on n = 980 without a win: arm-slack strip release (`poc/armslack_run.py` driving `swarm_armslack`; end strips are already flush, no valid result), `swarm_lpslack` (no change), and the earlier masked-pocket melt (`swarm_new159`, +1.9e-5).
- `poc/hoprow.py` – 631 shake-and-squeeze rounds on the reduced model (global jitter, local shakes, sliding a tilted row): best result 2.7e-7 above the start; no improvement.
- `poc/delrow.py` – delete each of 506 active squares of n = 981 and squeeze: best 31.795725, far above 31.7793.
- `poc/insrow.py` – insert a square into n = 979 at the 56 least-overlapping sites: first feasible result needs a box of 31.98.
- Composition of the team's cached bricks with extracted pockets: no pair reaches +19 below f = 0.7793 at K = 31.
