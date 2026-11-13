# Pocket squeeze: optimise pockets on their own, then recompose

## Idea

Most tracked packings are a K×K grid with a few rectangular *pockets* on the anti-diagonal seam; a pocket (w, h, +g, f) is a (w+f)×(h+f) rectangle holding w·h + g squares (`poc/rectangle_gadgets`). Every squeeze tool in this repo works on whole packings, where the box side is set by the tightest pocket. Working on pockets alone could:

1. find slack in a pocket that is not the binding one of its row,
2. improve a pocket once and reuse it at every n that contains it,
3. recombine pockets from different rows into rows nobody built.

## Method

- **`poc/pslp.py`** – sequential-LP squeeze of unit squares in a (w+f)×(h+f) rectangle, minimising f with free angles. Each LP linearises, for every nearby pair, the separating-axis gap on the currently best axis. The support function ½(|cos|+|sin|) is a max of smooth branches, and every near-active branch gets its own row, so a rigid joint rotation of an aligned block is representable. One shared elastic variable keeps the LP feasible when the iterate is slightly infeasible, so each step also repairs the second-order error of the previous one. Variables are scaled by the trust radius, which makes the solver tolerance shrink with the step; final violations are about 1e-15.
- **`poc/gadlib.py`** – extracts pockets: the minimal clean-cut rectangle around each tilted cluster of every tracked packing.
- **`poc/compose.py`** – composition DP and builder. m pockets fit a K+f box iff Σw ≤ K, Σh ≤ K and Σ(w+h+1) ≤ 2K+1; the builder places them on the seam and fills the rest with the two corner-anchored grids, giving K² + Σg squares.
- **`poc/slack_scan.py`, `poc/hop.py`** – squeeze every pocket alone; shake-and-squeeze basin hopping (global and local shakes).
- **`poc/seedgen.py`, `poc/factory.py`** – new pockets from scratch: a block of tilted rows, each row filled independently inside a rotated rectangle clipped to the pocket, plus upright fill; rigid scan, then squeeze.
- **`poc/sensitivity.py`** – lowers one pocket's f by 0.001 and re-runs the DP, to rank pockets by how many n they would improve.

## Measured

**Tools.**
- Five squares: converges to f = 0.707106781187 with the centre square at 45.000000°.
- Known pockets, loosened to f = 0.72…1.42 and perturbed, squeeze back to within 9e-6…2.4e-2 of the known f in about one second (15 runs; 2 return exactly).
- The DP reproduces all 555 non-trivial cells of the table, and the builder rebuilds n = 908, 631, 628, 535, 293 and 27 from their pockets with the right count, side and no overlap.

**Certified wins (dust).** Recomposing pockets from main `252fe8b` (and from the rows in `poc/public_watch_nov13`) beats the tracked row at **16 n, sum of per-n score gains 3.50e-8**: n = 232, 413, 442, 485, 500, 501, 530, 577, 632, 797, 854, 911, 913, 914, 971, 972. The largest are n = 413 (4.19e-7 in L, its own two pockets re-seated) and n = 797 (2.98e-7, using a pocket from n = 914). All 16 groups in `final.csv` were certified from its exact bytes with the team gate functions, literal and pandas, as strict wins: [docs/wins.md](docs/wins.md). These are the first officially certified packings from the composer.

**Negative results.**
- *Hidden slack* ([docs/slack_scan.txt](docs/slack_scan.txt)): of 337 extracted pockets, a plain squeeze lowers only three below the best known value of their type: (8,16,+6) 0.761536 → 0.760436, (11,15,+3) 0.637029 → 0.636881, and (8,22,+12) by 7e-6. Five more only return to a value already known from another row.
- *Basin hopping* ([docs/hop_40rounds.txt](docs/hop_40rounds.txt)): 40 rounds on each of 62 pocket types improve five. (11,15,+3) → 0.626015, (8,16,+6) → 0.759224, (23,10,+14) 0.860976 → 0.856630, (6,18,+8) 0.817966 → 0.817013, (22,8,+12) by 8e-7. The 56 others, including every pocket that binds many rows, do not move. Improved pockets are in `data/improved_pockets.json`.
- *Recomposition with those pockets*: 0 improved n (threshold 1e-7). None of the five is the binding part of any row.
- *Seed factory* ([docs/factory_9x15.txt](docs/factory_9x15.txt)): for the (9,15) pocket, rigid seeds reach +3 at f = 0.90 and +4 at 0.95, but squeeze poorly: of 2,500 seeds the best are +4 at 0.7643, +5 at 0.7785, +6 at 0.7945, +7 at 0.8485; a +3 seed reaches 0.777 after 12 shakes. The known (9,15,+3) is at 0.6087 and the library has (8,16,+6) at 0.7592. A slack-rewarding LP objective did not change where seeds jam.

## What the data says

- Our best non-45° pockets are blocks at 24–27° whose rows are slid along themselves (rows of 5, 7, 8, 7, 7, 8, …), so the block ends mesh with the upright squares; a few loose squares sit at the tips. A rigid block has the wrong split of squares between rows and corners, and no continuous motion repairs that.
- Wasted area (K+f)² − n at K = 31 is about 30 for the per-n leaders at every j = 1…12; ours was 30.5–32.3 on Nov 12. For the (13,16,+4) pocket that is the same 212 squares in a box about 0.03 smaller per side.
- [docs/sensitivity.md](docs/sensitivity.md): the classical 45° pockets rank first but are rigid. Among relaxed pockets, 0.001 off (8,11,+2) improves 11 n, off (13,16,+4) 7 n, off (9,15,+3) 5 n, each worth 2–5e-4 in score.

## Open

Row-level moves (slide a whole row by half a square, move a square between a row end and a corner) are the natural next shake operators; random shakes do not change the binding pockets.
