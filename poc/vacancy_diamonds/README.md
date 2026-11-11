# vacancy_diamonds: two-diamond layout search at n=401. Result: negative, nothing built

Estimated gain: 0. No packing below L=20.535534 found, no CSV written, merged.csv untouched, no `squares score` run (nothing overlap-free below ours to score). Leader (20.526143) not reproduced.

## Incumbent layout (extracted from merged.csv n=401; `poc/extract.py`, picture `docs/inc401.png`, red = tilted)
- L = 20.535534 = 17 + 5/sqrt2. Squares: 351 upright (angle 0 mod pi/2), 49 tilted at 45deg (+-0.003), 1 off-lattice (angle 0.747 rad, i.e. 42.8deg, at (-6.03,3.00) in centred coords: it is the 25th square of diamond A, nudged).
- Two 5x5 rhombi (5 rows x 5 columns of 45deg squares, corner-to-corner span 5*sqrt2 = 7.07), centres (centred coords) A=(-6.0,3.0), B=(3.0,-6.0): symmetric about x=-y, centre offset 9 in each axis (integer, so the lattice phase is shared). Rows along u=(x+y)/sqrt2 are 5 squares each, tilted-lattice spacing 1.
- Upright 20x20 lattice = 400 squares fits in L=20.5355; slack 0.5355 per axis. Net gain +1 comes from the two diamonds (50 tilted) displacing 49 upright cells total... i.e. the pair of diamonds plus the seam staircases and corrugated top strip (rows x~0.7..9.7, y 7.7..9.8, angles off-lattice ~1e-3) net one extra square.
- Vacancies: visible white gaps are the four tips of each diamond (a ~0.5 x 1 notch at each tip, 8 total), one 1x~0.3 white strip between two column groups near the top (x~-0.3, y 7.5..10), a 0.3-high gap row at the right (x 7.5..10, y~0), a gap at (-1..0,-1.3) and the small notches along 45deg seams. Sum of vacancy area is small (each < 0.5 square).

## Experiments
1. Layout counting model (`poc/colmodel.py`, `poc/hybrid.py`, `poc/t1.py`): tilted a x b blocks plus upright columns that slide independently (exact DP over column offsets; peeling of full rows/cols where block-free; row or column model for the core). With the incumbent's two 5x5 blocks at the incumbent positions this model gives at most 394 (350 + ... plus 50 tilted... model total 394; actual layout holds 401). So the model misses ~7 squares that the real layout gets from rows+columns staggered in the same seam region; it cannot discriminate 401-capable layouts, and I did NOT use it to claim any count-vs-L table. No counts-vs-L table is reported because of that (no layout variant could be certified at >=401 by this model; the baseline no-diamond model gives 400).
2. Vacancy collapse / row-column shift test (`poc/slp.py`): sequential LP on the incumbent topology. All 401 squares free to translate (trust region 0.03, 0.0005..), container L variable, linearised separating-axis constraints from the current contact set (all pairs within centre distance sqrt2+0.2), walls. Angles of lattice squares (upright within 0.003 of lattice, tilted within 0.003 of 45) held fixed because relative rotation of aligned squares is a first-order |dtheta| kink that a linearisation cannot represent (my first attempt with free angles produced bogus dL=-tr steps with violation ~2 tr; discarded). Result: optimum dL = +2e-8 (= constraint margin 1e-9 times chain length), i.e. no first-order descent direction at all: the incumbent is locally rigid under any shift of rows/columns/diamonds by up to 0.03. Shifting whole rows/columns into vacancies cannot lower L because the vacancies are not on a load path that is slack; L is set by chains upright rows + diamond seam (17 + 5/sqrt2).
   Caveat: angle-free first-order moves (corrugated strip tilts, diamond fan) are not covered here; d1_chain showed penalty relaxation with free angles at L down to 20.526 has residual violation >= 8.7e-4 (no basin).

## Measured vs ours vs leader (n=401)
| | L | status |
|---|---|---|
| ours (merged.csv) | 20.535534 | overlap-free, exact |
| best found here | 20.535534 | no improvement (LP dL=+2e-8) |
| leader | 20.526143 | not reproduced; structure unknown |

## Not done
Enumeration of different diamond sizes/positions/third diamond with a counter strong enough to reach >=401; n=442 not touched; time ~10 CPU min. To go further a counter must allow row and column staircases simultaneously around a diamond (e.g. a MILP over lattice cells with 45deg seam cell sets), the column-slide model is too weak (394 vs 401 on the incumbent itself).
