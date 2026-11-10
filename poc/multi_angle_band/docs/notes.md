# Notes

Top-10 high-n targets (rank_high_n_gaps.py logic rerun in memory against current merged.csv; full top 25 in top25_ranking.json):
626 (gap 1.04e-3), 634 (9.4e-4), 677 (9.4e-4), 730 (8.5e-4), 785 (7.7e-4), 687 (7.3e-4), 842 (7.0e-4), 686 (6.8e-4), 901 (6.4e-4), 971 (6.1e-4).
Note vs older high_n_gap_ranking.md: 634 moved to #2, 738 dropped to #14 (incumbent for 738 improved), 687/686 entered the top 10.
gap = L/sqrt(n)-1 minus median of +-5 neighbours. It is an excess, not an achievable gain.

Method (poc/mab.py): candidate pool = upright squares on 4 corner-anchored lattices + tilted squares of two diagonal-strip segments
(independent angle ~45deg +- 0.2 rad, independent lattice phase, off-center strip offset c, width w). HiGHS MILP picks the max
non-overlapping subset (pairwise SAT conflicts, 1e-7 contact tolerance). Random config search, 6 s MILP limit, ~90-100 s per n.

Raw results (count of squares placed at container L; "short" = n - count):
- L = L_inc (n 626, 634): 625 (=25^2), 629 (short 5).
- L = L_inc + 0.1: 626 -> 625, 677 -> 676, 730 -> 729, 785 -> 784 (all = k^2, short 1); 634 -> 633 (short 1).
- L = L_inc + 0.15: 626 -> 628, 634 -> 634.
- 0.2/0.3/0.4: 626 -> 631/634/637; 634 -> 637 at 0.2.
Official-metric checks (poc/certify.py, centre dilation 1.0000002, literal and pandas round trip, single n):
- n=626: L = 25.67081993 (incumbent 25.52081528, +0.1500, worse); 
- n=634: L = 25.85711197 (incumbent 25.70710678, +0.1500, worse).
Both are valid, non-grid (L < ceil(sqrt n)=26), and both lose to the incumbent. No full-CSV candidate was built since nothing beat an incumbent.

Why it fails: at the incumbent L the pool never even reaches k^2+1; the incumbent k^2+1 packings (L=k+0.5208) use Pell-17 sliding/non-lattice
tilted chains, not a rigid lattice with a free phase. A rigid lattice band loses ~0.15 of side length. Fix would need continuous slides, i.e. the already-failed free-angle re-polish / hinged-band territory.
