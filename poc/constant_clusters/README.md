# constant_clusters: leader L-K shared-constant clusters. Result: negative, nothing built

Estimated score gain from this POC: 0. No packing built, no CSV written, `merged.csv` untouched, no `squares score` run (nothing to score).
Data: `../leader_curve_analysis/docs/per_n_comparison.csv` (leader L_lb, ours), Pell cores `../pell_corrugation/data/border-scan/cores.json`.
Scripts: `poc/cluster.py` (clustering, tol 1e-8 on L_lb-K), `poc/roots.py` (root/core matching, border bound). Outputs: `docs/clusters.csv` (all members), `docs/lost_clusters.csv`.

## Clusters (>=2 members, tol 1e-8, n>=2)
31 clusters. Non-lossy ones (ours == leader, stock constants 0 / 0.5355 / 0.7071 / 0.657 / 1.0 and friends) are in `docs/clusters.csv`.
15 clusters have at least one n where we lose (all other clusters are ties):

| c = L-K | members (n) K,D | team | max gap | root n |
|---|---|---|---|---|
| 0.506591029 | 901 (30,1), 962 (31,1) | Cats | 0.0142 | 901 |
| 0.513434648 | 626 (25,1), 677, 730, 785 (K=26,27,28; D=1) | Cats | 0.0074 | 626 |
| 0.524647938 | 442 (21,1), 485 (22,1) | Cats | 0.0109 | 442 |
| 0.564693168 | 444 (21,3), 487 (22,3) | Achal Jain | 0.0067 | 444 |
| 0.565624195 | 680 (26,4), 733 (27,4) | Achal Jain | 0.0058 | 680 |
| 0.580898766 | 580 (24,4), 846 (29,5) | Cats | 0.0029 | 580 |
| 0.589053943 | 292 (17,3), 789 (28,5) | Cats | 0.0023 | 292 |
| 0.590016439 | 259 (16,3), 445, 488, 681, 734, 967 (K=21,22,26,27,31; D=4,4,5,5,6) | Cats | 0.0014 | 259 |
| 0.596043117 | 228 (15,3), 534 (23,5) | Subramanya / Cats | 0.0106 | 228 |
| 0.599535465 | 328 (18,4), 365 (19,4) | Cats | 0.0148 | 328 |
| 0.605884509 | 908 (30,8), 969 (31,8) | Cats | 0.0311 | 908 |
| 0.684143606 | 537 (23,8), 739 (27,10) | Cats / KuanHaoC | 0.0082 | 537 |
| 0.724372709 | 202 (14,6), 263 (16,7) | Cats | 0.0088 | 202 |
| 0.740410172 | 297 (17,8), 332, 409 | Cats | 0 (tie) | 297 |
| 0.880960372 | 70 (8,6), 179, 207 | Mesquare / Cats | 0.0004 | 70 |

(Lost-cluster list with exact members and per-cluster gain: `docs/lost_clusters.csv`.)

## Structure: every cluster is a root plus upright L-borders
Between members of one cluster, dn = 2*t*K_root + t^2 + (small extra), t = K_member - K_root. That is exactly the square count of an upright L-border of
thickness t around a core of side K_root + c (strips t x floor(Lc+t) and t x floor(Lc)). Extra squares beyond pure border (D growth): 0 for clusters 1,2,3,5,6,14,15
(pure border, D constant), 1-3 for the others (e.g. 537 -> 739: +2, 292 -> 789: +2, 259 -> 967: +3). So the leader reuses "core + border" the same way we do; the border
extension is free and cheap, and the real content is the **root**, always the smallest member, found per-constant by bespoke optimisation.

## Core matching against our catalog (negative)
- Ours at any smaller n' with ours L-K equal to c (tol 1e-6): none. Nearest near-misses are not equal: n=102 (c=0.6058287 vs 0.6058845, 5.6e-5), n=37 (0.59862 vs 0.59954),
  n=230 (0.68293 vs 0.68414), n=123 (0.59138 vs 0.59002), n=227/258 (0.5636 vs 0.5647). All differ by far more than a shared-core constant would.
- Pell core catalog (cores.json, fractional part of L_corrugated for w=1..16): no fractional part within 2e-3 of any lost-cluster c except none (the nearest is w=13 0.69096 vs 0.68414 of 537/739, off by 0.0068; that core is our n=739 winner itself).
  Pell-17 cores live at L-K = 0.69-0.99 style constants, the lost clusters sit at 0.51-0.61 (D=1..8, different family) or the 0.684 case, so none of our cores supplies the root.
- General border bound over the whole incumbent: for every n, min over n'<n, t of L_ours(n') + t subject to n' + t*floor(L'+t) + t*floor(L') >= n. It beats our L at **0 of 1000 n**
  (`poc/roots.py`). So the incumbent is already border-closed: no border of any of our own packings can improve any n, and the lost clusters need a new root, not a border.
  (Bound is theoretical and ignores that a rotated-bbox core may not accept an upright border; it still gives zero candidates.)

## What would be needed
A root (unit-square packing) at: n=626 (L=25.513435), 901 (30.506591), 442 (21.524648), 444, 680, 328, 908, 537, 202, 259, 228, 292, 580 at the above constants. A single root
propagates by pure border to the other members: root 626 alone covers 677/730/785, 328 -> 365, 908 -> 969, 901 -> 962, 442 -> 485, 444 -> 487, 680 -> 733 (these members
fully inherit the root's constant with 0 extra squares). Upper bound on score if all 15 roots were solved and the members followed: sum(gap/sqrt n) = **0.0108**
(clusters 14 and 15 alone: 0.0015 + 0.0019; the three D=1 roots at K=25/30/21: 0.0031). Not achievable here: no core exists in our catalog and the roots are not closed forms
(see k2p10_curve_geometry; quadratic search found nothing).

## Measured results
None. Measured L at all lost n: not available (nothing built). We hold the incumbent value at each n; leader values are the theoretical targets above, not measured by us.
