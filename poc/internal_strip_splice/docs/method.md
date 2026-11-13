# Search method

The first pass chooses a deleted slab on each axis. A square survives an axis
if its entire projected interval lies to the left or right of that slab. A
pair of cuts retains the intersection of the two survivor sets. After moving
the high-coordinate piece by the slab width, the two pieces meet along the
cut line. The original angles and the geometry within each of the resulting
four pieces are preserved.

For a fixed width, survivor changes occur at a square's upper projected edge
or at its lower projected edge minus that width. The implementation sweeps
these positions and deduplicates equal survivor masks. It compares the masks
directly on small instances. For larger products of cut counts it accumulates
joint deletion rectangles over the cut-pair table and uses inclusion-exclusion
to count survivors. The cuts leave at least 5% of the donor's squares fully on
each side so the scan exercises internal cuts.

The second pass takes the closest cut pairs and splits by square centers at
three offsets along each slab. It rejects placements outside the nominal
contracted box and finds cross-piece collisions using the separating axes.
For bipartite conflict graphs, the complement of a minimum vertex cover
gives a maximum independent set. For other graphs, a bounded binary MILP
selects a conflict-free subset. The MILP can return a feasible result without
proving its count maximal.

The adaptive pass permits a square within 0.6 units of either slab edge, or
inside that slab, to move with either piece along that axis. At the
intersection, a square can have four allowed translations. The MILP chooses
at most one placement per donor square and excludes every conflicting pair.
The enclosing rectangle and the original angles stay fixed.

The NumPy geometry is a search filter. Candidates that fit enough squares
and improve the float side by at least `2e-8` are reconstructed using decimal
coordinates from the donor. The shared official metric checks their literal
and default pandas round-trip forms, with small clearance dilations tested
when needed. A candidate has to reduce the certified side in both forms.
