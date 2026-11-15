# Codex coordinator ledger

Role: coordinator only. This folder records hypotheses, rankings, assignments,
and outcomes. Experiments are run by one subagent per hypothesis in a separate
POC folder; the coordinator does not run packing searches.

## Ground truth

| quantity | value |
|---|---|
| synced upstream `main` | `b5ad5368c408e5940d8e1fc6a62c2cbcf243c47f` |
| merge artifact | `merged-22.14341046689.csv` |
| certified score | **22.143410466888355** |
| goal | **<= 22.0999** |
| remaining required reduction | **0.043510466888355** |
| median-aware leader snapshot | `poc/leader_curve_analysis/data/perpuzzle_best.csv` |

The fresh audit uses the synced canonical
`poc/pell_corrugation/data/border-scan/merged.csv`. The leader snapshot is used
only to rank targets and is not source geometry. The largest current
per-`n` gaps are:

| rank | n | K | gap score |
|---:|---:|---:|---:|
| 1 | 803 | 28 | 0.0008230309 |
| 2 | 328 | 18 | 0.0008190416 |
| 3 | 791 | 28 | 0.0007550433 |
| 4 | 908 | 30 | 0.0007218406 |
| 5 | 969 | 31 | 0.0006987508 |
| 6 | 365 | 19 | 0.0006972490 |
| 7 | 861 | 29 | 0.0006935237 |
| 8 | 367 | 19 | 0.0006610732 |
| 9 | 747 | 27 | 0.0006184778 |
| 10 | 804 | 28 | 0.0006028869 |

The positive gap mass is spread widely. A single-pocket or local-polish result
cannot reach the target; hypotheses are ranked by whether they can produce a
reusable family or a large nonlocal basin change.

## Already saturated

The repository READMEs and coordinator ledgers rule out spending another cycle
on:

- fixed-angle LP, Lipschitz SLP, and ordinary lpslack polish;
- single-square deletion/insertion, uniform shrink, and ordinary neighbor
  deletion;
- rigid 45-degree bands, simple off-45 bands, and straight rational zippers;
- one-corner upright lattice plus one tilted band;
- rectangular pocket composition, proper-pocket lift saturation, and the
  existing polish/re-cut cycle;
- one-layer L-border cascade and selected donor-child transplant;
- local masked melts, ordinary mechwalk, and random-start inflation;
- public notebook/dataset harvesting and any Kaggle submission.

## Cycle 1 hypotheses

All five are fresh relative to those dead ends. Each subagent must refine the
construction if its first quick probe reveals a simpler falsification, but it
must not fall back to a saturated local search.

### H1 `heterogeneous_lattice_mip`

**Rank:** 1. **Expected gain:** 0.003-0.010 across the K=28..31 family.

Generate candidate squares from many independently phased and oriented local
lattices, then solve one global mixed-orientation set-packing problem at a
target side. The key difference from earlier band searches is that orientation
and phase are heterogeneous over the box and selected jointly under exact
non-overlap, rather than assigned to one rectangular pocket or one diagonal
band.

Quick targets: `803, 791, 908`. Probe 1-3 `n` first; scale if a target reaches
within `1e-3` of its incumbent or produces a strict certified win.

### H2 `rigid_cluster_crossover`

**Rank:** 2. **Expected gain:** 0.001-0.005 through nonlocal basin exchange.

Decompose two nearby packings into contact-connected rigid clusters, align
equivalent wall and seam interfaces, and exchange one large cluster while
retaining the complementary cluster. Repair only the resulting seam. This is a
two-parent topology crossover, not a fixed donor crop or one-way motif
transplant.

Quick targets: the top three `803, 791, 908`; preferred donors are nearby
current rows with the same `K` or adjacent `K`. A probe should stop if both
parents collapse back to their original basins.

### H3 `continuous_rowstate_transfer`

**Rank:** 3. **Expected gain:** 0.002-0.008 across many high-K rows.

Replace the failed quantized skyline DP with an event-driven transfer over
actual square contacts. Row states include free angle, fractional x-offset,
left/right wall ownership, and a small vector of seam occupancies. Optimize
exact count versus height, then compact the assembled layout. This targets
nonperiodic transitions between two or three local orientations.

Quick targets: `861, 747, 804`. Show an improvement in retained-square count
at the candidate side before attempting certification.

### H4 `face_cycle_flip`

**Rank:** 4. **Expected gain:** 0.0005-0.003 over jammed high-K targets.

Build the planar contact graph, find short faces that carry a full square of
free area, and flip the ownership of boundary squares across the face while
preserving all other contacts. This is a discrete cycle operation; it is not a
point relocation, a single-square deletion, or a directionless mechwalk.

Quick targets: `969, 908, 791`. A useful probe is a feasible flipped face that
leaves a coherent second move, even before a certified reduction.

### H5 `multiscale_corner_tiles`

**Rank:** 5. **Expected gain:** 0.001-0.004, especially for the 328/365/367
cluster and high-D corner families.

Enumerate boundary tiles formed by two or three staggered angle layers, each
with its own period and wall attachment, and compose them with an exact
integer-capacity DP. Unlike earlier single-layer rational zippers or rigid
corner gadgets, the tile may contain multiple layers and a sloping internal
interface.

Quick targets: `328, 365, 367`. A tile must beat the current side or supply a
new reusable capacity state clearly different from the failed one-layer
staircase families.

## Outcome log

Cycle 1 launched after the `b5ad536` sync. Outcomes are appended here as agents
return. A negative result must have its README, header-only `final.csv`, and a
commit before its experiment folder is removed from the working tree.

- **H1 `heterogeneous_lattice_mip`: negative.** The quick probe at
  `incumbent_L - 0.0002` reached only 756/803, 756/791, and 872/908 squares.
  The global candidate pool for 803 reduced to the ordinary `28 x 28` grid;
  the short HiGHS run did not find a competitive basin. This sampled variant is
  closed because it has no one-square near miss. Failure commit: `ac278b6`
  (preserved through `3e0d974`).
- **H2 `rigid_cluster_crossover`: negative at the structural gate.** The
  largest incumbent contact components at 803/791/908 are 37/10/8 squares.
  Adjacent and same-`n` parents did not contain matching replaceable large
  components; the 803 alternative has the same 37-square bounding box.
  A 4.4-5.1M-placement completion probe also failed to place the missing
  square. Repairing with the known local stack would have violated the
  hypothesis. Failure commit: `732a2b3` (preserved through `8d74bba`).
- **H3 `continuous_rowstate_transfer`: negative for the vertical ribbon
  decomposition.** At sides no larger than the incumbent, the best three-ribbon
  assemblies held 842/861, 729/747, and 784/804 squares. Adjusted angle and
  phase did not overcome the fractional side-ribbon clipping. This is a
  defensible structural stop, not a general proof against all row-state models.
  Failure commit: `e64f8f5` (preserved through `ad53584`).

## Cycle 2 hypotheses

These are opened while H4/H5 finish so the worker pool stays occupied. None is
a rename of a failed local method: each asks for a different shared-state or
multi-move search mechanism.

### H6 `parametric_frontier_continuation`

**Rank:** 1. **Expected gain:** 0.004-0.012 through reusable family wins.

For a fixed `D = n-K^2` family, treat the incumbent sides and construction
parameters as samples along `K` and fit a shared parametric geometry across
several adjacent `K`. Re-opt phase, angle, wall ownership, and end-cap lengths
together, preferably forcing a nested/monotone family. The unit of success is
a construction that improves multiple family members, not a one-off polish.

Quick targets: `D=7/8` high-K members and the `328/365` pair.

### H7 `optimal_transport_basin_transfer`

**Rank:** 2. **Expected gain:** 0.001-0.005 from nonlocal basin transport.

Use an unbalanced optimal-transport map between packings at related `n` to
move whole center/angle mass into a lower-side basin while inserting or
removing the cardinality difference at the largest geometric voids. This is a
smooth many-square continuation, not a donor crop, single-parent deletion, or
rigid-cluster exchange.

Quick targets: `803, 791, 908` with adjacent-`n` and same-family parents.

### H8 `dual_cycle_compensation`

**Rank:** 3. **Expected gain:** 0.001-0.004.

Search pairs of contact-face cycle flips whose individual vacancies are
infeasible but whose combined ownership swap is feasible. Require the two
cycles to share a seam and replace the resulting combined topology with a
certified local solve. H8 starts only if H4 shows a near-feasible single flip.

### H9 `void_density_triage`

**Rank:** 4. **Expected gain:** 0.001-0.006 in aggregate.

Run an all-gap audit for exact placements of the next one or two squares into
the current box, including rotated and fractional-position placements. Rank
rows by certified multi-square insertion potential before attempting repair.
This targets many small wins across the remaining mass rather than a single
large family.

### H10 `contact_graph_evolution`

**Rank:** 5. **Expected gain:** 0.002-0.008 if a new basin is found.

Represent a packing by its contact graph and orientation labels, evolve graph
rewrite operations under exact non-overlap penalties, then realize the best
graphs as coordinates and compact them. The search is over topology families,
not coordinate mutation alone.
