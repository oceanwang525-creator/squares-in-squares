# Codex coordinator ledger

Role: coordinator only. This folder records hypotheses, rankings, assignments,
and outcomes. Experiments are run by one subagent per hypothesis in a separate
POC folder; the coordinator does not run packing searches.

## Ground truth

| quantity | value |
|---|---|
| synced upstream `main` | `ddcef3407faba663d2dd994b4b014fbd95dfbbd3` |
| merge artifact | `merged-22.14333595647.csv` |
| certified score | **22.14333595647** |
| goal | **<= 22.0999** |
| remaining required reduction | **0.04343595647** |
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
- **H5 `multiscale_corner_tiles`: negative.** Across 180 exact MIS probes at
  `L_inc - 1e-4`, two/three-layer boundary tiles held only 44-46 squares while
  the same strip of upright grid held 72-76. The best combined layouts held
  324/328, 361/365, and 361/367. The observed per-layer sawtooth loss is too
  large to justify a larger DP. Failure commit: `dafaff9` (preserved through
  `879dd49`).
- **H4 `face_cycle_flip`: negative.** The sampled contact graphs are nearly
  fully triangulated (1,813-2,362 elementary faces at 908/969). Across
  672/633/780 transforms, 22/22/35 candidates were valid at only
  `L_inc - 2e-5`, but every one re-jammed to the incumbent; the best apparent
  difference was `2.4e-15`, below the strict-win gate. Failure commit:
  `8f21585` (preserved through `2116b2a`).

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

### Cycle 2 outcome

- **H7 `optimal_transport_basin_transfer`: POSITIVE.** A Sinkhorn transport
  from `n=789` into the `n=791` basin produced a strict dual-certified win.
  The measured side falls from `28.627635758531549059...` to
  `28.627533748315897057...`, so `dL=-1.020102156488e-4` and the official
  score gain is `-3.627065712784e-6`. The complete 791-square group passed a
  fresh coordinator re-check in both transports. Two 6-target scale waves
  (`790,792,804,848,861,969` and `328,365,367,456,747,982`) found no second
  win. Commit: `967402b`. PR: #134.
- **H6 `parametric_frontier_continuation`: negative.** No shared rational-band
  `(p,q,W)` template reached any tested `D=7/8` row below its incumbent; the
  closest screen was still 2-7 squares short and expanded to 98 overlaps at
  `n=791`. The `328/365` pair also required different parameters and remained
  4/5 squares short. Failure commit: `9c580f5` (preserved through `eacd46c`).
- **H8 `dual_cycle_compensation`: negative.** At 969/908, 40 nearby face pairs
  per target yielded 480 transforms and 64 fully re-jammed candidates each.
  The best penetration at `L_inc - 2e-5` was only `1.16e-8` and `7.34e-9`;
  no pair produced a complementary vacancy or a lower valid side. Failure
  commit: `d1066bd` (preserved through `8eec198`).
- **H9 `void_density_triage`: negative for direct card-add.** At
  `L_inc - 1e-7`, 24.7-70.0M angle/position checks per target found no legal
  insertion into the canonical `n-1` row for 803/328/791. For 803, the first
  step of the `n-2 + 2` route also failed after 64.3M checks. This closes the
  direct insertion audit, not arbitrary rearrangement. Failure commit:
  `50f11d1` (preserved through `17baee8`).

## Cycle 3 hypotheses

### H11 `crossn_graph_motif_alignment`

**Rank:** 1. **Expected gain:** 0.001-0.005.

Match repeated labeled subgraphs (orientation classes, wall attachments, seam
roles) between packings at different `n`, not merely contiguous coordinate
components. Reassemble aligned motifs and certify the complete group. This is
a graph-edit continuation across `n`, distinct from H2's cluster exchange and
H10's same-`n` single-edge rewrites.

Quick targets: `803, 791, 908`.

### H12 `multiresolution_cell_dictionary`

**Rank:** 2. **Expected gain:** 0.002-0.006.

Build a dictionary of exact local cell states from the current portfolio, with
wall, seam, empty, and orientation signatures. Solve a coarse cell-cover DP
followed by exact coordinate realization. The coarse states are reusable
macrocells rather than rectangles or one-layer gadgets.

### H13 `symmetry_ensemble_average`

**Rank:** 3. **Expected gain:** 0.001-0.004.

Take dihedral copies of several asymmetric high-performance packings, align in
a common frame, and build a consensus geometry by contact-weighted medians.
Break degenerate averaging only where the resulting contact pattern remains
realizable. This is ensemble continuation, not a single-parent SLP.

### H14 `boundary_slack_sculpt`

**Rank:** 4. **Expected gain:** 0.0005-0.003.

Parameterize the four wall profiles and move slack in coherent waves across
chains of wall-adjacent squares. Use exact collision events to change ownership
of wall contacts rather than applying uniform dilation or random jitter.

### H15 `basin_switch_dag`

**Rank:** 5. **Expected gain:** 0.001-0.004.

Enumerate short sequences of already-measured feasible topology moves and treat
them as directed basin edges. Search two- and three-move paths whose
intermediate penetrations are nonzero but decreasing, then certify only the
endpoint. The search is a DAG over mechanisms rather than a fresh random walk.

### Cycle 3 outcome

- **H11 `crossn_graph_motif_alignment`: negative.** Across 208/280/200
  motif re-jam attempts for 803/791/908, the best 803 penetration was
  `3.846e-4`; the other targets returned to their incumbent side (only a
  `2.3e-15` apparent difference at 791). Shared star motifs largely encode
  repeated local lattice symmetry rather than portable capacity. Failure
  commit: `5015828` (preserved through `6e18afa`).
- **H10 `contact_graph_evolution`: negative.** At `L_inc - 1e-4`, 48
  single-edge rewritten graphs per target were relaxed and optimized. Their
  best maximum penetrations were `5.84e-6`, `5.62e-6`, and `6.76e-6`, but the
  best sparse cases for 803/791/908 still had 768/886/2,561 penetrating pairs.
  These are dense conflicting states, not repairable seams. Failure commit:
  `1c74a12`.
- **H12 `multiresolution_cell_dictionary`: negative for disjoint band cells.**
  With 8/10/12 bands and all dihedral transforms, the best covers were
  672/803, 672/791, and 900/908. The 908 case had spare height but lost at
  least eight squares at band cutoffs. Failure commit: `3595989`.
- **H13 `symmetry_ensemble_average`: negative.** Dihedral contact-weighted
  averaging of same-`n` alternatives produced 1,557 overlap pairs at 803,
  715 at 328, and 2,170 at 908. The closest candidates were copies/micro-
  perturbations, not complementary basins. Failure commit: `5365c5d`.
- **H14 `boundary_slack_sculpt`: negative.** Across 144 coherent wall waves
  per target, the best 803 residual was `1.65e-8`, but full re-jamming and
  fixed-angle LP returned the incumbent side. At 908/969 the best residuals
  were `1.67e-5`/`2.11e-5`. No one-square near miss survived. Failure commit:
  `9106a20`.

## Cycle 4 hypotheses

These are follow-ons to the one verified H7 mechanism, but each changes a
search variable that can produce a distinct basin. They do not repeat a failed
local polish stack.

### H16 `ot_multidonor_barycenter`

**Rank:** 1. **Expected gain:** 0.001-0.005.

Build an unbalanced barycenter from several donors on both sides of a target
row, with donor weights optimized for wall/seam fit. H7 used one donor at a
time; mixed donor mass can preserve a good bulk from one row and an endcap
from another.

### H17 `ot_residual_patch`

**Rank:** 2. **Expected gain:** 0.0002-0.002.

Start from H7's near-feasible transported candidates whose safe dilation
exceeded the side budget, identify the small set of penetration pairs, and
exchange one or two squares only inside that conflict patch. This avoids the
global dilation that killed the 328/367 near misses.

### H18 `oriented_mass_transport`

**Rank:** 3. **Expected gain:** 0.0005-0.003.

Transport square centers jointly with a folded-orientation label and a
small-cost angle ground metric. Group the resulting assignments into coherent
orientation patches before realizing coordinates. This tests whether
Sinkhorn's center-only matching destroyed useful angle structure.

### H19 `champion_transform_fanout`

**Rank:** 4. **Expected gain:** 0.0001-0.001.

Apply the exact transform family of the certified n=791 winner to neighboring
and same-`D` rows, allowing integer card-count shifts at the transformed
boundary. The donor remains our own construction; this is a verified-basin
fanout, not a public-coordinate transplant.

### H20 `contact_entropy_targeting`

**Rank:** 5. **Expected gain:** diagnostic, then 0.0005-0.003.

Rank near-feasible candidates by contact-label entropy and patch count rather
than dilation alone. Low-entropy conflicts should admit an attractor-local
rewrite; high-entropy conflicts are rejected before expensive repair.

## Cycle 5 hypotheses

H16 proves that local transport geometry can carry meaningful improvements
well beyond the first win. The next cycle keeps that mechanism while changing
how donor mass and target rows are selected.

### H21 `localized_donor_weights`

**Rank:** 1. **Expected gain:** 0.0002-0.003 across related rows.

Let multi-donor barycenter weights vary by wall quadrant or contact region
instead of using one global weight vector. One donor may supply the bulk while
another supplies only an endcap or seam.

### H22 `transport_sequence_beam`

**Rank:** 2. **Expected gain:** 0.0002-0.002.

Treat successive donor maps and small repairs as states in a short beam
search, keeping candidates by residual and topology diversity. H7/H16 use one
map then one repair; this tests two or three alternating map/repair steps.

### H23 `near_plateau_source_selection`

**Rank:** 3. **Expected gain:** 0.0001-0.001 in aggregate.

Choose transport sources by chord-count and seam compatibility among rows
whose side lengths are within `0.02`, rather than only by index distance. This
targets rows with close geometric plateaus but different cardinalities.

### H24 `transport_masked_melt`

**Rank:** 4. **Expected gain:** 0.0001-0.001.

Use the best transport seed as a topology, freeze its coherent bulk, and run a
masked soft-a melt only around the transport residual. This is a hybrid of the
proven transport seed and the previously effective masked-pocket melt.

### H25 `angle_class_reassignment`

**Rank:** 5. **Expected gain:** diagnostic, then up to 0.001.

After center transport, reassign squares to folded-angle classes by a global
minimum-cost matching, then rebuild only interfaces between orientation
classes. This avoids putting angle directly into the Sinkhorn metric, which
H18 showed is counterproductive.

### Operational outcome

- **OT top-gap fanout (`poc/ot_gap_fanout`): negative.** Twenty top-gap rows
  found no strict win in the first wave. Focused repair reduced `n=330` to only
  `8.36e-10` above incumbent, so a parameter-grid continuation was run. Its
  best zero-margin side was `18.692388155424336`, just `8.56e-13` below the
  incumbent, below the `-1e-9` win threshold; the next shrink retained
  `7.4e-11` to `1.6e-10` overlap. The row is locally closed for this family.
  Commits: `147ea03`, `f9188bf`. Folder removed after preserving both.
- **H22 `transport_sequence_beam`: negative.** Two transport/repair stages
  were screened on 803/367/328 plus 791/908 controls. Stage one sometimes
  reached low residual (e.g. `1.35e-8` at 791), but stage two destroyed the
  useful wall placement; all best safe sides exceeded incumbents. Failure
  commit: `8425350`.
- **H24 `transport_masked_melt`: negative.** Transport-seeded contact-ring
  melts reduced local penetrations to `8.8e-8..3.0e-7`, but the residual
  networks were large enough that safe dilation exceeded the target side on
  all tested rows, including 791/908 controls. Failure commit: `9f3b2e2`.
- **H25 `angle_class_reassignment`: negative.** Screening 36,288 starts per
  target after center transport still left connected interface overlaps.
  Best safe sides exceeded incumbents at 803/328/791; the 791 control needed
  `1.63e-3` extra side. Failure commit: `db5187d`.
- **H21 `localized_donor_weights`: POSITIVE.** A spatial field over the
  n+3/n+4 donor mixture lowered the certified 791 group from
  `28.626523948339...` to `28.626213659843...`, a further
  `dL=-3.102884960217e-4` and about `-1.103258882e-5` score. A 45,000-start
  extended wave was tie-only. Commit: `8880f33`; folded into PR #134.
- **H27 `localized_neighbor_band`: negative.** Localized donor maps at
  790/792/907 reached only `5.36e-5`, `4.63e-5`, and `3.23e-5` residual.
  Minimum safe dilation exceeded the incumbent by about `1.0e-3..1.5e-3`,
  so there was no one-square/near-feasible sign for the remaining neighbors.
  Failure commit: `42e4ac0`.
- **H28 `localized_topgaps`: negative.** Localized transport on
  969/861/747/804/848 found best residuals of `3.50e-5..8.08e-5`, but safe
  dilation missed the incumbent by at least `1.1e-3`. Failure commit:
  `1cdd0fa`.
- **H28 `localized_topgaps`: negative.** The H21 localized-weight sweep was
  applied to 969/861/747/804/848 with 27,000 starts per row. Best residuals
  were `3.50e-5..8.08e-5`; safe dilation exceeded every incumbent by at least
  `1.1e-3`. Failure commit: `1cdd0fa`.
- **H23 `near_plateau_source_selection`: POSITIVE.** Selecting donors by
  side-length/contact similarity chose 790 and 792 for `n=791` and produced
  a complete group at `28.626405023313754301...`. It improves the certified
  H16 row by `1.18925025816e-4`, and is a strict dual-transport win.
  A neighbor wave found no second row. Commit: `b36a567`; folded into PR #134.

### Cycle 4 outcome

- **H7 extended scale-up: POSITIVE, second row.** A 3,528-start sweep across
  `367,328,803,908,969,861` produced a strict `n=908` win from donor 912.
  The complete group certifies at `30.627535212069181892...` in both modes,
  `dL=-1.005464596e-4` and score gain about `-3.329e-6`. The other five rows
  were tie-only or above incumbent. Commit: `6126a2c`; folded into PR #134.
- **H16 `ot_multidonor_barycenter`: POSITIVE.** Mixing donors `n+2` and
  `n+3` at weights `1:2` improves the already certified 791 row again. The
  full group certifies at `28.627434433035796152...` in both transports,
  `dL=-9.931531524e-5` versus H7 and `-2.013255495746e-4` versus the
  upstream row. H16 adds about `-3.5326e-6` score on top of H7. It was folded
  into existing PR #134. Commit: `fc8aab4`.
- **H16 extended: POSITIVE, converged.** Four focused waves lowered the same
  791 row to `28.626523948339571162...`, a total `dL=-1.111810191976e-3`
  against upstream and about `-3.953141948e-5` score. The final wave was
  tie-only after 1,050 screened and 53 repaired starts. A six-row neighbor
  wave screened 211,680 starts and found no second row. Commit: `f6285f4`;
  folded into PR #134.
- **H17 `ot_residual_patch`: negative.** H7's near-feasible rows are globally
  distributed failures, not small end defects. At 328/365/367 the initial
  violations `9.51e-6/2.05e-5/1.63e-6` affected 767/726/593 pairs in one
  dominant component. Twenty-four separated patches per target changed the
  residual by at most about 1%, leaving it thousands of times above local
  feasibility. Failure commit: `216616b`.
- **H18 `oriented_mass_transport`: negative.** Joint center/folded-angle
  transport screened 90 maps and 1,800 starts per target, but every best
  candidate used the center-only control (`lambda_angle=0`). Orientation mass
  increased residual to `8.59e-4/3.40e-5/4.13e-4` and safe dilation exceeded
  the incumbent at all three targets. Failure commit: `6902bb6`.
