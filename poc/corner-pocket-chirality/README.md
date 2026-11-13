# Corner-pocket chirality replay

## Idea

The earlier dual-pocket scan normalized each tilted pocket but transformed its
axis-aligned exit supports without applying the same translation. The supports
were therefore displaced before placement. This POC treats the pocket and its
supports as one rigid bundle and replays the highest-ranked old cells at the
current incumbent side.

The target cells are `971 A x A` and `972 A x 971 A`. The scan keeps the exact
tilted polygon geometry, enumerates all eight dihedral transforms and half-unit
channel phases, and refills the residual upright region with the exact MIS
implementation from `poc/highn-mondrian/poc/fillfast.py`.

## Result

Negative so far; no packing reached the `n=971` target and no `final.csv` is
included.

| template | scan | best count | short |
| --- | --- | ---: | ---: |
| `972 A x 971 A` | corrected rigid supports, all sym, half phases | 965 | 6 |
| `972 A(+0.25 deg) x 971 A` | fine phase and rotation sweep | 966 | 5 |
| `971 A x 908 A` | all 64 dihedrals, 16 phase pairs | 959 | 12 |
| `971 A x 908 B` | all 64 dihedrals, 16 phase pairs | 959 | 12 |
| `969 A x 969 A` | all 64 dihedrals, 16 phase pairs | 958 | 13 |
| `969 A x 969 B` | all 64 dihedrals, 16 phase pairs | 957 | 14 |
| `969 A x 970 A` | all 64 dihedrals, 16 phase pairs | 960 | 11 |
| `908 A x A` | all 64 dihedrals, zero phase | 955 | 16 |
| `908 A x A` | plus 4 half-grid phase pairs | 955 | 16 |

Using the complete `n=908` packing as a fixed core and exactly refilling the
new `n=971` box reaches `969/971`. A 60-window upright repack finds no local
gain, and grid searches for one additional square at six common pocket angles
find no legal placement.

Two further exact checks were negative:

- keeping only the 202 tilted `n=908` squares and refilling uprights peaked at
  955;
- removing only upright squares conflicting with a compact source motif peaked
  at 965/971 over 318 placements.

A wider motif AABB threshold produced 11,147 motifs and 33,441 tasks. The first
300 tasks still peaked at 965, so that run was stopped.

The exact local set-packing implementation initially reported an invalid 1014
because sparse constraint-matrix rows and columns were swapped. After that bug
was fixed, both the one-window smoke test and a corrected 300-window run stayed
at 969/971. The 1014 result must not be cited as a packing.

A candidate is not a win until the unchanged official metric passes both the
literal file and the pandas `read_csv` / `to_csv` round trip. The best
uncertified probe reached `966`, so no file in `data/` is a submission.

## Files

- `poc/chirality_scan.py` - corrected extraction, placement and replay.
- `poc/narrowed_pair_scan.py` - narrowed-pocket composition with process-level
  phase search.
- `poc/setpack_tail.py` - exact local set-packing around a high-count seed.
- `poc/motif_gap_scan.py` - compact source-motif placement search.
- `poc/core_motif_fill.py` - motif insertion followed by exact upright refill.
- `docs/LITERATURE.md` - paper-derived rationale and scope.
- `docs/NOTES.md` - implementation notes and measured commands.
- `data/` - small replay summaries; no large coordinates unless a win exists.
