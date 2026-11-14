# Corner-lattice ownership borrow

## Idea

Several nearby high-gap groups may share repeating upright/tilted blocks.  The
probe starts from the certified incumbent group with one fewer square (`n-1`)
and allows a bounded set of squares to leave their local ownership window.  The
remaining packing stays rigid.  The freed seam is then refilled by the unchanged
exact upright MILP from `poc/highn-mondrian/poc/fillfast.py`, rather than a
rectangular crop or fixed donor slice.

The side targets come only from the public per-n table
`poc/frontier_nov12/docs/per_n_gap.csv`; no coordinate data from that source is
used.  The donor coordinates come from the trusted canonical incumbent.

## Scope

This was the required 15-minute screen.  It tested donor `n-1` for
`682, 744, 844, 877, 979` with:

- `1x1`: 80 center-first single-square releases per target;
- `2x2` and `3x3`: 30 deterministic center-first local windows on `877` and
  `979`;
- exact MILP refill with a 3 second kernel time limit per probe.

Before the scripted runs, exploratory one-square releases on all five targets
and adjacent-pair releases on `682, 744, 844, 979` also had no hit.  These are
bounded local scans, not exhaustive ownership searches.

## Run

```sh
/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python \
  poc/corner_lattice_borrow/poc/release_scan.py \
  --targets 682 744 844 877 979 --windows 1x1 --limit 80 --milp-seconds 3 \
  --output poc/corner_lattice_borrow/data/single_release.json

/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python \
  poc/corner_lattice_borrow/poc/release_scan.py \
  --targets 877 979 --windows 2x2 3x3 --limit 30 --milp-seconds 3 \
  --output poc/corner_lattice_borrow/data/quick_release.json
```

## Result

No packing reached the target count.  Every scripted local refill stayed at the
donor count, exactly one square short:

| n | donor | windows | probes | best filled | short |
|---:|---:|---|---:|---:|---:|
| 682 | 681 | `1x1` | 80 | 681 | 1 |
| 744 | 743 | `1x1` | 80 | 743 | 1 |
| 844 | 843 | `1x1` | 80 | 843 | 1 |
| 877 | 876 | `1x1`, `2x2`, `3x3` | 140 | 876 | 1 |
| 979 | 978 | `1x1`, `2x2`, `3x3` | 140 | 978 | 1 |

Because no candidate reached its required group size, none was eligible for
geometry certification and `final.csv` remains correctly header-only.

## Negative findings

- A one- or two-directional local release is not enough to break into the
  adjacent packing topology in these groups.  The freed seam refills, but it
  does not create the missing square.
- This does not falsify the broader hypothesis.  It only shows that releasing
  one square or small center-first upright windows, with all other donor squares
  rigid, gives no sign of improvement on the five selected high-gap targets.
- A productive follow-up would need whole rigid block ownership changes or
  continuous contact repair, not more isolated single-square releases.  That is
  outside the quick-test budget.

The target remains far away: one small local win at the observed side gaps would
change the total score by only about `1e-4`, while the team target needs at least
`0.053` in total.
