# Dual-wall asymmetry LP

## Idea

Test whether high-gap incumbents contain a first-order move missed by
uniform shrink and all-square SLP: keep every square's orientation group fixed,
partition squares into orientation/spatial blocks, let those blocks translate
independently, and represent the four container boundaries separately while
minimizing the larger resulting width/height.  The quick test used
`n = 259, 414, 445` from the allowed list.  The public side table
`poc/frontier_nov12/docs/per_n_gap.csv` was used only to choose these side
targets; no coordinates or other data from it entered the solver.

The incumbent artifact is the upstream merge.  The carried union's unique group
list does not include these three `n`, so the merge sides are also the current
best-union sides for this quick test:

| n | incumbent L |
|---:|---:|
| 259 | `16.59137814549773602790309137238184190465154052676621` |
| 414 | `20.84066404810378346647400790352280059516729995306992` |
| 445 | `21.59137814549783261923132169162108279309693609978327` |

## Model

`poc/dualwall_lp.py` creates blocks in one of three modes:

1. `orientation`: one block per canonical square orientation group.
2. `anchor`: orientation group crossed by nearest-wall anchor.
3. `grid-2`: orientation group crossed by a centered 2x2 spatial cell.

Each block has one rigid `(dx,dy)` translation.  A linear program keeps one
separating axis per nearby pair and has independent left/right/bottom/top
extent variables.  Its objective is `max(width,height)`.  After each solve the
coordinates are re-emitted and checked by the unchanged official evaluator in
both literal and actual pandas transport modes.  No candidate is exported unless
both are strict wins.

## Commands

From the repository root:

```sh
git show upstream/merge:merged-22.15325720501.csv > poc/dualwall_asymmetry_lp/data/merge.csv

python3 poc/dualwall_asymmetry_lp/poc/dualwall_lp.py \
  --incumbent poc/dualwall_asymmetry_lp/data/merge.csv \
  --output poc/dualwall_asymmetry_lp/data/dualwall-margin1e-9.json \
  --wins poc/dualwall_asymmetry_lp/data/dualwall-margin1e-9-wins.json \
  --n-list 259,414,445 \
  --modes orientation,anchor,grid-2 \
  --rounds 6 \
  --trust none
```

The checked-in bounded result was produced with the same arguments except
`--output .../dualwall-bounded.json`, `--wins .../dualwall-bounded-wins.json`,
`--rounds 4`, and `--trust 1e-7`.

As a control, the existing trusted all-square fixed-angle LP was run with:

```sh
python3 poc/alln_fixed_lp/poc/alln_fixed_lp.py \
  --incumbent poc/dualwall_asymmetry_lp/data/merge.csv \
  --report poc/dualwall_asymmetry_lp/data/base-report.json \
  --output poc/dualwall_asymmetry_lp/data/unbounded.json \
  --wins poc/dualwall_asymmetry_lp/data/unbounded-wins.json \
  --n-list 259,414,445 --rounds 12 --jobs 3
```

## Result: negative

The all-square fixed-angle control found 0 wins in 0.8 s.  Its LP sides were
above the incumbents by approximately `1.73e-8`, `2.16e-8`, and `2.22e-8` for
`259`, `414`, and `445` respectively, consistent with active constraints plus
the control's `1e-9` pair/wall safety margins.

The dual-wall runs also found **0 strict dual-transport wins**.  With the final
`MARGIN=1e-9`, every one of the nine orientation/anchor/grid tasks was
LP-infeasible.  During model debugging, the zero-margin LP was feasible but
produced only ties or candidates rejected as certified overlapping squares; a
`1e-12` margin behaved similarly.  The checked-in bounded run at `trust=1e-7`
found 0 wins: one literal tie, seven certification failures, and one
LP-infeasible task.

Detailed machine output is in `data/dualwall-margin1e-9.json`,
`data/dualwall-bounded.json`, and the older diagnostic JSON files.  Because
there is no certified strict win, `final.csv` remains header-only.

## Honest findings

This was a narrow subspace test, not a proof that all possible dual-wall
decompositions fail.  The block choices were orientation, nearest-wall anchor,
and a coarse 2x2 spatial split; other cutsets could behave differently.  The
evidence does not support spending more time on this family for these three
targets.  The zero-margin certification failures also show that this block LP's
first-order contact approximation is not automatically safe enough for exact
candidate export.
