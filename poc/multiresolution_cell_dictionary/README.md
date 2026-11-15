# Multiresolution cell dictionary

## Idea

A local cell is a rigid fragment of a certified packing, cropped from a
horizontal band and normalized to its exact occupied bounding box.  The
dictionary records the square count, width, height, source, orientation and
wall context of each fragment.  A target is assembled by selecting one
fragment per band and stacking the fragments in disjoint vertical intervals.

Unlike a rectangular strip or a one-layer corner gadget, a dictionary cell may
contain several orientations and offset upright blocks.  Its exact geometry is
retained; only rigid translations are applied.  This is therefore a
coarse-to-fine cover test with an exact coordinate realization, not a warm-start
LP or an inferred capacity formula.

## Quick probe

The probe reads only the in-repository canonical packing and constructs cell
dictionaries from compatible high-`n` rows.  It tests 8, 10 and 12 horizontal
bands, with the eight dihedral transforms of each source.  For each band it
keeps the shortest distinct cell for each exact square count.  A dynamic
program then finds the exact count and minimum total height attainable with one
cell per band.

```sh
PY=/Users/ocean/Documents/Squares/Squares_in_Squares_Claude_handoff_20261111/.venv/bin/python
PYTHONPATH=poc/baseline/poc/src $PY \
  poc/multiresolution_cell_dictionary/poc/probe.py \
  --targets 803,791,908 --delta 0.0001 \
  --out poc/multiresolution_cell_dictionary/docs/results.json
```

A candidate is exported only after it contains exactly `n` squares, is inside
the target box, and is strictly shorter in both literal-decimal and actual
pandas `read_csv`/`to_csv` transport.

## Results

**Quick probe: negative.** The probe targeted
`L_inc - 1e-4` and tested 8, 10 and 12 bands with all eight dihedral
transforms. The exact DP found no cover with at least `n` full squares inside
the target side.

| n | target L | 8 bands | 10 bands | 12 bands | best |
|---:|---:|---:|---:|---:|---:|
| 803 | 28.8171658768 | 672 / 24.0000 | 562 / 21.5453 | 648 / 24.0734 | 672 |
| 791 | 28.6275357585 | 672 / 24.0000 | 560 / 20.0000 | 648 / 24.0734 | 672 |
| 908 | 30.6275357585 | 722 / 25.5735 | 900 / 30.0000 | 720 / 24.0000 | 900 |

Each table entry is `maximum assembled square count / stacked height`. The
closest result was `900/908` at 10 bands, using only `30.0 < 30.6275` of the
available height. The deficit is therefore not a tuning miss: independently
stackable cell rows discard every source square that crosses a band boundary.
On these jammed layouts that cutoff cost is at least eight squares even where
the remaining height is sufficient.

This closes the quick band-cut dictionary. It does not prove that overlapping
macrocells or seam-stitched cells are impossible; either would require a new
mechanism that preserves squares crossing cell boundaries. No complete
dual-certified strict winner was produced, so `final.csv` remains header-only.
