# Periodic frontier refit (negative)

## Idea

Use the public per-`n` frontier table only as a side target. Fit small
contact-driven periodic blocks to the incumbent's non-axis patches, let the
block phase and rectangle dimensions vary, and then look for a full-packing
seam assembly. This differs from the earlier rigid 45-degree strips because the
isolated patches are free-angle and the surrounding upright lattice is kept as
the end-piece fixture.

## Run

```sh
python3 poc/periodic_frontier_refit/poc/refit_probe.py \
  --targets 380,701,846,979 \
  --out poc/periodic_frontier_refit/docs/quick_summary.json
```

The baseline is `upstream/merge:merged-22.15325720501.csv` at tree
`238087b9787a14b8ad597d697a7319d604d74b5f`. The probe reads the four target
side values from `poc/frontier_nov12/docs/per_n_gap.csv` for context only:

| n | incumbent L | public frontier L |
|---:|---:|---:|
| 380 | 19.96135002381115 | 19.959412 |
| 701 | 26.91277200521607 | 26.910417 |
| 846 | 29.58378219015212 | 29.580899 |
| 979 | 31.76653660684230 | 31.763618 |

## Result

**No certified win and no useful seam assembly.** `final.csv` is therefore
still header-only. The reusable quick test records three checks:

1. The trusted full-packing joint x/y/angle SLP starts at the incumbent and
   returns the same side length for every target. There is no local fixed-angle
   or free-angle headroom along the existing contact graph.
2. The largest non-axis patches can sometimes be compressed in isolation. For
   example, at a 5% rectangle shrink the 37-square patches in `n=846` and the
   197-square patch in `n=979` remain overlap-free internally.
3. Replacing an isolated compressed patch in the incumbent does **not** produce
   a valid full packing: the patch moves into the former pocket region and
   collides with neighboring upright cells. Including nearby seam cells in the
   relaxation did not recover a strict side reduction either.

The useful negative observation is that periodic/rectangular compression of the
inner patch is real but is not the missing mechanism. The obstacle is the finite
seam/end-piece assembly, consistent with the earlier `strip_cell`,
`pell_corrugation`, `wildcard_round3`, and `rectangle_gadgets` findings. A later
attempt would need an explicit gadget seam search or a different end-piece
family; simply compressing the fitted interior block is not enough.

Machine-readable measurements are in `docs/quick_summary.json`.
