# Post-merge gap refresh

## Idea

Every newly merged POC can introduce packings whose contact solver stopped
with a small positive separating-axis gap. Scaling all square centers toward
the exact bounding-box centre preserves unit size and every rotation, and is
overlap-free while the smallest pair ratio `gap / centre_distance` remains
positive. This pass reopens that slack after the latest merge instead of
assuming an older shrink result still applies.

## Method

For each complete `n` group in the current `upstream/merge` artifact:

1. Compute the exact separating-axis gap over every nearby square pair.
2. Derive the maximum safe uniform centre scale.
3. Emit a conservative candidate, convert to the public `id,x,y,deg` format,
   and certify the literal file plus its actual pandas round trip.
4. Export only complete groups whose certified `L` is strictly below the
   current merge in both transports.

## Result

Win. Against `upstream/merge` (`be377e1`, artifact
`merged-22.15330692498.csv`), **80 complete groups** are strictly smaller in
both literal and actual pandas certification.

The uniform pass reduced the score by
**`6.032982278591945611098263809e-7`**.

Largest side reductions:

| n | dL |
|---:|---:|
| 694 | `1.283814308164005e-6` |
| 736 | `1.173315027486778e-6` |
| 915 | `8.941957746037843e-7` |
| 632 | `7.808322926878865e-7` |
| 542 | `7.723063168998752e-7` |

All 80 groups were reverified again after conversion to `id,x,y,deg`, in which
the `deg` column is degrees and is converted back to radians by the union
parser. The exported file contains only strict wins and passes the repository's
literal final-CSV structure check.

### Follow-up fixed-angle LP

A 12-round, no-trust-region fixed-angle LP was then run over the full 1,000
groups in the current upstream merge. It found 28 strict groups, including
three outside the softened set (`n=333`, `740`, `861`). The tested LP reduction
on those complete groups is
**`1.515274578189450651025466411e-6`**.

After merging the two stages, there are **81 strict groups**. The projected
union score is **`22.15330461280231`**, down by
**`2.3121733860875793e-6`** from the current merge. `docs/lp-wins.json` and
`docs/full-lp-wins.json` contain the LP rows, and `docs/lp-merge.json` records
the replacement set.
