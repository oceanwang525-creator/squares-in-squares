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

Win. Against `upstream/merge` (`e3c11f7`, artifact
`merged-22.15326971562.csv`), **77 complete groups** are strictly smaller in
both literal and actual pandas certification.

Against the original merge snapshot, the uniform pass reduced the side-length
score by **`6.032982278591945611098263809e-7`**.

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
groups in the original merge snapshot. It found 28 strict groups, including
three outside the softened set (`n=333`, `740`, `861`). Its reduction against
that snapshot was **`1.515274578189450651025466411e-6`**.

Four groups from those stages (`796`, `853`, `909`, `972`) were superseded by
the newer upstream merge and were removed by `poc/rebase_current.py`. The
remaining **81 strict groups** were then re-certified against
`merged-22.15326971562.csv`; **77 are still strictly smaller in both
transports**.

The certified full-union score is **`22.15326747482295`** after the pandas
round trip (`22.153267474822947` literal), down by
**`2.24079705e-6`** from the merge-branch filename baseline. Both modes accept
all 1,000 groups. `docs/final-gate.json` is the current gate,
`docs/per-n.md` contains every exact per-`n` before/after length, and
`docs/lp-wins.json`, `docs/full-lp-wins.json`, and `docs/lp-merge.json` retain
the construction-stage evidence.
