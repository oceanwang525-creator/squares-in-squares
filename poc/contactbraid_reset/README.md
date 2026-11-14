# Contact braid reset (negative)

## Idea

Instead of deleting one square or relaxing the whole packing, find the shortest
active contact path joining opposite walls, treat its neighborhood as a coherent
braid, open that band at a slightly larger box, and let a gentle ILS reattach it.
The run then uses the trusted joint-angle SLP, fixed-angle LP tightening, and
official certification. A second bounded test compares the documented donor
strip-deletion/cross-cut reset on the same rows.

Only `poc/frontier_nov12/docs/per_n_gap.csv` side values were read as side
targets. No coordinates were taken from it, public notebooks, Kaggle, or public
datasets. All geometry began from the trusted canonical merged file.

## Baseline and targets

The stated team baseline is the union of
`upstream/merge/merged-22.15325720501.csv` and
`poc/carried_lpslack_wins/final.csv`, certified at `22.153257204712842`. The
checkout exposes that incumbent through `poc/pell_corrugation/data/border-scan/merged.csv`
plus the carried final. Target rows selected were `592` and `928`.

Geometric triage found that the active pair-contact graph is disconnected across
the box for several proposed targets. `592` has a 27-square left/right braid but
no bottom/top contact path; `928` has a 29-square bottom/top braid but no
left/right contact path. This is already an important correction to the original
one-row intuition: these packings are not one connected braid crossing the box.

## Run

```sh
python3 poc/contactbraid_reset/poc/braid_reset.py \
  --targets 592 928 --threads 4 --quick

python3 poc/contactbraid_reset/poc/braid_reset.py \
  --targets 592 928 --threads 4

python3 poc/contactbraid_reset/poc/crosscut_compare.py \
  --targets 592 928 --threads 4 --quick
```

The script compiles the existing trusted C++ helpers on first use. JSON reports
are in `docs/`.

## Results

No certified win. `final.csv` remains exactly header-only.

| n | incumbent L | frontier side target | braid outcome |
|---:|---:|---:|---|
| 592 | `24.79903358948414` | `24.796759942913` | quick miss; deep pass had 1/12 feasible ILS results and still no valid win |
| 928 | `30.891965183951985` | `30.889321856603` | quick and deep passes had 0/6 and 0/12 feasible ILS results; closest residual `2.66e-4` |

Best polished candidates, when valid, were worse than the incumbent. In the
quick braid run the closest certified values were approximately `24.799037288`
for `592` and `30.891966798` for `928`. The bounded cross-cut/strip-deletion
comparison was worse: `592` missed by about `+2.165e-3`, and `928` by about
`+3.415e-5`.

## Honest negative findings

1. A shortest active-contact path is too narrow to be a coherent reset row in
   these incumbents. Its frozen exterior pulls the soft braid back into the same
   basin, while widening or softening the band produces overlaps.
2. Several candidate graphs are disconnected between wall pairs. Assuming a
   connected braid overstates the topology; there are separate small islands and
   wall components rather than one removable row.
3. Donor strip deletion, the documented stronger comparison, did not improve the
   pilot. Its coarse cuts had high initial penetration and its best polish stayed
   worse than the incumbent.
4. The nearest braid candidates were only polished incumbents, not new
   structures. There was no sign of a target-side breakthrough, so per the 15
   minute gate this POC stops rather than scaling up.

No Kaggle/public notebook/dataset was used, nothing was submitted, and no shared
file or other POC folder was modified.
