# Optimization loop log

Incumbent for every comparison is the current tracked
`poc/pell_corrugation/data/border-scan/merged.csv`, certified with the
unchanged official metric in `poc/baseline`.

## Starting point (2026-11-11)

- Official score of the tracked `merged.csv`: **22.199718022945788**
- Literal certification: 1000/1000 accepted, 0 missing, 0 rejected.
- Pre-existing `merged-literal-official.json` and
  `merged-pandas-official.json` agree exactly.
- Team target: below **22.1894**. The team's better file is not merged yet;
  every loop must re-pull and re-score before trusting an incumbent.

## Loop 1 - fresh public-output sweep, including nested outputs

**Idea.** The repository's November 10 public sweep only downloaded top-level
CSV outputs of 61 kernels and ignored nested output folders. Kaggle now lists
several newer public kernels whose public LB is already below 22.1894
(for example `jonathan1506/50-top-2-6-22-1793-fault-dp-fq-stretch` at
22.1792686666). Fetch their submission CSVs anonymously, prefilter every n by
floating bounding box, then certify only strict candidates with the official
metric and merge the best certified packing per n.

**Why it might work.** A public LB below 22.1894 proves at least one valid
1000-row submission exists there. Even if the whole file is not better at every
n, its per-n rows can be merged independently. This is a data-acquisition idea,
not a new geometric construction; it is the fastest way to establish whether the
remaining gap is already present in public outputs.

**Measured result (win).** Nine public submissions were fetched to
`/tmp/opencode/kout`. Floating prefilter against the 22.199718 incumbent found
185 strict candidates in the best file and 188 in the union. All 188 certified
with the official metric and replaced the incumbent rows, giving 22.178867829.

While that import was being certified, the team merge `e6f7b69` landed at
22.19906754760468. The final tracked file is the per-n minimum of the two:
188 n from the team file, 190 from the public import, 622 ties.

- Literal score of the union: **22.178843629051656**, 1000/1000 accepted.
- Pandas transport score of the union: **22.178843629051656**, 1000/1000 accepted.
- Reduction vs the team merge: **0.020223918553029**.
- Reduction vs the earlier tracked incumbent: **0.020874393894132**.
- SHA256 of the union: `4516c499130da19ac9b0c1ec01de2b65f59bfd10c47d82029b6c2882295ab2c9`.
- POC: `poc/fresh_public_sources_nov12`; full report in
  `poc/fresh_public_sources_nov12/docs/results.md`.
- The tracked `merged.csv` was overwritten only after both full checks passed.

## Loop 2 - post-resweep public refresh and secondary outputs

**Context.** The team resweep `e6f7b69` committed at 2026-11-12 05:05 UTC and
scanned 149 kernels. Several kernels were created or updated after that
snapshot, and the Loop 1 fetch only scanned `submission.csv` of nine kernels.
Idea: scan the secondary CSVs already downloaded (anchor files,
`submission_merged.csv`, `submission_ours.csv`) and refresh every kernel whose
`lastRunTime` is newer than the resweep or that is missing from its 149-kernel
log, then merge any certified per-n win against the 22.178843629 union.

### Attempt 2a - secondary CSVs from the Loop 1 downloads

Scanned all 34 full submission-shaped CSVs under `/tmp/opencode/kout`
(including 24 `lucifer19` anchors, `submission_merged.csv`, and
`submission_ours.csv`; `embedded_baseline.csv` has a different header).
Floating prefilter against the union found **0** candidate n. Negative; the
final submissions already dominate their own anchors and variants at every n.

### Attempt 2b - refresh kernels newer than the resweep

**Idea.** Fetch current outputs of the seven kernels missing from the
resweep log (`alexchilton/squares-101-...`, `busyaprime/...`,
`jonathan1506/...`, `keshigami/...`, `mzishran/...`, `penguin069/...`,
`denniscanter/...`) plus kernels updated after the resweep snapshot
(`kmmmvs/...` at 05:45 UTC, `prvsiyan/...` at 05:04, `malyshevdanil/...` at
04:52, `alexchilton/fault-line-...` at 04:55,
`saamhm/lb-22-257046-...` at 05:40), then certify strict per-n candidates.

**Why it might work.** Public kernels are still converging; the resweep commit
was only minutes before the latest `prvsiyan` update and the Loop 1 snapshot
missed several kernels entirely. Per-n merging is independent of overall LB.

**Measured result.** Two strict per-`n` wins were found, both in the refreshed
`malyshevdanil/squares-in-squares-living-atlas-of-tight-packings` submission:
`n=487` (`delta L = 2.7859200741e-9`) and `n=733`
(`delta L = 2.7509094e-9`). All other refreshed kernels and all 34 secondary
CSVs from Loop 1 gave zero candidates.

- Literal score: **22.178843628823806**, 1000/1000 accepted.
- Pandas transport score: **22.178843628823806**, 1000/1000 accepted.
- Certified reduction: **2.2784924593596167e-10**.
- SHA256: `aa4fa2ae4fb4ee27f5230d3349eee839536b2efb0351c41d0f54e1c13cb75353`.
- POC: `poc/post_resweep_refresh`; report in
  `poc/post_resweep_refresh/docs/results.md`.
