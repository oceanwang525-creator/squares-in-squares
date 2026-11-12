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

## Loop 2 - pending

The public import plus team union raises the bar to 22.178843629. The next loops
should use the new packings as seeds and try construction/search ideas that are
not already logged as failed under `poc/`. Every comparison must be against the
new tracked `merged.csv` (or a later team merge after re-pulling).
