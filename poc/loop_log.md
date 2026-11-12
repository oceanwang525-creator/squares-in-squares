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

## Loop 3 - refresh external GitHub witness banks

**Idea.** The team resweep `public_resweep_other` mined four external GitHub
packing banks as of 2026-11-10. `jlevy/squares` gained commits on 2026-11-11
and 2026-11-12 with new `de-winter-*` and `ozden-*` witness/certificate
directories. Re-clone the banks at current HEAD, harvest the new rational
corner/center-basis witnesses, `.cert` files, `.npz` banks, and generic
JSON/CSV layouts, then certify strict per-`n` candidates against the tracked
incumbent. Repair any float64-transport overlap by minimal uniform dilation.

**Why it might work.** These are exact witnesses from independent packing
researchers, not re-polished versions of our own rows. The resweep commit
predates the newest de Winter/Ozden witnesses by one to two days.

**Measured result (win).** Seven strict wins, all from new `jlevy/squares`
witnesses: `n=132` (gain `8.383e-4`), 156 (`1.692e-4`), 157 (`5.345e-4`),
180 (`1.025e-4`), 237 (`5.715e-4`), 268 (`1.238e-10`), 269 (`3.164e-5`).
`n=132` and `n=157` needed a `1e-14` dilation to survive pandas float64
transport; the first unrepaired merge was rejected 998/1000.

- Literal score: **22.17659599299808**, 1000/1000 accepted.
- Pandas transport score: **22.17659599299808**, 1000/1000 accepted.
- Certified reduction: **0.002247635825722413**.
- SHA256: `f18356d4f428ea90791ae6a1f26b0ef4c2e864a6ea3c9d04008eec664dd7ee03`.
- POC: `poc/external_witness_refresh`; report in
  `poc/external_witness_refresh/docs/results.md`.

## Loop 4 - fresh kernel/dataset tail after Loop 3

**Idea.** Loop 3 only refreshed the four GitHub banks. The kernel list now has
one brand-new kernel (`lianlian99/notebook256e4579c0`, public 22.1789653025,
last run 06:09 UTC) and several datasets were updated around or after the
resweep snapshot: `prvsiyan/squares-verified-best-of-1105` (05:04 UTC),
`jonathan1506/joegun-squares-packings` (03:51), and
`cshara/squares-in-squares-current-best` (Nov 11 22:05). Fetch their current
CSVs, prefilter by floating bbox, and certify strict per-n candidates against
the 22.17659599299808 incumbent. Also re-check the public gallery index; its
displayed L values are only six-decimal rounded, so only per-n JSON
coordinates can produce a real win.

**Why it might work.** These are independent public portfolios updated after
the last sweep, and per-n merging does not care about their overall LB.

**Measured result (negative).** Fetched the new `lianlian99/notebook256e4579c0`
submission (31.3 MB), the now-populated `kmmmvs/squares-public-merge-daily`
submission, and current `prvsiyan/squares-verified-best-of-1105`,
`jonathan1506/joegun-squares-packings`, and
`cshara/squares-in-squares-current-best` datasets. All 20 submission-shaped
CSVs prefiltered to **0** candidate n. The JoeGun `layouts.json` 45 entries are
exactly the packings already in our union (their L equals ours at every n) and
their published `sq` arrays have small float overlaps, so no repair can beat
the tracked rows. Gallery index has 165 apparent 1e-7-scale "wins" but those are
only six-decimal display rounding; the gallery's top-gap n are all worse than
ours. No change to `merged.csv`.

### Loop 4b - full-portfolio fixed-angle LP with no trust region

**Idea.** Every prior "repair" re-solved candidates inside a small trust region.
For each group, fix all angles and solve the box-side LP over all centres with
**no bounds**, rebuilding the nearby-pair set and active axis from the LP
solution (3 SLP rounds). Any LP side below the incumbent is then gated by the
unchanged exact certifier. This is a rigorous test for translational slack.

**Measured result (negative).** 28-group probe (the ten largest leader gaps
plus grids and mid-range n): 27 `no-gain` with LP side = base + ~2.7e-8 (the
1e-9 constraint margins), one `certify-fail` whose LP gain 8e-9 is inside the
margin. No group has exploitable fixed-angle translational slack. POC:
`poc/alln_fixed_lp`.

## Loop 5 - unscanned external GitHub packing banks

**Idea.** The resweep's four repos were `Minuii`, `jlevy`, `evand`,
`CacinieP`. GitHub search now surfaces many more independent banks that the
repository has never scanned: the `kenaznia-doga/s*-upper-bound` family,
`JoostdeWinter/square-packing`, `TheSnakeFang/squarepack-certs`,
`DigitalEuan/squares_in_squares`, `cubres/kaggle-squares-in-squares`,
`raojohng/squares-in-squares`, `anopheles/packingproblem`,
`TobiasHFS/squares-in-squares`, `omerfisler/kaggle-unit-squares-packing`,
`rushikesh648/Squares-in-square-`,
`mohsensbu-jpg/square_packing_records_karimi`,
`chromoany/r68-square-packing`, plus `jlevy/squares` at its newest HEAD
(`45e2ecca`, 07:08 UTC, newer than the Loop 3 clone). Clone them, harvest
witness/cert/yaml/npz/CSV/JSON packings, and certify strict per-n candidates
against the 22.17659599299808 incumbent.

**Why it might work.** `kenaznia-doga`'s published bounds are already matched,
but its repos ship the raw witness geometry, and `jlevy` gained a commit after
the Loop 3 refresh with a new `ozden-s269-issue962` witness.

**Measured result.** Pending.

### Loop 5a - bank scan at float margin 1e-9

All twelve banks plus `jlevy/squares@45e2ecca` cloned and scanned (34,899
files). One candidate above the 1e-9 margin: `n=269`, float side
`16.899469075855` against our `16.900994579951`. `n=2` from
`overlap-negative-control.yaml` is a deliberate invalid control and fails
certification as expected.

### Loop 5b - tighter 1e-13 margin

Re-scan at `tol=1e-13` found 17 candidate `n`. Exact certification plus pandas
transport:
- **n=269: WIN.** `16.90099457995132482 -> 16.89946907585504653` (ΔL
  `0.0015255`, gain `9.301e-5`). No dilation needed; both literal and pandas
  certify.
- n=132, 157: raw de Winter witnesses certify `1.1e-13`/`1.2e-13` better than
  the tracked dilated rows but fail pandas transport. The tracked 1e-14-dilated
  rows stay.
- n=442, 530, 531, 577, 578, 627, 678, 786, 843, 902, 963: `couzo-index`
  certificates are literal wins of `1.1e-13`..`2.7e-13` but all fail pandas
  transport; unusable dust.
- n=172, 199: witnesses fail exact certification (regularized drawing form).

**Measured result (win).** Only n=269 survives both gates.

- Literal score: **22.17650298137176**, 1000/1000 accepted.
- Pandas transport score: **22.176502981371762**, 1000/1000 accepted.
- Certified reduction: **0.00009301162632211604**.
- SHA256: `7de2a8432ca6380678127734fb611cc5c4f090e151d9e804475b2c7ffb4eef05`.
- POC: `poc/external_bank_expansion`; report in
  `poc/external_bank_expansion/docs/results.md`.

## Loop 6 - full-portfolio fixed-angle LP with no trust region

**Idea.** Novel form of the polish idea: for each of the 1000 groups, fix all
angles and solve the box-side LP over all centres with **no bounds**, rebuilding
the nearby-pair set and active axis from the LP solution (3 SLP rounds). Any LP
side strictly below the incumbent is gated by the unchanged exact certifier.
Prior repair work only ever solved inside a small trust region, so this is the
first rigorous whole-portfolio test for translational slack.

**Why it might work.** Rows imported from other teams' submissions were
certified as-is but never re-optimised against our exact constraint set; if a
row's contact pattern leaves a direction free, the unbounded LP can compress it.

**Measured result (win).** Two strict wins out of 1000 groups:

- n=595: `24.86192565354330775 -> 24.86192535001240769` (ΔL `3.0353e-7`,
  gain `1.244284e-8`)
- n=875: `29.98873321773301107 -> 29.98873319247497383` (ΔL `2.5258e-8`,
  gain `8.5459e-10`)

Both certify identically for literal and pandas transport; no dilation needed.
The other 995 groups return an LP side a constant `~2.7e-8` above the incumbent
(the accumulated 1e-9 safety margins), i.e. they are already at the fixed-angle
LP optimum; 3 groups give sub-margin LP solutions that the exact gate rejects.

- Literal score: **22.176502968074328**, 1000/1000 accepted.
- Pandas transport score: **22.176502968074328**, 1000/1000 accepted.
- Certified reduction: **1.3297431220727618e-8**.
- SHA256: `693aab989cc52cab0310337ab6bbf53319f7b8a759e27c1fa1081e8fb5cb3baf`.
- POC: `poc/alln_fixed_lp`; report in `poc/alln_fixed_lp/docs/results.md`.

## Loop 7 - per-n unions with the team public-refresh artifacts

**Idea.** Two team artifacts landed while Loop 5/6 were being pushed:
`poc/public_refresh_nov12` (`bdf9805`, `dbd248f`) and
`poc/certified_union_nov12` (`f8a97b7`, which also replaced the tracked
`merged.csv`). A team file can be worse overall and still hold the best
individual puzzle, so compare official reports per `n`, copy the winning rows,
and re-certify the union.

**Measured result (win, two stages).**

1. Union Loop 6 with `public_refresh_nov12`: the team refresh is strictly
   better at **33** `n`, including the two largest remaining leader gaps:
   - n=971: `31.674933279210167 -> 31.664318325447744` (ΔL `0.010615`, gain
     `3.4065e-4`)
   - n=687: `26.727922061358046 -> 26.725053394441424` (ΔL `0.002869`, gain
     `1.0945e-4`)
   Result **22.175937603064224**, 1000/1000 both transports.
2. The team's `certified_union_nov12` then landed at **22.175937592741963**,
   better overall. Adopted as the new base and diffed per `n`: the stage-1
   intermediate is still better at **11** `n` (n=595, 875, 487, 733, 444, 966,
   851, 677, 730, 735, 591; total `2.015e-9`). Final union:
   **22.175937590726818**, 1000/1000 both transports.

- Literal score: **22.175937590726818**, 1000/1000 accepted.
- Pandas transport score: **22.175937590726818**, 1000/1000 accepted.
- Certified reduction vs the team file: **2.0151448357305352e-9**.
- SHA256: `53b76adab4dcbd6a0e76369eb37c7c83444ea0a429a5559d5bbc53571bbd4bd9`.
- POC: `poc/team_union_nov12`; report in
  `poc/team_union_nov12/docs/results.md`.

**Lesson.** Diff official reports per `n` after every team merge, not just the
totals, and re-diff again if the team lands another file.

## Loop 8 - midday public kernel refresh

**Idea.** After Loop 5 the kernel list grew to 154 and four kernels produced new
or changed CSV outputs after the scan: `nilanjansaha123/squares-gpu-seam-dp-soft-rigid-highs`
and `samanyu1808/sis-submit` (both 07:57 UTC), `grennwolf/...santa-2025`
(07:29) and the new `seshurajup/1000-puzzles-ensemble-lb-22-2095` (06:56).
Fetch their current outputs and certify strict per-`n` candidates.

**Measured result (win).** One strict win, `n=694` from
`grennwolf/submission_improved.csv`:

- `26.827240193605122 -> 26.826047646280325` (ΔL `0.0011925`, gain
  `4.5268477398e-5`), literal and pandas identical, no dilation.

The other three kernels gave 0 candidates, and the grennwolf file re-scanned at
a `1e-13` margin still yields only `n=694`.

- Literal score: **22.17589232224942**, 1000/1000 accepted.
- Pandas transport score: **22.17589232224942**, 1000/1000 accepted.
- Certified reduction: **4.5268477398062724e-5**.
- SHA256: `eec27294fbd9991e5a7d65dc447070f89715869e27399fadd3c7d4ddefc2ef61`.
- POC: `poc/kernel_refresh_noon`; report in
  `poc/kernel_refresh_noon/docs/results.md`.

## Loop 9 - live tracker and per-bound witness repos

**Idea.** The `s269-upper-bound-1688973007` repository references a live tracker
(`https://dogaozden.com/math/live/square-packing/`) whose JSON feed publishes
current per-`n` sides, and the same account publishes one exact-rational
witness repo per improvement (`kenaznia-doga/s<n>-upper-bound*`). Poll the feed,
flag every `n` below the incumbent, then clone and certify the matching witness
repos.

**Why it might work.** This is the same lab that produced the n=269 row we
imported earlier; the newest repo (`1688973007`, pushed 03:00 UTC) claims
`16.889730073596`, below both our row and the Kaggle per-puzzle leader entry
(`16.889730079125`).

**Measured result (win).** The feed scan gives exactly one candidate, `n=269`:

- `16.899469075855046 -> 16.889730073595637` (ΔL `0.009739`, gain
  `5.9379744775e-4`), literal and pandas identical, no dilation.

Negative checks: the other 291 feed drawings are 5-decimal poses whose apparent
`~4e-6` gaps are rounding; `s179-upper-bound-13880960371` publishes
`13.880960371558225`, `3e-12` worse than our `13.880960371555588`; `s237` is
the same row we already track.

- Literal score: **22.17529852480167**, 1000/1000 accepted.
- Pandas transport score: **22.175298524801672**, 1000/1000 accepted.
- Certified reduction: **5.937974477501639e-4**.
- SHA256: `9195b04515268df0cd53143dd1442fdd63e7f32c0d3a170100665525f353926b`.
- POC: `poc/live_tracker_watch`; report in
  `poc/live_tracker_watch/docs/results.md`.

## Loop 10 - morning public kernel refresh

**Idea.** The public kernels now iterate on top of the merged portfolio: four
re-ran between 07:29 and 09:09 UTC (`prvsiyan` 08:43, `jazivxt` 08:51 with a
new public LB `22.1779795658`, `jonathan1506` 09:02, `grennwolf` 07:29).
Fetch their current outputs and merge every strict per-`n` winner.

**Measured result (win).** **114** strict per-`n` wins, each certified below
the incumbent in both literal and pandas transport with no dilation:

- largest: n=642 `25.826248643 -> 25.817553669` (gain `3.4316e-4`), n=504
  `22.915135075 -> 22.911415009`, n=862 and n=922 (gain `1.2e-4` each), n=507
  `7.0e-5`.
- sources: `prvsiyan` 104 n, `jonathan1506` 22 n, `jazivxt` 20 n (overlapping).

- Literal score: **22.173995906838663**, 1000/1000 accepted.
- Pandas transport score: **22.173995906838663**, 1000/1000 accepted.
- Certified reduction: **0.001302617963011434**.
- SHA256: `e5ef17f711ea2e674eb03b006dfa6b3c018fd3ac5e053cf7166896b0715cda92`.
- POC: `poc/kernel_refresh_morning`; report in
  `poc/kernel_refresh_morning/docs/results.md`.

**Lesson.** Once public kernels consume our portfolio, each of their re-runs is
a free polish pass over hundreds of `n`. Re-list kernels at least every 30
minutes and merge per `n`.

## Loop 11 - whole-portfolio free-angle SLP

**Idea.** The fixed-angle LP only moves centres. Add each square's rotation as
an LP variable, linearising extents and pair radii to first order in the angle,
with a shrinking trust region, rebuilt contact set and six SLP rounds. This is
the strongest local test left: every square can translate and rotate at once,
unlike the earlier one-square coordinate descent.

**Why it might work.** The portfolio was assembled from many sources with
different polish pipelines; a joint angle step could relieve a contact pattern
that no single-square move can.

**Measured result (negative).** 1000/1000 groups return an SLP side equal to
the incumbent (status `no-gain`); no group is strictly below, so nothing was
emitted. Top-gap probe (971, 911, 910, 972, 793, 634, 791, 908, 969, 803) was
also 0/10 in 9 seconds; the full run took about four minutes.

- `merged.csv` unchanged at SHA256
  `e5ef17f711ea2e674eb03b006dfa6b3c018fd3ac5e053cf7166896b0715cda92`.
- POC: `poc/free_angle_slp`; report in `poc/free_angle_slp/docs/results.md`,
  per-`n` status in `docs/scan.json`.

**Conclusion.** The tracked packings are local optima in the full `(x, y, θ)`
space. With Loop 6 (no translational slack) and `leader_poc_k2p4` (n=328 strict
free-angle local optimum), the remaining leader gap is structural: it needs
different contact topologies or constructions, not more polishing.

## Loop 12 - harvest watcher and systematic source discovery

**Idea.** Automate one harvest cycle (newest kernel outputs, live feed, team
push) and pair it with systematic GitHub/Kaggle search for repositories that
publish exact certificates per bound. The n=269 win came from exactly such a
per-bound repo, so the search is worth repeating as new ones appear.

**Measured result (negative).** The first cycle after Loop 10:
kernels newer than `09:33:56` = 1 (`saamhm`, 0 candidates); live feed
(`at 01:38-08:00`) = 0 candidates above `1e-12`; `origin/main` unchanged.
Discovery scan of twelve new sources:

- `BuddenPing/square-packing-certificates` (n=105,146,153,207,261,272,302),
  `lollipoll/couzo-five-exact-certificates`,
  `SidG2k1/square-packing-refinements` (14 exact refinements at ~`1e-13`),
  `ebdeleeuw/square-packing-n70`, nine Kaggle-competition repos, and the
  `samanyu1808/sis-final-packings` dataset: **0 candidates**.
- The only scanner hits were `EvanOman/.../spheres.json` sphere coordinates,
  rejected as not square packings.

The refinement repos improve *published* sides by ~`1e-13`, but the tracked
portfolio already matches or beats every reference they refine.

- `merged.csv` unchanged at SHA256
  `e5ef17f711ea2e674eb03b006dfa6b3c018fd3ac5e053cf7166896b0715cda92`.
- POC: `poc/harvest_watch`; report in `poc/harvest_watch/docs/results.md`.

## Loop 13 - tiny pass re-run on the current portfolio

**Idea.** The team's `tiny_pass.py` (deletion from n+k and upright border wraps
followed by deletion) added only `2.49e-10` when it ran on the November 12
intermediate file. The tracked file has since changed at more than 200 `n`, so
re-run the same pass against the current incumbent.

**Measured result (win).** 8 strict wins, all literal and pandas identical:

- largest: n=904 `wrap-845-t1`, `30.563136023802043 -> 30.563136023765352`
  (dL `3.669e-11`, the only win above `1e-13`);
- deletions at the flat pairs n=442, 577, 627, 678, 492 (dL `1e-17`..`6.5e-14`);
- wraps at n=732 (`wrap-679-t1`) and n=964 (`wrap-679-t5`).

Wrap candidates at 442, 530, 577, 678, 786, 843, 902, 963 all returned
`no gain` (wrapped side `1e-13` above the incumbent).

- Literal score: **22.173995906837433**, 1000/1000 accepted.
- Pandas transport score: **22.173995906837433**, 1000/1000 accepted.
- Certified reduction: **1.2292409126146337e-12**.
- SHA256: `308fc61effc296bebefa57617d095b5b81055e425541fbb25d9b55296c25d302`.
- POC: `poc/tiny_pass_rerun`; report in `poc/tiny_pass_rerun/docs/results.md`.

**Note.** Dust by construction, but the pass is nearly free and every strict win
is legitimate; re-run it whenever the file changes at the flat pairs.

## Loop 14 - sigmaborov contact-polish kernel

**Idea.** A new public kernel (`sigmaborov/squares-in-squares-certified-contact-polish`,
first seen 10:12 UTC) publishes a contact-polish of the frozen public baseline:
a 40 MB submission plus `local_witnesses.json` with 37 per-`n` witnesses. Its
public score (`22.2043`) is far worse than the tracked file, but per-`n` polish
can still win.

**Measured result (win).** 3 strict wins, all literal and pandas identical:

- n=694 `26.825233542153100 -> 26.825233532081742` (gain `3.822e-10`)
- n=551 `23.925171829038760 -> 23.925171823365849` (gain `2.416e-10`)
- n=504 `22.911415008793174 -> 22.911415005703776` (gain `1.376e-10`)

The other 34 local witnesses and the frozen baseline gave 0 candidates.

- Literal score: **22.17399590607584**, 1000/1000 accepted.
- Pandas transport score: **22.17399590607584**, 1000/1000 accepted.
- Certified reduction: **7.615905611345757e-10**.
- SHA256: `42421d82d582a6f05e738d13e830ef328161a39f469d00896f58ee521b29cdb5`.
- POC: `poc/sigmaborov_contact_polish`; report in
  `poc/sigmaborov_contact_polish/docs/results.md`.

## Loop 15 - fixed-angle LP re-run after later merges

**Idea.** Newly imported rows (kernels, team unions, tiny pass) have never been
through the unbounded fixed-angle box-side LP. The file changed at more than
150 `n` since Loop 6, so re-run the same whole-portfolio LP (3 SLP rounds,
exact literal + pandas gate).

**Measured result (win).** One new strict win:

- n=740 `27.70664651748958335 -> 27.70664651341567700` (ΔL `4.07e-9`, gain
  `1.4976e-10`), literal and pandas identical.

Status counts: 997 `no-gain`, 1 `certify-fail`, 1 `lp-infeasible`, 1 `WIN`.
The LP cost about two minutes for all 1000 groups.

- Literal score: **22.17399590592608**, 1000/1000 accepted.
- Pandas transport score: **22.173995905926084**, 1000/1000 accepted.
- Certified reduction: **1.4975977578826128e-10**.
- SHA256: `3950c42c66bfde7bba8b5dd9991d3b5551e1e5f7518ca10ecb0229663a2cf3b0`.
- POC: `poc/alln_fixed_lp`; reports `docs/rerun-*.json`.

**Operational note.** Re-run this LP after every large merge; it is cheap and
matches only the newly imported rows.

## Loop 16 - 32-thread parallel execution

**Idea.** Use the full Ryzen 9 9950X (16 cores / 32 threads) for search and
certification: one process per `n`, BLAS/OpenMP pinned to 1 in every worker,
largest `n` first with `imap_unordered`, deterministic sorted merge. The final
whole-file literal and pandas checks stay single-process.

**Measured throughput (before -> after).**

| stage | serial | 32 jobs | speedup |
|---|---:|---:|---:|
| fixed-angle LP, 1000 groups | 476 groups/min (125.9 s) | **7112 groups/min (8.4 s)** | 15.0x |
| free-angle SLP, 1000 groups | ~240 s | **12.7 s** | ~19x |
| merge/certify gate, 114 candidate `n` | ~5 min | **8.0 s** | ~14x wall, 2m44s user |
| external bank scan, 34900 files | tens of minutes | **~28 s** | large |

**CPU confirmation.** During a heavier 12-round LP run, `top` reported
`%Cpu(s): 96.3 us, 1.3 id`, and 35 Python processes summed to **2986%** (≈30 of
32 hardware threads), with individual workers at 99-100%.

**Correctness.** The parallel merge gate reproduced the incumbent byte-for-byte
on a 114-candidate test where every candidate was already merged (0 wins), and
the parallel LP reproduced the serial `n=740` result. Results are merged in
sorted `n` order, so output is deterministic.

**Memory.** 52 GiB installed, 42 GiB available; 32 workers stayed well under
10 GiB (each LP worker is ~300 MB peak). No swap was touched.

Tools updated: `poc/alln_fixed_lp/poc/alln_fixed_lp.py`,
`poc/free_angle_slp/poc/free_angle_slp.py`,
`poc/external_witness_refresh/poc/merge_witnesses.py`,
`poc/external_witness_refresh/poc/scan_new.py`. All take `--jobs 32`.

## Loop 17 - deeper fixed-angle LP re-run (12 rounds)

**Idea.** The Loop 16 parallel LP test used 12 SLP rounds instead of the default
three and found a win that three rounds missed. Re-run at 12 rounds on the
current file.

**Measured result (win).** One strict win:

- n=595 `24.86192535001240769 -> 24.86192534775471239` (ΔL `2.258e-9`, gain
  `9.2556e-11`), literal and pandas identical.
- Status: 999 `no-gain`, 1 `WIN`. Cost: about 15 s at 32 jobs.

- Literal score: **22.173995905833525**, 1000/1000 accepted.
- Pandas transport score: **22.173995905833525**, 1000/1000 accepted.
- SHA256: `71e6520007115285f6768890b48efaf9f521a87e0c1d7325ddc78cb875f1639b`.
- POC: `poc/alln_fixed_lp`; reports `docs/rerun2-*.json`.

**Operational note.** Use 12 rounds (about 15 s at 32 jobs) as the default for
re-runs; three rounds can stop short of the LP optimum.

## Loop 18 - contact-topology search (seeded tie-breaking)

**Idea.** The fixed-angle LP fixes one separating axis per nearby pair; when two
axes are nearly tied, the choice is arbitrary. Add a seeded worker per
`(n, seed)` that picks randomly among axes within a margin of the best, re-runs
the 12-round SLP, and keeps the best certified result per `n`. This searches
alternative contact graphs around each incumbent packing, parallelized across
32 processes.

**Measured result (negative).**

| run | tasks | time at 32 jobs | wins |
|---|---:|---:|---:|
| `--seeds 8 --tie-tol 1e-6 --rounds 12` | 8,000 | 271 s | **0** |
| `--seeds 4 --tie-tol 1e-3 --rounds 12` | 4,000 | 155 s | **0** |

No alternative contact graph produces a certified lower side. Together with
Loops 6/11/17 (fixed-angle LP at 3/12/20 rounds and free-angle SLP at 6/12
rounds), the tracked portfolio is a local optimum in centres, angles and
near-tied contact topologies.

- `merged.csv` unchanged at SHA256
  `71e6520007115285f6768890b48efaf9f521a87e0c1d7325ddc78cb875f1639b`.
- POC: `poc/alln_fixed_lp` (`--seeds`, `--tie-tol`).
