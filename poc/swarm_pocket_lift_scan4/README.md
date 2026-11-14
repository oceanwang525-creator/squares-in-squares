# swarm_pocket_lift_scan4

Status: **positive**. Eight 0.03125-step augmented-donor scans completed and
the exact seam DP produced 24 complete strict wins. See
[`docs/results.md`](docs/results.md) for the full certified table and runtime.

## Hypothesis

Proper sub-pockets remain undersampled between the 0.0625 phase lines used by
`poc/pocket_lift_scan3`. A 0.03125-step scan should recover narrower
`(w,h,g)` pockets with lower `f`, which the exact staircase-seam DP can lift to
new `(K,D)` cells.

## Incumbent

The incumbent is the newest file published on `upstream/merge`:

```
merged-22.15248431101.csv
commit 2714efd...
sha256 5c9bad10cc9a74147d4423ab2b134915156da1f7814474ca224d8c69ffd32db4
groups 1000, rows 500500
```

The file is read directly as the donor source and official-side baseline. No
`merged.csv` is edited.

For the final donor pass, PR #72's exact H128/H131/H132 geometries are overlaid
for `n = 442,485,530,577,588,634,695,861,926`. This donor augmentation is
read-only with respect to the PR worktree, and the official baseline report
remains the unmodified `upstream/merge` artifact.

## Fine phases

The prior step-`0.0625` scans used different phase classes. At step `0.03125`,
the complementary phase classes are `0` and `1/32`. These four 2D lattices are
run as independent processes:

```
(0, 0)
(0, 1/32)
(1/32, 0)
(1/32, 1/32)
```

Four additional diagonal/cross lattices at `1/128`, `1/64`, and `7/128` are
also run when cores are available. Raw scan JSON and the incumbent snapshot are
kept out of Git.

## Pipeline

For each phase:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export VECLIB_NUM_THREADS=1 KMP_NUM_THREADS=1
python poc/swarm_pocket_lift_scan4/poc/scan_pockets.py \
  poc/swarm_pocket_lift_scan4/data/upstream-merged.csv \
  poc/swarm_pocket_lift_scan4/data/pockets_f003125_oOFFX_OFFY.json \
  --step 0.03125 --fmax 1.15 --cap 3000 --nmax 31 \
  --procs 1 --offx OFFX --offy OFFY --verbose
```

The actual runs use `poc/run_phases.sh`, which starts all eight phases
concurrently and gives each process the single-thread environment above:

```sh
PYTHON=$PY poc/swarm_pocket_lift_scan4/poc/run_phases.sh \
  poc/swarm_pocket_lift_scan4/data/donors_augmented.csv \
  poc/swarm_pocket_lift_scan4/data/pockets_f003125_aug \
  442,485,530,577,588,634,695,861,926
```

The copied scan3 pipeline then:

1. merges the incumbent gadget library, the previously sanitized scan3 pocket
   library, and every new scan in one Pareto pass;
2. re-measures every pocket record from the newest incumbent donor bytes;
3. runs the exact staircase-seam DP;
4. rebuilds and certifies only predicted cells;
5. exports only complete groups with `dL < -1e-9` in both literal decimal and
   real pandas `read_csv`/`to_csv` transports.

## Reproduce

```sh
PY=/tmp/h94-venv/bin/python
D=poc/swarm_pocket_lift_scan4

$PY poc/faultline_dp2/poc/mkreport.py \
  $D/data/upstream-merged.csv $D/data/report.json
$PY poc/faultline_dp2/poc/gadlib.py \
  $D/data/upstream-merged.csv $D/data/gadlib.json

$PY $D/poc/merge_pockets.py --gadlib $D/data/gadlib.json \
  --out $D/data/pockets_pareto_scan4.json \
  --seeds $D/data/seeds_scan4.json \
  $D/data/pockets_prev_sane.json $D/data/pockets_f003125_aug_o*.json
$PY $D/poc/sanitize.py $D/data/donors_augmented.csv \
  $D/data/pockets_pareto_scan4.json $D/data/pockets_pareto_scan4_sane.json
$PY $D/poc/dp.py extra --lib $D/data/gadlib.json \
  --pockets $D/data/pockets_pareto_scan4_sane.json \
  --report $D/data/report.json --out $D/data/pred_scan4.json
$PY $D/poc/build_cert.py --pred $D/data/pred_scan4.json \
  --pockets $D/data/pockets_pareto_scan4_sane.json \
  --csv $D/data/donors_augmented.csv \
  --report $D/data/report.json --outdir $D/data/built_scan4
$PY $D/poc/mkfinal.py --built $D/data/built_scan4
$PY $D/poc/verify_final.py
```

## Validation

```sh
$PY -m pytest -q poc/swarm_pocket_lift_scan4/tests
BASE_SHA=$(git rev-parse upstream/main) HEAD_SHA=$(git rev-parse HEAD) \
  $PY .github/scripts/check_final_csv.py
git diff --check
```

No Kaggle submission and no public notebook or dataset harvesting is used.
