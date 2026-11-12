# Reproduction notes

## Validation

From the repository root, install the baseline package and run:

```sh
python -m pip install -e ./poc/baseline/poc
python poc/contact_slp_submission189/poc/validate_submission.py \
  poc/contact_slp_submission189/data/submission.csv \
  --out poc/contact_slp_submission189/docs
```

The validator uses the unchanged evaluator vendored in `poc/baseline`. It reads
the submitted decimal strings directly for one run. For the other, it reads the
same complete file with the default `pandas.read_csv` numeric inference and
writes it with `to_csv(index=False)` before evaluating. A report with any
missing or rejected puzzle fails validation. The side comparison uses the two
committed measurement receipts for upstream `merged.csv` at
`5a0914194ee23635a0d720e48b0e12d542271567`.

## Search and selection

The geometry and fixed-angle repair routines used during the experiment came
from the [sqk toolkit](https://github.com/chaydavs/squares-in-squares/tree/ddb32bed3f3f1d34c2ab0067684504a4677bc509).
Use that revision on `PYTHONPATH` for the refinement scripts. NumPy and SciPy
are required in addition to the baseline package.

The seed CSV for the search was an earlier portfolio, `final_atlas.csv`,
built from the earlier `final.csv` and the public sources below. The refinement
scripts accept any complete seed CSV with the competition header.
`submission189.csv` was the acceptance baseline for selecting whole layouts.

```sh
python poc/contact_slp_submission189/poc/refine_support.py seed.csv support.csv 593
python poc/contact_slp_submission189/poc/refine_corners.py seed.csv corners.csv 814 873 915 921 925 934
python poc/contact_slp_submission189/poc/select_best.py submission189.csv candidate.csv support.csv corners.csv
```

The two refinement variants linearize either the support expressions or the
selected-axis corner projections. The selector requires a per-puzzle official
score decrease greater than `1e-8` before replacing a complete group. This
avoids churning rows over differences at export precision.

Seed sources include the upstream
[pell corrugation portfolio](https://github.com/koboshchan/squares-in-squares/tree/5a0914194ee23635a0d720e48b0e12d542271567/poc/pell_corrugation/data/border-scan),
[Malyshev Danil’s public packing atlas](https://www.kaggle.com/code/malyshevdanil/squares-in-squares-living-atlas-of-tight-packings),
and [The Squares Project](https://github.com/jlevy/squares). The committed full
CSV and validation receipts are self-contained for score reproduction.
The seven adopted search groups all started from Malyshev Danil's
`submission.csv` in the public atlas. The support variant produced 593.
The corner variant produced 814, 873, 915, 921, 925, and 934.

