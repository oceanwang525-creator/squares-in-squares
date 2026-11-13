# Certified POC finals and union

Every added or changed POC folder must have final.csv in the PR head with exact header id,x,y,deg. Header-only files are valid. Nonempty files may include winning, tied, incumbent or worse complete groups; every included group must pass unchanged official geometry certification in both literal-decimal and actual pandas transport modes. Partial, duplicate, nonfinite and invalid groups are forbidden.

Run `python poc/auto_union/poc/union.py --export` to regenerate finals by scanning packing CSVs within each POC. Export retains the best literal candidate per n among dual-valid source groups, even if nonwinning against main. The actual four-column exported bytes must also pass dual geometry certification. Conversion-invalid groups are excluded. No valid candidates produces only the header. This explicit operation may rewrite many finals; ordinary union never regenerates them.

Finals use id,x,y,deg with s-prefixed decimals. Canonical remains poc/pell_corrugation/data/border-scan/merged.csv with id,n,square_id,x,y,angle in radians. Conversion uses 95-digit arithmetic and 75 significant output digits. Sources/hashes are recorded in docs/exports.json.

Run without --export to union the incumbent with every poc/*/final.csv. Every nonempty group is dual-certified, including nonwinning groups; any invalid final aborts. The incumbent always remains a candidate. Only strictly smaller L in both modes replaces the current best per n; tied and worse candidates are ignored, not rejected. Full 1000-group literal/pandas certification with zero per-n regressions is required before atomic replacement. No improvement preserves original bytes. Concurrent incumbent changes abort. Publication workflow owns canonical writes.

The trusted-base PR workflow checks structure only and never executes PR code. Before merge, separately certify ALL included groups geometrically in both official modes and the full proposed union 1000/1000 with zero regressions against fresh main. Record environment and provenance. Refresh checks after main changes.

Tests use `python -m unittest discover -s poc/auto_union/poc -p 'test_*.py'` and `python -m unittest discover -s .github/scripts -p 'test_*.py'`. Existing dependencies include pandas, mpmath, numpy and shapely. Match the submission environment's pandas version. No final full-union cache is used. No Kaggle submission functionality is included.
