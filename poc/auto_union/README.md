# Certified POC finals and union

Every added or changed POC folder in a PR must have final.csv in the PR head. Use the exact header id,x,y,deg. A header-only final explicitly records no certified strict wins and is valid. Nonempty finals contain only complete n groups strictly improving L against fresh main in BOTH unchanged official literal-decimal and actual pandas read_csv/to_csv transport checks. No snapshots, ties, losses or partial groups. A final need not cover all 1000 n values.

Run `python poc/auto_union/poc/union.py --export` to explicitly regenerate finals by recursively scanning packing CSVs within each POC. Both source modes must strictly improve the current incumbent, and the actual four-column exported bytes must pass the same dual-mode strict gate. Groups lost during degree conversion or transport are excluded. Zero surviving wins produces only the canonical header, including when replacing a previous final. This explicit export operation can rewrite many finals; ordinary union does not regenerate them. Historical finals are not bulk rewritten as part of this policy update.

POC finals use id,x,y,deg with s-prefixed decimals to preserve precision through pandas. The canonical incumbent is `poc/pell_corrugation/data/border-scan/merged.csv`, with id,n,square_id,x,y,angle in radians. Conversion uses 95-digit arithmetic and 75 significant output digits. Sources and hashes are recorded in docs/exports.json; metadata-only CSVs are skipped and recorded.

Run without --export to union the incumbent with every poc/*/final.csv. Header-only finals are skipped as empty candidate sets. Invalid nonempty finals abort rather than silently disappear. Only strictly smaller L in both literal and pandas modes is selected; historical ties cannot replace the incumbent. The full 1000-puzzle official literal and actual pandas checks and per-n nonregression gates must pass before atomic replacement. No improvement leaves incumbent bytes unchanged. A concurrent incumbent edit aborts the write. Run without concurrent checkout writers; refresh and revalidate after rebasing.

The PR workflow is a structure-only gate over EVERY touched POC folder, using trusted base code and a read-only token. It does not certify geometry or wins. Separate official literal and actual pandas strict-win certification against fresh main is required before merge, with provenance and environment recorded. PR code is never executed by the structure check.

Tests: `python -m unittest discover -s poc/auto_union/poc -p 'test_*.py'` and `python -m unittest discover -s .github/scripts -p 'test_*.py'`.

Dependencies are the existing baseline dependencies (pandas, mpmath, numpy, shapely). Use the same pandas version as your submission environment. Full checks are intentionally expensive; --workers controls group certification parallelism. Audit cache entries are keyed by exact literal and transported row content. No cache is used for the final full-union gate.

No Kaggle submission functionality is included.
