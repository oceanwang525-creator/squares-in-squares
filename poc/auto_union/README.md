# Certified POC finals and union

Run `python poc/auto_union/poc/union.py --export` to scan packing CSVs recursively within each POC. Each final contains only complete groups from that folder that pass the unchanged official metric both literally and after actual pandas read_csv/to_csv transport, with both side lengths no greater than the incumbent. Metadata-only CSVs are skipped and recorded. Sources and hashes are in docs/exports.json. Files are not size-capped or truncated.

POC finals use id,x,y,deg with s-prefixed decimals to preserve precision through pandas. The incumbent remains id,n,square_id,x,y,angle in radians. Conversion uses 95-digit arithmetic and 75 significant output digits; the actual exported bytes are recertified. Groups whose degree conversion introduces even a tiny regression are excluded and listed in transport_excluded, rather than weakening the gate. Sources are attributed by their containing POC; included baselines and imported submissions are recorded as such by path, not claimed as novel discoveries.

Run without --export to union merged.csv with every poc/*/final.csv. Invalid finals abort rather than silently disappear. Only strictly smaller L in both literal and pandas modes is selected. The full 1000-puzzle official literal and pandas checks and per-n nonregression gates must pass before an atomic replacement. No improvement leaves incumbent bytes unchanged. A concurrent incumbent edit aborts the write. Run without concurrent checkout writers; refresh and revalidate after rebasing.

Tests: python -m unittest discover -s poc/auto_union/poc -p 'test_*.py'.

No Kaggle submission functionality is included.

Dependencies are the existing baseline dependencies (pandas, mpmath, numpy, shapely). Use the same pandas version as your submission environment. Full checks are intentionally expensive; --workers controls group certification parallelism. Audit cache entries are keyed by both exact literal and transported row content. No cache is used for the final full-union gate.
