# Published catalog coordinate import

Imports packing coordinates as data from [jlevy/squares](https://github.com/jlevy/squares) and [evand/square-packing](https://github.com/evand/square-packing), retaining original witness provenance. Does not execute either project's code.

The November 10, 2026 extraction found coordinates for 350 distinct puzzle sizes. Compared with the newer contact-polish portfolio, only n=172,199,270,303 strictly improve. All four winning witnesses are from jlevy's known-best YAML catalog. These are imported solutions, not new constructions.

Run `python poc/catalog_import/poc/import_catalog.py` with mpmath and PyYAML installed and the baseline package importable. Set JLEVY_CATALOG and EVAND_CATALOG to source clones, and CATALOG_OUTPUT to a fresh writable directory. The extractor compares against the checked-in border-scan incumbent. Its candidates still require independent geometry checks and comparison with the latest portfolio before use. Extraction alone never modifies the incumbent or submits.

Scalar rationals and algebraic witness fields are evaluated at 80 decimal digits. Centers and basis vectors are converted to center/angle representation. Projected exact contacts can need a tiny dilation before decimal transport. n=199 used a 1e-13 relative center dilation; other winners needed none. Decimal precision is not a claim of equally precise numerical optimization.

See docs/results.md and data/wins.json for provenance and verification. The full merged portfolio preserves all 996 other rows from the newer incumbent. No Kaggle submission.
