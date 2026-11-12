# Public kernel resweep (Nov 10 2026)

Fixes the gaps of `../fresh_public_sources`: lists all 149 competition kernels (ListKernels with `page`=1..3, 8 sort orders), fetches the output of every one (126 have outputs; 0 listing/download failures), walks nested output folders and 9 zips, scans 1498 CSVs (53,342 unique per-n packings; 223 CSVs skipped, not submission-shaped, see `docs/scan_stats.json`).

Result: 376 n strictly better than the current `border-scan/merged.csv` under the exact evaluator (`out/wins_rows.csv`, `docs/wins.json`; our L per n in `docs/our_L.json`, fresh evaluation of merged.csv). Only the latest version of each kernel is reachable anonymously. Many gains are at the 1e-16 level (evaluator rounding); only 11 exceed 1e-9 (65 exceed 1e-12); 120 of the 376 fail the pandas read_csv/to_csv round trip alone (`pandas_roundtrip_ok` in wins.json), so the lead must run the whole-file literal and pandas checks.

Prefilter: bbox side < our L + 1e-9 (20,887 candidates). Exact literal evaluation was run for those with bbox < our L + 1e-13 (5,580 non-identical; larger bbox cannot be strictly better as-is). 162 were invalid; LP repair (fixed-angle SAT LP, `poc/verify.py`, dilation ladder) was attempted on 4 with none strictly beating ours, so no win is repaired.

Order: `list_kernels.py`, `fetch_all.py`, `extract_zips.py`, `our_L.py`, `scan.py`, `verify.py 3 {0,1,2}`, `finalize.py`. `data/` (untrusted downloads, ~5.5 GB) is gitignored.
