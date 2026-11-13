# Results

Upstream cd17a65, incumbent SHA-256 1f9c0fd3b7a88e5a0f6c44b567ff2ef354872812c6b0ad8e37807bd1a64e965c. The earlier independent official literal and pandas 3.1.0 reports for this same file both passed 1000/1000 at 22.173715450068453.

The scan tested 21,472 nonzero-angle windows for targets n=100..200. No window held enough squares below the target side. No coordinates advanced to certification.

The scan searches whole-square subsets of nearby incumbent donors and samples angles. It does not relax square positions after cropping. Reproduce from the repository root with NumPy, SciPy and Numba installed.

    python poc/rotated_midrange_windows/poc/search.py poc/pell_corrugation/data/border-scan/merged.csv poc/rotated_midrange_windows/data/scan.json candidates.csv
