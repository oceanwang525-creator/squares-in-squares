# Minimum square enclosure for archived midrange layouts

Search the entire rotation range of each packing for n=100–200. Compute the
convex hull of all square vertices. Between consecutive hull events, the width
and height are positive concave sinusoids; it suffices to evaluate the events
and the crossing where width equals height. This is a rigid whole-packing
rotation rather than a donor crop or a change to the contact graph.

The sweep read 113 CSVs, of which 46 contained midrange coordinates, and
evaluated 793 distinct layouts. None beat the `2c1e6a6` incumbent by the
`1e-10` side filter. See [scan data](data/scan.json).

```sh
python poc/midrange_hull_rotation/poc/search.py \
  poc/pell_corrugation/data/border-scan/merged.csv /tmp/hull-scan [ARCHIVE ...]
```

Requires NumPy and SciPy. Each archive argument can be a CSV or a directory.
Candidate geometry, if found, still requires official validation. The
[validation script](poc/certify.py) checks n=100–200 independently with the
official metric on literal and pandas 3.1.0 input and can verify candidate
rows in the full portfolio against the upstream proof.
