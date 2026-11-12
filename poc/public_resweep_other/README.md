# Public re-sweep (datasets, gallery, catalogs, sigmaborov history)

Data only; nothing downloaded was executed. `data/` (about 12 GB: dataset zips, git clones, sigmaborov v3-v18 outputs, gallery) is gitignored.

Output: `out/wins_rows.csv` (303 n, merged.csv id convention) and `docs/wins.json` (per n: new_L, our_L, source, candidate counts). Each win is exact-evaluator strict (< merged.csv L) in both the literal-string and pandas round-trip reads. `docs/dropped_pandas_unstable.json`: 128 n whose only wins (<~1e-12) vanish after a pandas float round trip.

Pipeline (`poc/`): `kd*.py` dataset search, `dl.sh`/`extract.py`, `sigma.py` (ListKernelSessionOutput per version), `gal.py`, `scan.py`/`scan_ext.py`/`scan_npz.py` (CSV/JSON/cert/yaml/npz harvest, keep bbox < L+1e-9), `verify.py` (exact evaluate, bbox < L-1e-14), `dilate.py` and `canon.py` (minimal uniform center dilation, float64-canonical rows), `gallery_lean.py` (fixed-angle LP, slack ladder), `final.py`.
Gallery: no n with label L below ours; lean LP re-repair of 553 tilted n gave no win. Candidates in (L+1e-14, L+1e-9) were not repaired (repair cannot shrink them below L).
