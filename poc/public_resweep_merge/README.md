# Public re-sweep merge (Nov 11 2026)

Union of `../public_resweep_kernels` (all Kaggle kernel outputs) and `../public_resweep_other` (datasets, gallery, catalogs, GitHub, sigmaborov), best pandas-stable candidate per n, swapped into `border-scan/merged.csv`.

- 315 n changed, each strictly better on the official evaluator; none worse. n=919 tied exactly and was left as the incumbent.
- Official `squares score`: 22.199718022945788 -> 22.199067547604685, 1000/1000 accepted, literal CSV and after pandas read_csv/to_csv (files are byte-identical).
- C++ certifier (`poc/cuda-loop/cuda/certifier`): 22.199067547604674 literal and with `--roundtrip`, 1000/1000.
- Most gains are tiny (~1e-16 in L); the largest are n=690 (4.4e-3), 543 (3.1e-3), 590 (2.5e-3), 756, 453, 928, 654.

`poc/build.py` builds the file, `docs/chosen.json` has the chosen source per n, `docs/*.json` hold the check reports. Original authors retain credit for the packings; data only, never executed.
