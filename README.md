# squares-in-squares

Square-packing research, organized by approach. Every approach has a short README, Python under `poc/`, reports under `docs/`, and CUDA/build sources under `cuda/` only where used.

| Approach | What it contains |
| --- | --- |
| [baseline](poc/baseline/README.md) | Shared official-metric package, CLI, small-n search and CPU tests |
| [cuda-loop](poc/cuda-loop/README.md) | Main packing/seeded kernels, C++ certifier, families and persistent search loop |
| [kat-algos](poc/kat-algos/README.md) | fp64 compression/basin-hopping portfolio and budget comparisons |
| [kat-band](poc/kat-band/README.md) | Thick bands, FTMC, local ruin/rebuild and portable pilot bundles |
| [gemini](poc/gemini/README.md) | Separate experimental Gemini CUDA kernels and driver |
| [diamond_core](poc/diamond_core/README.md) | Diamond-core construction |
| [fault_cut](poc/fault_cut/README.md) | Fault-cut construction |
| [shelf_knapsack](poc/shelf_knapsack/README.md) | Shelf/knapsack construction |
| [hinged_band](poc/hinged_band/README.md) | Independently hinged split-angle bands |
| [boundary_zipper](poc/boundary_zipper/README.md) | Boundary-connected seam insertion MILP |
| [research](poc/research/README.md) | [Argon brief](poc/research/docs/ARGON_BRIEF.md) and cross-approach failure ledger |

## Install and run

All commands below run from the repository root. Python 3.12+ is required.

```sh
cat poc/.gitignore >> .git/info/exclude
python -m venv .venv
. .venv/bin/activate
python -m pip install -e './poc/baseline/poc[dev,optimize,records]'
squares baseline --max-n 5 --output outputs/small.csv
squares validate outputs/small.csv --max-n 5 --require-complete
```

The exclude step keeps generated files and secrets out of Git without adding a tracked root dotfile.

The shared package is installed once; standalone approach scripts can then import `squares`. Gemini also imports host helpers from the main CUDA-loop approach. Kat-band's portable pilot bundle remains self-contained and flat internally.

## CPU checks

```sh
python -m pytest -q -c poc/baseline/poc/pyproject.toml
python -m compileall -q poc
python -m build poc/baseline/poc
make -C poc/cuda-loop/cuda certifier families_cli
```

The full kat-band tests additionally require PyTorch. Construction transport tests require pandas 3.1.0, matching their recorded certification environment. Do not silently skip these dependencies when claiming a full-suite pass.

## CUDA builds

```sh
make -C poc/cuda-loop/cuda all ARCH=sm_80
make -C poc/kat-algos/cuda all ARCH=sm_80
make -C poc/gemini/cuda all ARCH=sm_80
```

Choose the architecture for your actual GPU. CUDA runtime results are not implied by a successful CPU build or a Makefile dry run. No automatic Kaggle submission is performed.

## Validation and results

The unmodified official metric and its separate license live in [the shared package](poc/baseline/poc/vendor/NOTICE). Proposed improvements must pass literal-decimal and actual pandas transport certification. Historical results are preserved under the relevant approach, not relabeled as new results. The hinged-band and boundary-zipper POCs produced valid but worse outputs, with zero wins.

Project code is [MIT licensed](LICENSE). See [detailed baseline usage](poc/baseline/docs/USAGE.md) and each approach's docs for provenance and limits.

## Contributing results

Every added or changed `poc/{name}` folder in a PR must contain `final.csv` in the PR head, even when that CSV itself did not change. Use the exact header `id,x,y,deg`. A header-only file is the correct result when the POC has no certified strict wins. Fully deleted POC folders are exempt.

Nonempty finals contain only complete n groups that strictly reduce L against fresh main in BOTH the unchanged official literal-decimal check and actual pandas read_csv/to_csv transport certification. Do not publish incumbent snapshots, ties, regressions, or incomplete groups as final results. A final may contain just the n values won by that POC, not all 1000. IDs use `001_0` through `1000_999` as applicable; decimals may use an s prefix to preserve precision through pandas.

The PR workflow checks every touched POC's final structure using trusted base code and a read-only token. It never executes PR code. This is NOT geometry certification or proof of improvement. Before merging, separately compare against fresh main, certify literal and actual pandas transport, and retain only strict wins in both modes. Record source paths, incumbent revision/hash, certification environment and results in the POC's provenance. Refresh those checks if main changes.

Historical finals are not automatically rewritten by this policy change. The union independently certifies candidates and only selects dual-mode strict improvements; a tie in a historical file cannot displace the incumbent.
