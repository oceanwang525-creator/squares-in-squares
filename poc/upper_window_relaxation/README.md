# Donor windows and perturbations for n=167–200

Tested against the team incumbent at `2c1e6a6`. The target coordinate rows were identical to the earlier `cd17a65` file used when the donor run began. The measured searches found no candidate at least `1e-9` shorter than its incumbent, so no candidate CSV was generated.

The first search took windows from nearby larger incumbents at seven small rotations. It let the window start wider than the target, then moved and rotated the retained squares before compacting the layout. It tested 153 donor starts and separately ran a corner-contact sequential LP on 18 non-grid incumbents.

The second search ran 116 starts. It perturbed local patches or dispersed squares, inserted a square into the lower neighbor, or deleted a square from the upper neighbor. Each trial minimized squared separating-axis penetration and wall overflow at a fixed side before repair and compaction. [Trial counts](data/results.json)

Run from the repository root with Python 3.12, NumPy, SciPy, Numba and the dependencies of the baseline metric. `sqk` must be importable from the `chaydavs/squares-in-squares` checkout at `ddb32be` via `PYTHONPATH` or `SQUARE_PACKING_SQK`. Set `SQUARE_PACKING_OUTPUT` to a writable work directory for per-trial JSON and any candidate CSV.

```sh
python poc/upper_window_relaxation/poc/recrop.py
python poc/upper_window_relaxation/poc/restarts.py
```
