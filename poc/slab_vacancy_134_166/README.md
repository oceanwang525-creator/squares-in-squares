# Slab transfer and vacancy search, n=134–166

Search against the team portfolio at `2c1e6a6`. Most cases in this range already
fit an integer square. The search starts from tilted layouts, moves intact rows
and columns between nearby problem sizes, or makes vacancies in a slightly
larger layout and then minimizes its side.

The script searches five families:

| Mode | Seed layouts and refinement |
| --- | --- |
| `compact` | Fixed-angle LP followed by corner-based sequential LP for incumbent and near-incumbent layouts. |
| `cuts` | Remove one vertical and one horizontal slab of width 0.94, 1.00 or 1.06, or insert two unit slabs. Enumerate cuts between center events, sample up to 48 per donor and target, rank by overlap and side excess, and repair the best 12 per target. |
| `deletion` | Delete one or two squares from layouts of size n+1 or n+2. Prioritize near-wall squares and test up to 75 variants per target with LP, angular perturbations and corner-based refinement. |
| `diamond` | Delete one square from the n=148 Goebel layout and rotate its diamond through six orientations before penalty relaxation and LP repair. Test one deletion per symmetry class to target n=147. |
| `insertion` | Insert one square into the n=146 incumbent. Sample a regular grid and Voronoi vertices at twelve angles, use the seven lowest-overlap separated centers per angle, then run penalty relaxation and LP repair. |

The geometry and fixed-angle LP come from [sqk at
`ddb32bed`](https://github.com/chaydavs/squares-in-squares/tree/ddb32bed3f3f1d34c2ab0067684504a4677bc509).
The final check uses the official evaluator vendored in `poc/baseline`, both on
literal CSV decimals and after the default pandas 3.1.0 read/write round trip.
The side must improve by more than `1e-9` in both evaluations.

Install NumPy, SciPy, pandas 3.1.0 and `./poc/baseline/poc`. Add the `sqk`
checkout to `PYTHONPATH`, then run from the repository root:

```sh
for mode in compact cuts deletion diamond insertion; do
  python poc/slab_vacancy_134_166/poc/search.py "$mode" --output "/tmp/squares-$mode"
done
```

The reported searches also supplied the local `final_polished.csv` and
`pubnext/final_composed.csv` portfolios as optional `--donor` inputs to the
first three modes. The two n=147 modes use only `merged.csv`. All 1,079 trials
finished with no strict improvement. See [results by method](data/results.json).
A candidate CSV is written only after a verified gain.
