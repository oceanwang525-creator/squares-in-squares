
# Vacancy Deletion and Defect Propagation

Vacancy-propagated staircase and band relaxation POC suite for large-n square packings.
Addresses discrete capacity quantization in algebraic staircase bands by creating local vacancies that propagate through contact graphs, allowing continuous container shrinkage.

- **POC 1**: Flank-tooth deletion on synthetic cores.
- **POC 2**: Band-column end deletion and vacancy collapse.
- **POC 3**: Defect propagation on the public $n = 972$ incumbent ($w = 13, t = 4$ Pell-17 core).

## POC 3: Public $n = 972$ Defect Propagation

The certified public incumbent at $n = 972$ is an uncorrugated $(w = 13, t = 4)$ Pell-17 packing jammed at $L_{45} + 4 = 31.692388155428$. It partitions into 739 upright squares and 233 tilted squares arranged in 13 strips: $(13, 13, 21 \times 8, 13, 13, 13)$.

Deleting $m \in \{1, 2\}$ squares eliminates the longitudinal jamming condition along the target column, allowing adjacent columns to shift into the cavity and relaxing the boundary through a 4-sign-piece container shrink LP to target $n = 971$ and $n = 970$.

## Incumbent Comparison Ledger (merged.csv)

| Target n | Candidate L | Our Incumbent L (merged.csv) | Delta L | Gate Status |
|:---|---:|---:|---:|:---|
| 970 | 31.69238816 | 31.65685425 | +0.03553391 | Falsified |
| 971 | 31.69238816 | 31.67877988 | +0.01360828 | Falsified |

### Usage

Run from the repository root after installing the shared baseline package:

```sh
python poc/vacancy_deletion/poc/defect_propagation_972.py --help
```

See `docs/defect_propagation_972.md` for detailed mathematical derivations and results.
