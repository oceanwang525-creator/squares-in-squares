# Vacancy Deletion on Pell-17 Cores (POC 1)

Parametric perimeter tooth deletion on flank staircases of Pell-17 cores (`D = 17`).
Tests exact-count matching at discrete count deficits (`n = 908, 909, 543, 970`) and executes the fast falsification gate defined in `poc/research/docs/q5_beyond_atlas_and_plan.md`.

## Incumbent Comparison Ledger (merged.csv)

| Target n | Candidate L | Our Incumbent L (merged.csv) | Delta L | Gate Status |
|:---|---:|---:|---:|:---|
| 908 | 30.69092100 | 30.63702956 | +0.05389144 | Falsified |
| 909 | 30.69092100 | 30.65685425 | +0.03406675 | Falsified |
| 543 | 23.82842712 | 23.80242872 | +0.02599840 | Falsified |
| 970 | 31.69092100 | 31.65685425 | +0.03406675 | Falsified |

## Execution

Run from repository root with shared squares package installed:

```sh
# Run Fast Falsification Gate
python3 poc/vacancy_deletion/poc/vacancy_deletion.py --test-falsification

# Generate target suite and official transport certification
python3 poc/vacancy_deletion/poc/vacancy_deletion.py --generate --iterations 25

# Run unit tests
python3 -m unittest poc/vacancy_deletion/poc/test_vacancy_deletion.py
```

See [measured report](docs/vacancy_deletion.md).
