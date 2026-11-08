# Vacancy Deletion on Pell-17 Cores (POC 1)

Parametric perimeter tooth deletion on flank staircases of Pell-17 cores (`D = 17`).
Tests exact-count matching at discrete count deficits (`n = 908, 909, 543, 970`) and executes the fast falsification gate defined in `poc/research/docs/q5_beyond_atlas_and_plan.md`.

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
