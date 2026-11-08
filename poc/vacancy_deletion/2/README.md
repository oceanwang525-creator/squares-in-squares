# Vacancy Deletion POC 2: Band-Column End Deletion and Vacancy Collapse

## Brief Summary & Result
Tests vacancy collapse by deleting 1–2 squares directly from diagonal band columns (outermost columns $c=0, 24$ or central pinch columns) on the certified $n_0 = 739$ Pell-17 core, followed by L-border wrapping ($t=3$ for $n=909, 908$ and $t=4$ for $n=971$) and 4-sign-piece epigraph LP polish.

**Result: Falsification Gate Failed ($\Delta L > 0$).**
- Target $n=909$: polished $L \approx 30.690921$, incumbent $L = 30.65685425$ ($\Delta L = +0.034067$).
- Target $n=971$: polished $L \approx 31.690921$, incumbent $L = 31.67877988$ ($\Delta L = +0.012141$).
- **Geometric Blocker**: In the uniform 25-column Pell-17 core, deleting a square from column $c$ relieves longitudinal jamming for that single column, but the other 24 intact columns still span rigidly between flank staircases $a_1$ and $a_2$. The container side $L$ is held by the intact columns, preventing the flank staircases from collapsing inward.

## Code Structure
- `poc/band_vacancy.py`: Core deletion generator, vacancy collapse seeding, L-border wrapper, and 4-sign-piece LP evaluation.
- `poc/test_band_vacancy.py`: Pytest/unittest suite testing column deletion counts, border wrapping counts, and gate falsification assertion.
- `docs/band_vacancy.md`: Detailed mathematical documentation, experimental results table, and geometric blocker analysis.

## Running Tests
Run with the shared environment:
```sh
/Users/lucaszhang/.local/bin/uv run --python /opt/homebrew/bin/python3 --with 'pandas,scipy,mpmath,pytest' pytest poc/vacancy_deletion/2/poc/
```
