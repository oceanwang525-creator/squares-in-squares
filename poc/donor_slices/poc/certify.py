"""Literal and pandas-transport certification for emitted donor-slice CSVs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from squares.core import certify, read_submission


def certify_file(path: Path, n: int) -> dict:
    literal = certify(n, read_submission(path)[n])
    pandas_text = pd.read_csv(path).to_csv(index=False)
    pandas_path = path.with_name(path.stem + ".pandas.csv")
    pandas_path.write_text(pandas_text, encoding="utf-8")
    transported = certify(n, read_submission(pandas_path)[n])
    return {
        "literal_L": literal,
        "pandas_L": transported,
        "pandas_path": pandas_path.name,
        "pandas_version": pd.__version__,
        "literal_ok": True,
        "pandas_ok": True,
        "same_decimal": literal == transported,
    }
