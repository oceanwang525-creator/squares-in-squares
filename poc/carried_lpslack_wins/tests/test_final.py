from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "check_final_csv", REPO / ".github" / "scripts" / "check_final_csv.py"
)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def test_header_and_complete_groups() -> None:
    text = (REPO / "poc" / "carried_lpslack_wins" / "final.csv").read_text(
        encoding="utf-8"
    )
    CHECK.validate(text)
    rows = list(csv.DictReader(text.splitlines()))
    groups = {}
    for row in rows:
        n, square_id = map(int, row["id"].split("_"))
        groups.setdefault(n, set()).add(square_id)
    assert len(groups) == 69
    assert {740, 861}.isdisjoint(groups)
    for n, ids in groups.items():
        assert ids == set(range(n))
