#!/usr/bin/env python3
"""Update a rows-only final.csv with newly gated complete puzzle groups."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "poc" / "claude-midn" / "poc"))
import rows as rowio  # noqa: E402


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: update_final.py FINAL_ROWS NEW_GATED_ROWS OUT_ROWS")
    base_path, update_path, out_path = map(Path, sys.argv[1:])
    groups = rowio.read_groups(base_path)
    updates = rowio.read_groups(update_path)
    replaced = []
    for n, candidate in updates.items():
        if n not in groups:
            groups[n] = candidate
            replaced.append(n)
            continue
        old = rowio.both(n, groups[n])
        new = rowio.both(n, candidate)
        if new is None:
            raise SystemExit(f"n={n}: new rows rejected by the official metric")
        if new[0] < old[0] and new[1] < old[1]:
            groups[n] = candidate
            replaced.append(n)
    rowio.write_rows(out_path, groups)
    print(
        f"wrote {len(groups)} complete n groups; "
        f"replaced={len(replaced)} {replaced}"
    )


if __name__ == "__main__":
    main()
