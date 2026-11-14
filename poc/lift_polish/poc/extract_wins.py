#!/usr/bin/env python3
"""Normalise optimiser outputs into the `wins.json` shape {n: {"rows": [...]}}.

Handles both
  * `{n: {"rows": [...]}}`            (alln_fixed_lp, lip_slp)
  * `[{...,"status":..,"rows":[..]}]` (swarm_lpslack results list)
and only keeps entries that actually carry rows.

usage: extract_wins.py OUT.json IN.json [IN2.json ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def harvest(obj) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if isinstance(obj, dict):
        for key, entry in obj.items():
            if isinstance(entry, dict) and isinstance(entry.get("rows"), list):
                out[key] = entry
    elif isinstance(obj, list):
        for entry in obj:
            if isinstance(entry, dict) and isinstance(entry.get("rows"), list):
                out[str(entry["n"])] = entry
    return out


def main() -> None:
    dst = Path(sys.argv[1])
    merged: dict[str, dict] = {}
    for src in sys.argv[2:]:
        obj = json.loads(Path(src).read_text(encoding="utf-8"))
        got = harvest(obj)
        print(f"{src}: {len(got)} row-bearing entries {sorted(map(int, got))}")
        for key, entry in got.items():
            if key not in merged or entry.get("new_L", "9" * 9) < merged[key].get("new_L", "9" * 9):
                merged[key] = entry
    dst.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    print(f"wrote {dst}: {len(merged)} entries")


if __name__ == "__main__":
    main()
